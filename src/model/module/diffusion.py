import torch
import torch.nn as nn
from rotary_embedding_torch import RotaryEmbedding
from tqdm import tqdm

from .nn.dit import DiTBlock, FinalLayer


class DiffusionModule(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.latent_dim = config.latent_dim
        self.latent_size = config.latent_size
        self.noise_steps = config.noise_steps

        self.beta = torch.linspace(config.beta_start, config.beta_end, self.noise_steps)
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        self.gamma_hat = self.gen_gamma_hat(self.noise_steps)

        self.model = DiffusionModel(config)

    def send_sigma_to_device(self, device):
        if device != self.beta.device:
            self.beta = self.beta.to(device)
            self.alpha = self.alpha.to(device)
            self.alpha_hat = self.alpha_hat.to(device)
            self.gamma_hat = self.gamma_hat.to(device)

    def gen_gamma_hat(self, noise_steps):
        gammas = []
        for t in range(noise_steps):
            alpha_hat_t_inv = torch.cumprod(
                torch.flip(self.alpha[:t], dims=(0,)), dim=0
            )
            gamma = torch.sum(torch.sqrt(alpha_hat_t_inv))
            gammas.append(gamma)
        gammas = torch.tensor(gammas)
        return gammas / gammas[-1]

    def forward(self, zq, t, c):
        return self.model(zq, t, c)

    def sample_timesteps(self, n):
        return torch.randint(1, self.noise_steps, (n,))

    def sample_noise(self, zq, t, mu_c):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None]
        mu_t = self.gamma_hat[t][:, None, None] * mu_c
        eps = torch.randn_like(zq)
        return sqrt_alpha_hat * zq + sqrt_one_minus_alpha_hat * eps + mu_t, eps + mu_t

    def train_step(self, zq, c_probs, mu_c):
        # zq (b, n, dim)
        # c (b,)
        t = self.sample_timesteps(zq.size(0)).to(zq.device)
        zq_t, noise = self.sample_noise(zq, t, mu_c)
        pred_noise = self(zq_t, t, c_probs)

        # samplig from x1
        t1 = torch.ones_like(t)
        zq_1, noise_1 = self.sample_noise(zq, t1, mu_c)
        pred_noise_1 = self(zq_1, t1, c_probs)
        beta = self.beta[t1][:, None, None]
        alpha = self.alpha[t1][:, None, None]
        alpha_hat = self.alpha_hat[t1][:, None, None]
        pred_zq_0 = (
            zq_1 - beta / torch.sqrt(1 - alpha_hat) * pred_noise_1
        ) / torch.sqrt(alpha)
        return pred_noise, noise, pred_zq_0

    @torch.no_grad()
    def sample(self, c_probs, mu_c, is_hard_mu_c=True):
        self.send_sigma_to_device(c_probs.device)

        b = c_probs.size(0)

        if not is_hard_mu_c:
            mu_c = torch.cat([m.unsqueeze(0) for m in mu_c], dim=0)
            mu_c = mu_c.unsqueeze(0)
            mu_c = torch.sum(
                mu_c * c_probs.view(b, self.n_clusters, 1, 1), dim=1
            )  # (b, npts, ndim)
        else:
            mu_c = torch.cat(
                [mu_c[c].unsqueeze(0) for c in c_probs.argmax(dim=-1)], dim=0
            )

        zq = torch.randn(
            (b, self.latent_size[0] * self.latent_size[1], self.latent_dim)
        )
        zq = zq.to(c_probs.device)
        zq = zq + mu_c

        for i in tqdm(list(reversed(range(1, self.noise_steps)))):
            t = torch.full((b,), i).long().to(c_probs.device)
            pred_noise = self(zq, t, c_probs)

            beta = self.beta[t][:, None, None]
            alpha = self.alpha[t][:, None, None]
            alpha_hat = self.alpha_hat[t][:, None, None]
            if i > 1:
                noise = torch.randn_like(zq)
            else:
                noise = torch.zeros_like(zq)
            zq = (zq - beta / torch.sqrt(1 - alpha_hat) * pred_noise) / torch.sqrt(
                alpha
            ) + torch.sqrt(beta) * noise

        return zq


class DiffusionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dim = config.latent_dim * 4
        self.latent_dim = config.latent_dim

        self.emb_x = nn.Linear(self.latent_dim, self.dim)
        self.rotary_emb = RotaryEmbedding(self.dim)
        self.emb_c = nn.Linear(config.n_clusters, self.dim)
        self.blocks = nn.ModuleList(
            [DiTBlock(self.dim, config.nheads_dit) for _ in range(config.n_ditblocks)]
        )
        self.fin = FinalLayer(self.dim, self.latent_dim)

    def pos_encoding(self, t):
        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, self.dim, 2).float() / self.dim)
        ).to(t.device)
        pos_enc_a = torch.sin(t.repeat(1, self.dim // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, self.dim // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc.to(t.device)

    def forward(self, x, t, c):
        x = self.emb_x(x)
        x = self.rotary_emb.rotate_queries_or_keys(x, seq_dim=1)

        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t)
        c = self.emb_c(c)
        c = c + t

        for block in self.blocks:
            x = block(x, c)
        x = self.fin(x, c)

        return x
