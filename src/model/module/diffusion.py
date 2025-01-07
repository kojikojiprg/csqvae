import torch
import torch.nn as nn
from rotary_embedding_torch import RotaryEmbedding
from tqdm import tqdm

from .nn.dit import DiTBlock, FinalLayer


class DiffusionModule(nn.Module):
    def __init__(self, config, with_condition: bool):
        super().__init__()
        self.n_clusters = config.n_clusters
        self.latent_dim = config.latent_dim
        self.latent_size = config.latent_size
        self.noise_steps = config.noise_steps

        self.beta = torch.linspace(config.beta_start, config.beta_end, self.noise_steps)
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        self.gamma_hat = self.gen_gamma_hat(self.noise_steps)

        self.model = DiffusionModel(config, with_condition)

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

    def forward(self, z, t, c):
        return self.model(z, t, c)

    def sample_timesteps(self, n):
        return torch.randint(1, self.noise_steps, (n,))

    def sample_noise(self, z, t, mu_c, sigma_c):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None]

        if mu_c is None:
            eps = torch.randn_like(z)
            return sqrt_alpha_hat * z + sqrt_one_minus_alpha_hat * eps, eps
        else:
            mu_t = self.gamma_hat[t][:, None, None] * mu_c
            eps = torch.randn_like(z) * sigma_c[:, None, None]
            return (
                sqrt_alpha_hat * z + sqrt_one_minus_alpha_hat * eps + mu_t,
                eps + mu_t,
            )

    def train_step(self, z, c_probs, mu_c, sigma_c, is_c_onehot=False):
        # zq (b, n, dim)
        # c_probs (b, n_clusters)
        if is_c_onehot:
            c_probs = torch.eye(self.n_clusters, device=z.device)[
                c_probs.argmax(dim=-1)
            ]

        t = self.sample_timesteps(z.size(0)).to(z.device)
        zq_t, noise = self.sample_noise(z, t, mu_c, sigma_c)
        pred_noise = self(zq_t, t, c_probs)

        # samplig from x1
        t1 = torch.ones_like(t)
        zq_1, noise_1 = self.sample_noise(z, t1, mu_c, sigma_c)
        pred_noise_1 = self(zq_1, t1, c_probs)
        beta = self.beta[t1][:, None, None]
        alpha = self.alpha[t1][:, None, None]
        alpha_hat = self.alpha_hat[t1][:, None, None]
        pred_zq_0 = (
            zq_1 - beta / torch.sqrt(1 - alpha_hat) * pred_noise_1
        ) / torch.sqrt(alpha)
        return pred_noise, noise, pred_zq_0

    @torch.no_grad()
    def sample(self, c_probs, mu_c, sigma_c, is_hard_mu_c=True):
        self.send_sigma_to_device(c_probs.device)

        b = c_probs.size(0)

        z = (
            torch.randn((b, self.latent_size[0] * self.latent_size[1], self.latent_dim))
            * sigma_c[:, None, None]
        )
        z = z.to(c_probs.device)
        z = z + mu_c

        for i in tqdm(list(reversed(range(1, self.noise_steps)))):
            t = torch.full((b,), i).long().to(c_probs.device)
            pred_noise = self(z, t, c_probs)

            beta = self.beta[t][:, None, None]
            alpha = self.alpha[t][:, None, None]
            alpha_hat = self.alpha_hat[t][:, None, None]
            if i > 1:
                noise = torch.randn_like(z) * sigma_c[:, None, None]
            else:
                noise = torch.zeros_like(z)
            z = (z - beta / torch.sqrt(1 - alpha_hat) * pred_noise) / torch.sqrt(
                alpha
            ) + torch.sqrt(beta) * noise

        return z


class DiffusionModel(nn.Module):
    def __init__(self, config, with_condition):
        super().__init__()
        self.with_condition = with_condition
        self.dim = config.latent_dim_diffusion
        self.latent_dim = config.latent_dim

        self.emb_x = nn.Linear(self.latent_dim, self.dim)
        self.rotary_emb = RotaryEmbedding(self.dim)
        self.emb_c = nn.Linear(config.n_clusters, self.dim)
        self.blocks = nn.ModuleList(
            [DiTBlock(self.dim, config.n_heads_dit) for _ in range(config.n_blocks_dit)]
        )
        self.fin = FinalLayer(self.dim, self.latent_dim)

        if not self.with_condition:
            self.emb_c.requires_grad_(False)

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

        if self.with_condition:
            c = self.emb_c(c)
            c = c + t
        else:
            c = t

        for block in self.blocks:
            x = block(x, c)
        x = self.fin(x, c)

        return x
