import torch
import torch.nn as nn
from rotary_embedding_torch import RotaryEmbedding
from tqdm import tqdm

from .nn.dit import DiTBlock, FinalLayer


class DiffusionModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.dim = config.latent_dim * 4
        self.latent_dim = config.latent_dim
        self.latent_size = config.latent_size
        self.noise_steps = config.noise_steps

        self.beta = nn.Parameter(
            torch.linspace(config.beta_start, config.beta_end, self.noise_steps),
            requires_grad=False,
        )
        self.alpha = nn.Parameter(1.0 - self.beta, requires_grad=False)
        self.alpha_hat = nn.Parameter(
            torch.cumprod(self.alpha, dim=0), requires_grad=False
        )

        self.emb_x = nn.Linear(config.latent_dim, self.dim)
        self.rotary_emb = RotaryEmbedding(self.dim)
        self.emb_c = nn.Linear(config.n_clusters, self.dim)
        self.blocks = nn.ModuleList(
            [DiTBlock(self.dim, config.nheads) for _ in range(config.n_ditblocks)]
        )
        self.fin = FinalLayer(self.dim, config.latent_dim)

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

    def sample_noise(self, z, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None]
        eps = torch.randn_like(z)
        return sqrt_alpha_hat * z + sqrt_one_minus_alpha_hat * eps, eps

    def sample_timesteps(self, n):
        return torch.randint(1, self.noise_steps, (n,))

    def train_step(self, z, c):
        # z (b, n, h)
        # c (b, n_clusters)
        t = self.sample_timesteps(z.size(0)).to(z.device)
        zt, noise = self.sample_noise(z, t)
        predicted_noise = self(zt, t, c)

        # samplig from z1
        t1 = torch.ones_like(t)
        z1, noise_1 = self.sample_noise(z, t1)
        predicted_noise_1 = self(z1, t1, c)
        beta = self.beta[t1][:, None, None]
        alpha = self.alpha[t1][:, None, None]
        alpha_hat = self.alpha_hat[t1][:, None, None]
        predicted_z = (
            z1 - (beta / (torch.sqrt(1 - alpha_hat))) * predicted_noise_1
        ) / torch.sqrt(alpha)
        return predicted_noise, noise, predicted_noise_1, noise_1, predicted_z

    @torch.no_grad()
    def sample(self, c):
        b = c.size(0)
        z = torch.randn((b, self.latent_size[0] * self.latent_size[1], self.latent_dim))
        z = z.to(c.device)

        for i in tqdm(list(reversed(range(1, self.noise_steps)))):
            t = torch.full((b,), i).long().to(c.device)
            pred_noise = self(z, t, c)

            alpha = self.alpha[t][:, None, None]
            alpha_hat = self.alpha_hat[t][:, None, None]
            beta = self.beta[t][:, None, None]
            # if i > 1:
            #     noise = torch.randn_like(z)
            # else:
            #     noise = torch.zeros_like(z)
            z = (
                (z - (beta / (torch.sqrt(1 - alpha_hat))) * pred_noise)
                / torch.sqrt(alpha)
                # + torch.sqrt(beta) * noise
            )

        return z
