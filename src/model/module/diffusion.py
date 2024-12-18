import torch
import torch.nn as nn

from .nn.dit import DiTBlock
from .nn.feedforward import MLP


class DiffusionModel(nn.Module):
    def __init__(self, config):
        super().__init__()

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

        self.emb_c = MLP(config.n_clusters, out_dim=config.latent_dim)
        self.layer = nn.ModuleList(
            [
                DiTBlock(config.latent_dim, config.nheads)
                for _ in range(config.n_ditblocks)
            ]
        )

    def pos_encoding(self, t):
        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, self.latent_dim, 2).float() / self.latent_dim)
        ).to(t.device)
        pos_enc_a = torch.sin(t.repeat(1, self.latent_dim // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, self.latent_dim // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc.to(t.device)

    def forward(self, x, t, c):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t)

        c = self.emb_c(c)
        c = t + c

        for layer in self.layer:
            x = layer(x, c)

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
        z_t, noise = self.sample_noise(z, t)
        predicted_noise = self(z_t, t, c)
        return predicted_noise, noise

    def sample(self, c, cfg_scale=3):
        b = c.size(0)
        z = torch.randn((b, self.latent_dim, self.latelt_size[0], self.latent_size[1]))
        z = z.to(c.device)

        for i in reversed(range(1, self.noise_steps)):
            t = torch.full(b, i).long().to(c.device)
            predicted_noise = self(z, t, c)

            if cfg_scale > 0:
                uncond_predicted_noise = self(z, t, None)
                predicted_noise = torch.lerp(
                    uncond_predicted_noise, predicted_noise, cfg_scale
                )
            alpha = self.alpha[t][:, None, None, None]
            alpha_hat = self.alpha_hat[t][:, None, None, None]
            beta = self.beta[t][:, None, None, None]
            if i > 1:
                noise = torch.randn_like(z)
            else:
                noise = torch.zeros_like(z)
            z = 1 / torch.sqrt(alpha) * (
                z - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise
            ) + torch.sqrt(beta) * noise.to(c.device)

        return z
