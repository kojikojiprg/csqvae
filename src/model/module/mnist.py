from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from rotary_embedding_torch import RotaryEmbedding

from .nn.feedforward import MLP
from .nn.resblock import ResidualBlock
from .nn.transformer import TransformerEncoderBlock


class ClassificationHead(nn.Module):
    def __init__(self, config: SimpleNamespace):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.latent_dim))

        self.pe = RotaryEmbedding(config.latent_dim, learned_freq=False)
        self.tre = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    config.latent_dim, config.nheads, config.dropout
                )
                for _ in range(config.nlayers)
            ]
        )

        self.mlp = MLP(config.latent_dim, out_dim=config.n_clusters)

        self.temperature = None
        log_param_q_cls = np.log(config.param_q_init_cls)
        self.log_param_q_cls = nn.Parameter(
            torch.tensor(log_param_q_cls, dtype=torch.float32)
        )

    def forward(self, z):
        b = z.size(0)
        z = torch.cat([self.cls_token.repeat(b, 1, 1), z], dim=1)

        z = self.pe.rotate_queries_or_keys(z, seq_dim=1)

        for encoder in self.tre:
            z = encoder(z)

        logits = self.mlp(z[:, 0, :])

        param_q = self.log_param_q_cls.exp()
        precision_q = 0.5 / torch.clamp(param_q, min=1e-10)
        logits = logits * precision_q
        return logits


class Encoder(nn.Module):
    def __init__(self, config: SimpleNamespace):
        super().__init__()

        latent_dim = config.latent_dim
        self.latent_dim = config.latent_dim
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, latent_dim // 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(latent_dim // 2),
            nn.ReLU(),
        )
        self.conv2 = nn.Conv2d(latent_dim // 2, latent_dim, 4, 2, 1, bias=False)

        self.res_blocks = nn.ModuleList(
            [ResidualBlock(latent_dim) for _ in range(config.n_resblocks)]
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        for res_block in self.res_blocks:
            x = res_block(x)

        return x


class Decoder(nn.Module):
    def __init__(self, config: SimpleNamespace):
        super().__init__()

        latent_dim = config.latent_dim
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(latent_dim) for _ in range(config.n_resblocks)]
        )

        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, latent_dim // 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(latent_dim // 2),
            nn.ReLU(),
        )
        self.conv2 = nn.ConvTranspose2d(latent_dim // 2, 1, 4, 2, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        for res_block in self.res_blocks:
            x = res_block(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sigmoid(x)
        return x
