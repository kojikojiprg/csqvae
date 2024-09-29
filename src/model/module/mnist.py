from types import SimpleNamespace

import torch.nn as nn

from .nn.resblock import ResidualBlock
from .nn.feedforward import MLP


class ClassificationHead(nn.Module):
    def __init__(self, config: SimpleNamespace):
        super().__init__()
        ndim = config.latent_ndim
        self.conv = nn.Sequential(
            nn.Conv2d(ndim, ndim * 2, 3, bias=False),
            nn.BatchNorm2d(ndim * 2),
            nn.ReLU(),
            nn.AvgPool2d(2, 1),
        )

        self.mlp = MLP(4 * 4 * ndim * 2, config.n_clusters)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.mlp(x)
        return x


class Encoder(nn.Module):
    def __init__(self, config: SimpleNamespace):
        super().__init__()

        latent_ndim = config.latent_ndim
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, latent_ndim // 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(latent_ndim // 2),
            nn.ReLU(),
        )
        self.conv2 = nn.Conv2d(latent_ndim // 2, latent_ndim, 4, 2, 1, bias=False)

        self.res_blocks = nn.ModuleList(
            [ResidualBlock(latent_ndim) for _ in range(config.n_resblocks)]
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

        latent_ndim = config.latent_ndim
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(latent_ndim) for _ in range(config.n_resblocks)]
        )

        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(latent_ndim, latent_ndim // 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(latent_ndim // 2),
            nn.ReLU(),
        )
        self.conv2 = nn.ConvTranspose2d(latent_ndim // 2, 1, 4, 2, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        for res_block in self.res_blocks:
            x = res_block(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sigmoid(x)
        return x
