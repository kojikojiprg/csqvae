import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, nch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(nch, nch, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(nch),
            nn.ReLU(),
            nn.Conv2d(nch, nch, (1, 1), bias=False),
            nn.BatchNorm2d(nch),
        )

    def forward(self, x):
        x = self.conv(x) + x
        return x
