import torch
import torch.nn as nn


class Embedding(nn.Module):
    def __init__(self, seq_len, hidden_ndim, latent_ndim):
        super().__init__()
        self.hidden_ndim = hidden_ndim

        self.conv_x = nn.Sequential(
            nn.Conv1d(seq_len, hidden_ndim, 1, bias=False),
            nn.SiLU(),
            nn.Conv1d(hidden_ndim, hidden_ndim, 1, bias=False),
            nn.SiLU(),
            nn.Conv1d(hidden_ndim, latent_ndim, 1, bias=False),
            nn.SiLU(),
        )
        self.conv_y = nn.Sequential(
            nn.Conv1d(seq_len, hidden_ndim, 1, bias=False),
            nn.SiLU(),
            nn.Conv1d(hidden_ndim, hidden_ndim, 1, bias=False),
            nn.SiLU(),
            nn.Conv1d(hidden_ndim, latent_ndim, 1, bias=False),
            nn.SiLU(),
        )

    def forward(self, x):
        x, y = x[:, :, :, 0], x[:, :, :, 1]
        x = self.conv_x(x)  # (b, latent_ndim, n_pts)
        y = self.conv_y(y)  # (b, latent_ndim, n_pts)
        x = torch.cat([x, y], dim=2)
        x = x.permute(0, 2, 1)
        # x (b, n_pts * 2, latent_ndim)

        return x
