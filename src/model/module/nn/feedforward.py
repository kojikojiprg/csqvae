import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    def __init__(self, in_ndim: int, out_ndim: int = None):
        super().__init__()

        if out_ndim is None:
            out_ndim = in_ndim
        hdim = int(in_ndim * 4 * (2 / 3))
        self.w1 = nn.Linear(in_ndim, hdim)
        self.w2 = nn.Linear(hdim, out_ndim)
        self.w3 = nn.Linear(in_ndim, hdim)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MLP(nn.Module):
    def __init__(self, in_ndim: int, out_ndim: int = None, dropout: float = 0.1):
        super().__init__()
        if out_ndim is None:
            out_ndim = in_ndim
        hdim = int(in_ndim * 4 * (2 / 3))
        self.mlp = nn.Sequential(
            nn.Linear(in_ndim, hdim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hdim, out_ndim),
        )

    def forward(self, x):
        return self.mlp(x)
