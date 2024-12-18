import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self,
        in_ndim: int,
        hidden_ndim: int = None,
        out_ndim: int = None,
        act_layer: nn.Module = nn.SiLU(),
        dropout: float = 0.1,
    ):
        super().__init__()
        if out_ndim is None:
            out_ndim = in_ndim
        if hidden_ndim is None:
            hidden_ndim = int(in_ndim * 4 * (2 / 3))

        self.fc1 = (nn.Linear(in_ndim, hidden_ndim),)
        self.act = act_layer
        self.dropout = nn.Dropout(dropout)
        self.fc2 = (nn.Linear(hidden_ndim, out_ndim),)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
