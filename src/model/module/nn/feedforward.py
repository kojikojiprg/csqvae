import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = None,
        out_dim: int = None,
        act_layer: nn.Module = nn.SiLU(),
        dropout: float = 0.1,
    ):
        super().__init__()
        if out_dim is None:
            out_dim = in_dim
        if hidden_dim is None:
            hidden_dim = int(in_dim * 4 * (2 / 3))

        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = act_layer
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
