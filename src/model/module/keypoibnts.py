from types import SimpleNamespace

import torch
import torch.nn as nn
from rotary_embedding_torch import RotaryEmbedding

from .nn import MLP, TransformerDecoderBlock, TransformerEncoderBlock


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


class ClassificationHead(nn.Module):
    def __init__(self, config: SimpleNamespace):
        super().__init__()
        ndim = config.latent_ndim
        self.conv = nn.Sequential(
            nn.Conv1d(ndim, ndim * 2, 1, bias=False),
            nn.SiLU(),
            nn.AvgPool1d(2),  # 38 -> 19
            nn.Conv1d(ndim * 2, ndim * 4, 1, bias=False),
            nn.SiLU(),
            nn.AvgPool1d(3, 2),  # 19 -> 9
            nn.Conv1d(ndim * 4, ndim * 8, 1, bias=False),
            nn.SiLU(),
            nn.AvgPool1d(3, 2),  # 9 -> 4
        )

        self.mlp = MLP(4 * ndim * 8, config.n_clusters)

    def forward(self, x):
        # x (b, n_pts, latent_ndim)
        x = x.permute(0, 2, 1)
        x = self.conv(x)  # (b, ndim, 4)
        x = x.view(x.size(0), -1)
        x = self.mlp(x)  # (b, n_clusters)
        return x


class Encoder(nn.Module):
    def __init__(self, config: SimpleNamespace):
        super().__init__()
        self.hidden_ndim = config.hidden_ndim

        self.emb_vis = Embedding(config.seq_len, config.hidden_ndim, config.latent_ndim)
        self.emb_spc = Embedding(config.seq_len, config.hidden_ndim, config.latent_ndim)
        self.pe = RotaryEmbedding(config.latent_ndim, learned_freq=True)
        self.encoders = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    config.latent_ndim, config.nheads, config.dropout
                )
                for _ in range(config.nlayers)
            ]
        )
        self.cls_head = ClassificationHead(config)

    def forward(self, x_vis, x_spc, mask=None):
        # x_vis (b, seq_len, 17, 2)
        # x_spc (b, seq_len, 2, 2)

        # embedding
        x_vis = self.emb_vis(x_vis)  # (b, 17 * 2, latent_ndim)
        x_spc = self.emb_spc(x_spc)  # (b, 2 * 2, latent_ndim)
        z = torch.cat([x_vis, x_spc], dim=1)
        # z (b, (17 + 2) * 2, latent_ndim)

        # positional embedding
        z = self.pe.rotate_queries_or_keys(z, seq_dim=1)

        for layer in self.encoders:
            z, attn_w = layer(z, mask)
        # z (b, (17 + 2) * 2, latent_ndim)

        c_logits = self.cls_head(z)  # (b, n_clusters)

        return z, c_logits


class Decoder(nn.Module):
    def __init__(self, config: SimpleNamespace):
        super().__init__()
        if not config.mask_leg:
            self.n_pts = (17 + 2) * 2
        else:  # mask ankles and knees
            self.n_pts = (17 - 4 + 2) * 2

        self.decoders = nn.ModuleList(
            [DecoderModule(self.config) for _ in range(self.n_pts)]
        )

    def forward(self, x_vis, x_spc, zq):
        b, seq_len = x_vis.size()[:2]
        x_vis = x_vis.view(b, seq_len, 17 * 2)
        recon_x_vis = torch.empty((b, seq_len, 0)).to(self.device)
        for i, decoder in enumerate(self.decoders[: 17 * 2]):
            recon_x = decoder(x_vis[:, :, i], zq[:, i, :])
            recon_x_vis = torch.cat([recon_x_vis, recon_x], dim=2)
        recon_x_vis = recon_x_vis.view(b, seq_len, 17, 2)

        x_spc = x_spc.view(b, seq_len, 2 * 2)
        recon_x_spc = torch.empty((b, seq_len, 0)).to(self.device)
        for i, decoder in enumerate(self.decoders[17 * 2 :]):
            recon_x = decoder(x_spc[:, :, i], zq[:, i, :])
            recon_x_spc = torch.cat([recon_x_spc, recon_x], dim=2)
        recon_x_spc = recon_x_spc.view(b, seq_len, 2, 2)

        return recon_x_vis, recon_x_spc


class DecoderModule(nn.Module):
    def __init__(self, config: SimpleNamespace):
        super().__init__()
        self.latent_ndim = config.latent_ndim

        self.x_start = nn.Parameter(
            torch.randn((1, 1, config.latent_ndim), dtype=torch.float32),
            requires_grad=True,
        )

        self.emb = MLP(1, config.latent_ndim)
        self.pe = RotaryEmbedding(config.latent_ndim, learned_freq=True)
        self.mlp_z = MLP(config.latent_ndim, config.latent_ndim * config.seq_len)
        self.decoders = nn.ModuleList(
            [
                TransformerDecoderBlock(
                    config.latent_ndim, config.nheads, config.dropout
                )
                for _ in range(config.nlayers)
            ]
        )
        self.mlp = nn.Sequential(
            MLP(config.latent_ndim, config.hidden_ndim),
            nn.SiLU(),
            MLP(config.hidden_ndim, 1),
            nn.Tanh(),
        )

    def forward(self, x, zq, mask=None):
        # x (b, seq_len)
        # zq (b, latent_ndim)

        b, seq_len = x.size()
        x = x.view(b, seq_len, 1)
        x = self.emb(x)  # (b, seq_len, latent_ndim)

        # concat start token
        x = torch.cat([self.x_start.repeat((b, 1, 1)), x], dim=1)
        x = x[:, :-1]  # (b, seq_len, latent_ndim)

        x = self.pe.rotate_queries_or_keys(x, seq_dim=1)

        zq = self.mlp_z(zq)
        zq = zq.view(b, seq_len, self.latent_ndim)
        for layer in self.decoders:
            x = layer(x, zq, mask)
        # x (b, seq_len, latent_ndim)

        recon_x = self.mlp(x).view(b, seq_len, 1)

        return recon_x
