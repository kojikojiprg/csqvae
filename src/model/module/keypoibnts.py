from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

from .nn.feedforward import MLP
from .nn.transformer import TransformerEncoderBlock


def get_n_pts(config: SimpleNamespace):
    if not config.mask_leg:
        n_pts = 17 + 2
    else:  # mask ankles and knees
        n_pts = 17 - 4 + 2
    return n_pts


class Embedding(nn.Module):
    def __init__(self, config, n_pts):
        super().__init__()
        self.seq_len = config.seq_len
        self.n_pts = n_pts
        self.latent_dim = config.latent_dim
        self.conv1 = nn.Sequential(
            nn.Conv1d(2, config.latent_dim // 8, 6, 3),
            nn.GroupNorm(1, config.latent_dim // 8),
            nn.SiLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(config.latent_dim // 8, config.latent_dim // 4, 5, 3),
            nn.GroupNorm(1, config.latent_dim // 4),
            nn.SiLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(config.latent_dim // 4, config.latent_dim // 2, 5, 2),
            nn.GroupNorm(1, config.latent_dim // 2),
            nn.SiLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(config.latent_dim // 2, config.latent_dim, 3, 1),
            nn.GroupNorm(1, config.latent_dim),
            nn.SiLU(),
        )

    def forward(self, x):
        # x (b, seq_len, n_pts, 2)
        x = x.permute(0, 2, 3, 1).contiguous()  # (b, n_pts, 2, seq_len)
        x = x.view(-1, 2, self.seq_len)  # (b * n_pts, 2, seq_len)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)  # (b * n_pts, ndim, 1)
        x = x.view(-1, self.n_pts, self.latent_dim)
        return x


class ClassificationHead(nn.Module):
    def __init__(self, config: SimpleNamespace):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.latent_dim))
        self.encoders = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    config.latent_dim, config.nheads, config.dropout
                )
                for _ in range(config.nlayers_cls)
            ]
        )
        self.mlp = MLP(config.latent_dim, config.n_clusters)

        log_param_q_cls = np.log(config.param_q_cls_init)
        self.log_param_q_cls = nn.Parameter(
            torch.tensor(log_param_q_cls, dtype=torch.float32)
        )

    def forward(self, z, is_train):
        # zq (b, n_pts, latent_dim)

        # concat cls_token
        cls_token = self.cls_token.repeat(z.size(0), 1, 1)
        z = torch.cat([cls_token, z], dim=1)
        # z (b, 1 + n_pts, latent_dim)

        if is_train:
            for layer in self.encoders:
                z, attn_w = layer(z)
            attn_w_tensor = None
        else:
            attn_w_lst = []
            for layer in self.encoders:
                z, attn_w = layer(z, need_weights=True)
                attn_w_lst.append(attn_w.unsqueeze(1))
            attn_w_tensor = torch.cat(attn_w_lst, dim=1)
        # z (b, 1 + n_pts, latent_dim)

        logits = self.mlp(z[:, 0, :])  # (b, n_clusters)

        param_q = self.log_param_q_cls.exp()
        precision_q_cls = 0.5 / torch.clamp(param_q, min=1e-10)
        logits = logits * precision_q_cls

        return logits, attn_w_tensor


class Encoder(nn.Module):
    def __init__(self, config: SimpleNamespace):
        super().__init__()
        self.latent_dim = config.latent_dim
        # self.emb = Embedding(config)
        self.emb_kps = Embedding(config, 13)
        self.emb_bbox = Embedding(config, 2)

        self.encoders = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    config.latent_dim, config.nheads, config.dropout
                )
                for _ in range(config.nlayers)
            ]
        )

    def forward(self, kps, bbox, is_train):
        # embedding
        kps = self.emb_kps(kps)
        bbox = self.emb_bbox(bbox)
        z = torch.cat([kps, bbox], dim=1)
        # z (b, n_pts, latent_dim)

        if is_train:
            for layer in self.encoders:
                z, attn_w = layer(z)
            attn_w_tensor = None
        else:
            attn_w_lst = []
            for layer in self.encoders:
                z, attn_w = layer(z, need_weights=True)
                attn_w_lst.append(attn_w.unsqueeze(1))
            attn_w_tensor = torch.cat(attn_w_lst, dim=1)
        # z (b, n_pts, latent_dim)

        return z, attn_w_tensor


class Reconstruction(nn.Module):
    def __init__(self, config, n_pts):
        super().__init__()
        self.seq_len = config.seq_len
        self.n_pts = n_pts
        self.latent_dim = config.latent_dim
        self.conv1 = nn.Sequential(
            nn.ConvTranspose1d(config.latent_dim, config.latent_dim // 2, 3, 1),
            nn.GroupNorm(1, config.latent_dim // 2),
            nn.SiLU(),
        )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose1d(config.latent_dim // 2, config.latent_dim // 4, 5, 2),
            nn.GroupNorm(1, config.latent_dim // 4),
            nn.SiLU(),
        )
        self.conv3 = nn.Sequential(
            nn.ConvTranspose1d(config.latent_dim // 4, config.latent_dim // 8, 5, 3),
            nn.GroupNorm(1, config.latent_dim // 8),
            nn.SiLU(),
        )
        self.conv4 = nn.Sequential(
            nn.ConvTranspose1d(config.latent_dim // 8, 2, 6, 3),
            # nn.GroupNorm(1, seq_len),
            nn.Tanh(),
        )

    def forward(self, zq):
        # zq (b, n_pts, ndim)
        zq = zq.reshape(-1, self.latent_dim, 1)  # (b * n_pts, ndim, 1)
        zq = self.conv1(zq)
        zq = self.conv2(zq)
        zq = self.conv3(zq)
        x = self.conv4(zq)
        # x (b * n_pts, 2, seq_len)
        x = x.view(-1, self.n_pts, 2, self.seq_len)
        x = x.permute(0, 3, 1, 2)
        # x (b, seq_len, n_pts, 2)
        return x


class Decoder(nn.Module):
    def __init__(self, config: SimpleNamespace):
        super().__init__()
        self.encoders = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    config.latent_dim, config.nheads, config.dropout
                )
                for _ in range(config.nlayers)
            ]
        )
        # self.recon = Reconstruction(config)
        self.recon_kps = Reconstruction(config, 13)
        self.recon_bbox = Reconstruction(config, 2)

    def forward(self, zq):
        # zq (b, n_pts, latent_dim)
        for layer in self.encoders:
            zq, attn_w = layer(zq)

        # x = self.recon(zq)
        # recon_kps, recon_bbox = x[:, :, :-2], x[:, :, -2:]
        recon_kps = self.recon_kps(zq[:, :13, :])
        recon_bbox = self.recon_bbox(zq[:, 13:, :])

        return recon_kps, recon_bbox
