import torch
import torch.nn as nn
import torch.nn.functional as F

from .nn.feedforward import MLP


class MaskedConv2d(nn.Conv2d):
    def __init__(self, mask_type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert mask_type in ["A", "B"]
        self.register_buffer("mask", self.weight.data.clone())
        self.mask.fill_(1)
        h = self.weight.size()[2]
        w = self.weight.size()[3]
        if mask_type == "A":
            self.mask[:, :, h // 2, w // 2 :] = 0
            self.mask[:, :, h // 2 + 1 :] = 0
        else:
            self.mask[:, :, h // 2, w // 2 + 1 :] = 0
            self.mask[:, :, h // 2 + 1 :] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)


class PixelCNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.book_size = config.book_size
        self.latent_size = config.latent_size
        nch = config.latent_dim

        self.in_emb = nn.Embedding(config.book_size, config.latent_dim)
        self.layers = nn.ModuleList()
        conv_block = nn.Sequential(
            MaskedConv2d(
                mask_type="A",
                in_channels=nch,
                out_channels=nch,
                kernel_size=7,
                stride=1,
                padding=3,
                bias=False,
            ),
            nn.BatchNorm2d(nch),
            nn.ReLU(inplace=True),
        )
        self.layers.append(conv_block)
        for _ in range(config.n_cnnblocks_pixelcnn):
            conv_block = nn.Sequential(
                MaskedConv2d(
                    mask_type="B",
                    in_channels=nch,
                    out_channels=nch,
                    kernel_size=7,
                    stride=1,
                    padding=3,
                    bias=False,
                ),
                nn.BatchNorm2d(nch),
                nn.ReLU(inplace=True),
            )
            self.layers.append(conv_block)

        self.emb_c = MLP(config.n_clusters, nch)
        self.layers_c = nn.ModuleList()
        for _ in range(config.n_cnnblocks_pixelcnn + 1):
            self.layers_c.append(nn.Conv1d(nch, nch, 1, bias=False))

        self.out_conv = nn.Conv2d(
            in_channels=nch, out_channels=config.book_size, kernel_size=1
        )

    def forward(self, z_indices, c_probs):
        z_indices = self.in_emb(z_indices)
        z_indices = z_indices.permute(0, 3, 1, 2)

        c_probs = self.emb_c(c_probs).unsqueeze(-1)

        z_indices = z_indices + c_probs.unsqueeze(-1)

        for layer, layer_c in zip(self.layers, self.layers_c):
            c_probs_tmp = layer_c(c_probs)
            z_indices = layer(z_indices) + c_probs_tmp.unsqueeze(-1)
        logits = self.out_conv(z_indices)
        return logits  # (b, book_size, h, w)

    def sample_prior(self, c):
        prior_indices = torch.randint(
            0, self.book_size, (c.size(0), self.latent_size[0], self.latent_size[1])
        ).to(c.device, torch.long)
        prior_indices[:, 1:, 1:] = 0

        for i in range(self.latent_size[0]):
            for j in range(self.latent_size[1]):
                logits = self(prior_indices, c)
                probs = F.softmax(logits[:, :, i, j], dim=1)
                prior_indices[:, i, j] = torch.multinomial(probs, 1, replacement=True)[
                    :, 0
                ].long()
        return prior_indices
