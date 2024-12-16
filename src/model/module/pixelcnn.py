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
        nch = config.latent_ndim

        self.layers = nn.ModuleList()

        conv_block = nn.Sequential(
            MaskedConv2d(
                mask_type="A",
                in_channels=1,
                out_channels=nch,
                kernel_size=5,
                stride=1,
                padding=2,
                bias=False,
            ),
            nn.BatchNorm2d(nch),
            nn.ReLU(inplace=True),
        )
        self.in_conv = conv_block
        for _ in range(config.n_resblocks_pixelcnn):
            conv_block = nn.Sequential(
                MaskedConv2d(
                    mask_type="B",
                    in_channels=nch,
                    out_channels=nch,
                    kernel_size=5,
                    stride=1,
                    padding=2,
                    bias=False,
                ),
                nn.BatchNorm2d(nch),
                nn.ReLU(inplace=True),
            )
            self.layers.append(conv_block)

        self.mlp_c = MLP(config.n_clusters, nch)

        self.out_conv = nn.Conv2d(
            in_channels=nch, out_channels=config.book_size, kernel_size=1
        )

    def forward(self, z, c):
        c = self.mlp_c(c)
        z = self.in_conv(z)
        z = z + c.view(c.size(0), c.size(1), 1, 1)
        for layer in self.layers:
            z = layer(z)
        logits = self.out_conv(z)
        return logits

    def sample_prior_indices(self, c):
        prior = torch.Tensor(c.size(0), 1, self.latent_size[0], self.latent_size[1]).to(
            c.device
        )
        prior.fill_(0)
        for i in range(self.latent_size[0]):
            for j in range(self.latent_size[1]):
                logits = self(prior, c)
                probs = F.softmax(logits[:, :, i, j], dim=1)
                prior[:, :, i, j] = torch.multinomial(probs, 1).float() / self.book_size

        return (prior * self.book_size).long()
