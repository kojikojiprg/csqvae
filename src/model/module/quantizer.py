from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def sample_gumbel(logits, eps=1e-10):
    U = torch.rand_like(logits)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    g = sample_gumbel(logits)
    y = logits + g
    return F.softmax(y / temperature, dim=-1)


def calc_distance(ze, book):
    distances = (
        torch.sum(ze**2, dim=-1, keepdim=True)
        + torch.sum(book**2, dim=-1)
        - 2 * torch.matmul(ze, book.t())
    )

    return distances


class GaussianVectorQuantizer(nn.Module):
    def __init__(self, config: SimpleNamespace):
        super().__init__()
        self.n_clusters = config.n_clusters
        self.book_size = config.book_size
        self.ndim = config.latent_ndim
        self.npts = config.latent_npts

        self.book = nn.Parameter(torch.randn(self.book_size, config.latent_ndim))

        # self.weights = nn.ParameterList(
        #     [
        #         nn.Parameter(torch.randn(config.latent_npts, self.book_size))
        #         for _ in range(config.n_clusters)
        #     ]
        # )

        mu = [torch.randn(1, config.latent_ndim) for _ in range(config.n_clusters)]
        self.mu = nn.ParameterList(
            [
                nn.Parameter(
                    torch.randn(config.latent_npts, config.latent_ndim) + mu[i]
                )
                for i in range(config.n_clusters)
            ]
        )

        self.temperature = None
        log_param_q = np.log(config.param_q_init)
        self.log_param_q = nn.Parameter(torch.tensor(log_param_q, dtype=torch.float32))

    def forward(self, ze, is_train):
        b = ze.size(0)

        param_q = self.log_param_q.exp()
        precision_q = 0.5 / torch.clamp(param_q, min=1e-10)
        logits = -calc_distance(ze.view(-1, self.ndim), self.book) * precision_q

        if is_train:
            encodings = gumbel_softmax_sample(logits, self.temperature)
            zq = torch.mm(encodings, self.book)
        else:
            indices = torch.argmax(logits, dim=-1).unsqueeze(1)
            encodings = torch.zeros(b * self.npts, self.book_size).to(ze.device)
            encodings.scatter_(1, indices, 1)
            zq = torch.mm(encodings, self.book)

        logits = logits.view(b, -1, self.book_size)
        zq = zq.view(b, -1, self.ndim)

        return zq, precision_q, logits

    def sample_from_c(self, c_probs, is_train):
        b = c_probs.size(0)

        if is_train:
            mu = torch.cat([m.unsqueeze(0) for m in self.mu], dim=0)
            mu = mu.unsqueeze(0)
            mu = torch.sum(
                mu * c_probs.view(b, self.n_clusters, 1, 1), dim=1
            )  # (b, npts, ndim)
        else:
            mu = torch.cat(
                [self.mu[c].unsqueeze(0) for c in c_probs.argmax(dim=-1)], dim=0
            )

        z = torch.randn(b, self.npts, self.ndim).to(c_probs.device)
        z = z + mu

        zq, _, logits = self(z, is_train)

        return zq, logits, mu
