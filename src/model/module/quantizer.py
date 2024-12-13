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
        self.weights = nn.ParameterList(
            [
                nn.Parameter(torch.randn(config.latent_npts, self.book_size))
                for _ in range(config.n_clusters)
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

    def sample_logits_from_c(self, c_probs):
        weights = torch.cat([w.unsqueeze(0) for w in self.weights], dim=0)
        # weights (n_clusters, npts, ndim)

        b = c_probs.size(0)
        logits = weights.unsqueeze(0).repeat(b, 1, 1, 1) * c_probs.view(
            b, self.n_clusters, 1, 1
        )
        logits = logits.sum(dim=1)

        return logits

    def sample_zq_from_c(self, c_probs, add_random):
        b = c_probs.size(0)
        logits = self.sample_logits_from_c(c_probs)
        logits = logits.view(-1, self.book_size)

        # if add_random:
        #     logits = logits + torch.randn_like(logits)

        indices = torch.argmax(logits, dim=-1).unsqueeze(1)
        encodings = torch.zeros(b * self.npts, self.book_size).to(c_probs.device)
        encodings.scatter_(1, indices, 1)
        zq = torch.mm(encodings, self.book)

        if add_random:
            zq = zq + torch.randn_like(zq)

        logits = logits.view(b, -1, self.book_size)
        zq = zq.view(b, -1, self.ndim)

        return zq, logits
