from types import SimpleNamespace

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


class GaussianVectorQuantizer(nn.Module):
    def __init__(self, config: SimpleNamespace):
        super().__init__()
        self.n_clusters = config.n_clusters
        self.book_size = config.book_size
        self.dim = config.latent_dim
        size = config.latent_size
        self.npts = size[0] * size[1]

        self.book = nn.Parameter(torch.randn(self.book_size, config.latent_dim))

    def calc_distance(self, z, precision_q):
        distances = -(
            torch.sum(z**2, dim=-1, keepdim=True)
            + torch.sum(self.book**2, dim=-1)
            - 2 * torch.matmul(z, self.book.t())
        )

        return distances * precision_q

    def forward(self, z, c_probs, mu, log_param_q, temperature, is_train):
        b = z.size(0)

        # if c_probs is not None and is_train:
        #     mu = torch.cat([m.unsqueeze(0) for m in self.mu], dim=0)
        #     mu = mu.unsqueeze(0)
        #     mu = torch.sum(
        #         mu * c_probs.view(b, self.n_clusters, 1, 1), dim=1
        #     )  # (b, npts, ndim)
        # elif c_probs is not None and not is_train:  # pred
        if c_probs is not None:
            mu_sampled = torch.cat(
                [mu[c].unsqueeze(0) for c in c_probs.argmax(dim=-1)], dim=0
            )
            z = z + mu_sampled
        else:  # pre-train sqvae and diffusion
            mu_sampled = None

        param_q = log_param_q.exp()
        precision_q = 0.5 / torch.clamp(param_q, min=1e-10)
        logits = self.calc_distance(z.view(-1, self.dim), precision_q)

        if is_train:
            encodings = gumbel_softmax_sample(logits, temperature)
            zq = torch.mm(encodings, self.book)
        else:
            indices = torch.argmax(logits, dim=-1).unsqueeze(1)
            encodings = torch.zeros(b * self.npts, self.book_size).to(z.device)
            encodings.scatter_(1, indices, 1)
            zq = torch.mm(encodings, self.book)

        logits = logits.view(b, -1, self.book_size)
        zq = zq.view(b, -1, self.dim)

        return zq, precision_q, logits, mu_sampled

    def z_to_zq(self, z, log_param_q):
        b = z.size(0)

        param_q = log_param_q.exp()
        precision_q = 0.5 / torch.clamp(param_q, min=1e-10)
        logits = self.calc_distance(z.view(-1, self.dim), precision_q)

        indices = torch.argmax(logits, dim=-1).unsqueeze(1)
        encodings = torch.zeros(b * self.npts, self.book_size).to(z.device)
        encodings.scatter_(1, indices, 1)
        zq = torch.mm(encodings, self.book)

        zq = zq.view(b, -1, self.dim)

        return zq
