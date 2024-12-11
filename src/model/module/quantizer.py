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


def calc_distance(ze, books, n_clusters, book_size):
    distances = (
        torch.sum(ze**2, dim=-1, keepdim=True)
        + torch.sum(books**2, dim=-1).view(n_clusters, 1, book_size)
        - 2 * torch.matmul(ze, books.permute(0, 2, 1))
    )

    return distances


class GaussianVectorQuantizer(nn.Module):
    def __init__(self, config: SimpleNamespace):
        super().__init__()
        self.n_clusters = config.n_clusters
        self.book_size = config.book_size
        self.books = nn.ParameterList(
            [
                nn.Parameter(torch.randn(self.book_size, config.latent_ndim))
                for _ in range(config.n_clusters)
            ]
        )

        self.temperature = None
        log_param_q = np.log(config.param_q_init)
        self.log_param_q = nn.Parameter(torch.tensor(log_param_q, dtype=torch.float32))

    def forward(self, ze, c_probs, is_train):
        if ze.ndim == 3:
            b, npts, ndim = ze.size()
        elif ze.ndim == 4:
            # image tensor
            b, ndim, h, w = ze.size()
            npts = h * w
            ze = ze.permute(0, 2, 3, 1).contiguous()
        else:
            raise ValueError

        param_q = self.log_param_q.exp()
        precision_q = 0.5 / torch.clamp(param_q, min=1e-10)

        if is_train:
            books = torch.cat(
                [book.view(1, self.book_size, ndim) for book in self.books], dim=0
            )
            logits = torch.empty((0, self.n_clusters, npts, self.book_size)).to(ze.device)
            zq = torch.empty((0, npts, ndim)).to(ze.device)
            for i, c_prob in enumerate(c_probs):
                ze_tmp = ze[i].view(1, -1, ndim).repeat(self.n_clusters, 1, 1)

                logit = (
                    -calc_distance(ze_tmp, books, self.n_clusters, self.book_size)
                    * precision_q
                )

                logits = torch.cat([logits, logit.unsqueeze(0)], dim=0)
                encodings = gumbel_softmax_sample(logits, self.temperature)

                zq_tmp = torch.matmul(encodings, books) * c_prob.view(self.n_clusters, 1, 1)
                # zq_tmp (1, n_clusters, npts, ndim)
                zq_tmp = zq_tmp.sum(dim=1)
                # zq_tmp (1, npts, ndim)
            zq = torch.cat([zq, zq_tmp], dim=0)
        else:
            books = torch.empty((0, self.book_size, ndim)).to(ze.device)
            logits = torch.empty((0, npts, self.book_size)).to(ze.device)
            for i, c in enumerate(c_probs.argmax(dim=-1)):
                book = self.books[c]
                books = torch.cat([books, book.view(1, self.book_size, ndim)], dim=0)

                logit = -calc_distance(ze[i], book, ndim) * precision_q
                logits = torch.cat([logits, logit.unsqueeze(0)], dim=0)

            indices = torch.argmax(logits, dim=2).unsqueeze(2)
            encodings = torch.zeros(b, npts, self.book_size).to(ze.device)
            encodings.scatter_(2, indices, 1)
            zq = torch.matmul(encodings, books)

        if ze.ndim == 4:
            # image tensor
            zq = zq.view(b, h, w, ndim)
            zq = zq.permute(0, 3, 1, 2)

        prob = torch.softmax(logits, dim=-1)
        log_prob = torch.log_softmax(logits, dim=-1)

        return zq, precision_q, prob, log_prob
