from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def sample_gumbel(shape, eps=1e-10):
    U = torch.rand(shape, device="cuda")
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    g = sample_gumbel(logits.size())
    y = logits + g
    return F.softmax(y / temperature, dim=-1)


def calc_distance(z_continuous, codebook, dim_dict):
    z_continuous_flat = z_continuous.view(-1, dim_dict)
    distances = (
        torch.sum(z_continuous_flat**2, dim=1, keepdim=True)
        + torch.sum(codebook**2, dim=1)
        - 2 * torch.matmul(z_continuous_flat, codebook.t())
    )

    return distances


class GaussianVectorQuantizer(nn.Module):
    def __init__(self, config: SimpleNamespace):
        super().__init__()
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
        log_param_q_cls = np.log(config.param_q_init)
        self.log_param_q_cls = nn.Parameter(
            torch.tensor(log_param_q_cls, dtype=torch.float32)
        )

    def forward(self, ze, c_logits, is_train):
        # ze (b, n_pts, latent_ndim)
        b, n_pts, latent_ndim = ze.size()

        param_q = 1 + self.log_param_q.exp()
        precision_q = 0.5 / torch.clamp(param_q, min=1e-10)

        if is_train:
            param_q_cls = 1 + self.log_param_q_cls.exp()
            precision_q_cls = 0.5 / torch.clamp(param_q_cls, min=1e-10)

            # cumpute clustering probs
            c_probs = gumbel_softmax_sample(c_logits * precision_q_cls, self.temperature)

            # create latent tensors
            zq = torch.zeros_like(ze)
            logits = torch.zeros((n_pts, self.book_size)).to(ze.device)

            # compute logits and zq from all books
            books = torch.cat(list(self.books.parameters()), dim=0)
            books = books.view(-1, self.book_size, latent_ndim)
            for j, book in enumerate(books):
                logitj = -self.calc_distance(ze, book) * precision_q
                logits = logits + logitj * c_probs[j]
                encoding = gumbel_softmax_sample(logitj, self.temperature)
                zq = zq + torch.matmul(encoding, book) * c_probs[j]
            # mean_prob = torch.mean(prob.detach(), dim=0)
        else:
            logits = torch.empty((0, n_pts, self.book_size)).to(ze.device)
            books = torch.empty((0, self.book_size, latent_ndim)).to(ze.device)
            for i, idx in enumerate(c_logits.argmax(dim=-1)):
                book = self.books[idx]
                books = torch.cat(
                    [books, book.view(1, self.book_size, latent_ndim)], dim=0
                )

                logit = -self.calc_distance(ze[i], book) * precision_q
                logits = torch.cat(
                    [logits, logit.view(1, n_pts, self.book_size)], dim=0
                )

            indices = torch.argmax(logits, dim=2).unsqueeze(2)
            encodings = torch.zeros(
                indices.shape[0],
                indices.shape[1],
                self.book_size,
                device=indices.device,
            )
            encodings.scatter_(2, indices, 1)
            zq = torch.matmul(encodings, books)
            # mean_prob = torch.mean(encodings, dim=0)
        # zq (b, npts, latent_ndim)

        prob = torch.softmax(logits, dim=-1)
        log_prob = torch.log_softmax(logits, dim=-1)

        logits = logits.view(b, n_pts, self.book_size)
        prob = prob.view(b, n_pts, self.book_size)
        log_prob = log_prob.view(b, n_pts, self.book_size)

        return zq, precision_q, prob, log_prob
