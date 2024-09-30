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


def calc_distance(ze, book, nch):
    ze = ze.view(-1, nch)
    distances = (
        torch.sum(ze**2, dim=1, keepdim=True)
        + torch.sum(book**2, dim=1)
        - 2 * torch.matmul(ze, book.t())
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
        log_param_q_cls = np.log(config.param_q_cls_init)
        self.log_param_q_cls = nn.Parameter(
            torch.tensor(log_param_q_cls, dtype=torch.float32)
        )

    def forward(self, ze, c_logits, is_train):
        ndim = ze.ndim
        if ndim == 3:
            b, npts, nch = ze.size()
        elif ndim == 4:
            # image tensor
            b, nch, h, w = ze.size()
            npts = h * w
            ze = ze.permute(0, 2, 3, 1).contiguous()
        else:
            raise ValueError

        param_q = 1 + self.log_param_q.exp()
        precision_q = 0.5 / torch.clamp(param_q, min=1e-10)

        if is_train:
            param_q_cls = 1 + self.log_param_q_cls.exp()
            precision_q_cls = 0.5 / torch.clamp(param_q_cls, min=1e-10)

            # cumpute clustering probs
            c_probs = gumbel_softmax_sample(
                c_logits * precision_q_cls, self.temperature
            )

            # create latent tensors
            zq = torch.zeros_like(ze)
            logits = torch.zeros((b, npts, self.book_size)).to(ze.device)

            # compute logits and zq from all books
            books = torch.cat(list(self.books.parameters()), dim=0)
            books = books.view(-1, self.book_size, nch)
            for i, book in enumerate(books):
                c_probs_i = c_probs[:, i].view(b, 1)
                logits_i = -calc_distance(ze, book, nch) * precision_q
                logits = logits + (logits_i.view(b, -1) * c_probs_i).view(
                    b, npts, self.book_size
                )

                encoding = gumbel_softmax_sample(logits_i, self.temperature)
                zqi = torch.matmul(encoding, book).view(b, -1) * c_probs_i
                zqi = zqi.view(b, h, w, nch)
                zq = zq + zqi
            # mean_prob = torch.mean(prob.detach(), dim=0)
        else:
            logits = torch.empty((0, npts, self.book_size)).to(ze.device)
            books = torch.empty((0, self.book_size, nch)).to(ze.device)
            for i, c in enumerate(c_logits.argmax(dim=-1)):
                book = self.books[c]
                books = torch.cat([books, book.view(1, self.book_size, nch)], dim=0)

                logit = -calc_distance(ze[i], book, nch) * precision_q
                logits = torch.cat([logits, logit.view(1, npts, self.book_size)], dim=0)

            indices = torch.argmax(logits, dim=2).unsqueeze(2)
            encodings = torch.zeros(b, npts, self.book_size).to(ze.device)
            encodings.scatter_(2, indices, 1)
            zq = torch.matmul(encodings, books)
            # mean_prob = torch.mean(encodings, dim=0)

        if ndim == 4:
            # image tensor
            zq = zq.view(b, h, w, nch)
            zq = zq.permute(0, 3, 1, 2)

        prob = torch.softmax(logits, dim=-1)
        log_prob = torch.log_softmax(logits, dim=-1)

        return zq, precision_q, prob, log_prob
