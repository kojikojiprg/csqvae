from types import SimpleNamespace
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch import LightningModule

from src.model.module.mnist import ClassificationHead, Decoder, Encoder
from src.model.module.quantizer import GaussianVectorQuantizer


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class SQVAE(LightningModule):
    def __init__(self, config: SimpleNamespace):
        super().__init__()
        self.config = config
        self.n_clusters = config.n_clusters
        self.temp_init_cls = config.temp_init_cls
        self.temp_decay_cls = config.temp_decay_cls
        self.temp_min_cls = config.temp_min_cls

        self.temp_init = config.temp_init
        self.temp_decay = config.temp_decay
        self.temp_min = config.temp_min

        self.latent_ndim = config.latent_ndim

        self.encoder = None
        self.decoder = None
        self.quantizer = None
        self.cls_head = None
        self.dim_x = 28 * 28
        self.flg_arelbo = True

    def configure_model(self):
        if self.encoder is not None:
            return
        self.encoder = Encoder(self.config)
        self.decoder = Decoder(self.config)
        self.quantizer = GaussianVectorQuantizer(self.config)
        self.cls_head = ClassificationHead(self.config)
        if not self.flg_arelbo:
            self.logvar_x = nn.Parameter(torch.tensor(np.log(0.1)))
        self.apply(weights_init)

    def configure_optimizers(self):
        opt = torch.optim.RAdam(self.parameters(), lr=self.config.lr)
        return opt
        # sch = torch.optim.lr_scheduler.ExponentialLR(opt, self.config.lr_lmd)
        # return [opt], [sch]

    def forward(self, x, is_train):
        ze = self.encoder(x)

        c_logits = self.cls_head(ze)
        c_prob = F.softmax(c_logits, dim=-1)
        c_log_prob = F.log_softmax(c_logits, dim=-1)

        zq, precision_q, prob, log_prob = self.quantizer(ze, c_prob, is_train)

        recon_x = self.decoder(zq)

        return (
            recon_x,
            ze,
            zq,
            precision_q,
            prob,
            log_prob,
            c_prob,
            c_log_prob,
        )

    def calc_temperature(self, temp_init, temp_decay, temp_min):
        return np.max(
            [
                temp_init * np.exp(-temp_decay * self.global_step),
                temp_min,
            ]
        )

    def loss_x(self, x, recon_x):
        mse = F.mse_loss(recon_x, x, reduction="sum") / x.size(0)
        if self.flg_arelbo:
            # "Preventing Posterior Collapse Induced by Oversmoothing in Gaussian VAE"
            # https://arxiv.org/abs/2102.08663
            loss_x = self.dim_x * torch.log(mse) / 2
        else:
            loss_x = mse / (2 * self.logvar_x.exp()) + self.dim_x * self.logvar_x / 2

        return loss_x

    def loss_kl_continuous(self, ze, zq, precision_q):
        return torch.sum(((ze - zq) ** 2) * precision_q, dim=(1, 2)).mean()

    def loss_kl_discrete(self, prob, log_prob):
        return torch.sum(prob * log_prob, dim=(0, 1)).mean()

    def training_step(self, batch, batch_idx):
        x, labels = batch

        # update temperature of gumbel softmax
        temp_cur = self.calc_temperature(self.temp_init_cls, self.temp_decay_cls, self.temp_min_cls)
        self.cls_head.temperature = temp_cur
        temp_cur = self.calc_temperature(self.temp_init, self.temp_decay, self.temp_min)
        self.quantizer.temperature = temp_cur

        # forward
        (
            recon_x,
            ze,
            zq,
            precision_q,
            prob,
            log_prob,
            c_prob,
            c_log_prob,
        ) = self(x, True)

        # ELBO loss
        lrc_x = self.loss_x(x, recon_x)
        kl_continuous = self.loss_kl_continuous(ze, zq, precision_q)
        kl_discrete = self.loss_kl_discrete(prob, log_prob)
        loss_dict = dict(
            x=lrc_x.item(),
            kl_discrete=kl_discrete.item(),
            kl_continuous=kl_continuous.item(),
            log_param_q=self.quantizer.log_param_q.item(),
            log_param_q_cls=self.cls_head.log_param_q_cls.item(),
        )

        # clustering loss
        c_prior = torch.full_like(c_prob, 1 / self.n_clusters)
        c_prob = torch.clamp(c_prob, min=1e-10)
        lc_prior = F.kl_div(c_prob.log(), c_prior, reduction="batchmean")
        loss_dict["c_elbo"] = lc_prior.item()

        if torch.all(labels == -1):  # all samples are unlabeled
            lc_real = 0.0
            loss_dict["c_true"] = 0.0
        else:
            mask_supervised = labels != -1
            lc_real = F.cross_entropy(c_prob[mask_supervised], labels[mask_supervised])
            loss_dict["c_true"] = lc_real.item()

        loss_total = (
            lrc_x * self.config.lmd_lrc
            + kl_continuous * self.config.lmd_klc
            + kl_discrete * self.config.lmd_kld
            + lc_real * self.config.lmd_c_real
            + lc_prior * self.config.lmd_c_prior
        )
        loss_dict["loss"] = loss_total.item()

        self.log_dict(loss_dict, prog_bar=True, logger=True)

        return loss_total

    def predict_step(self, batch):
        x, labels = batch

        (
            recon_x,
            ze,
            zq,
            precision_q,
            prob,
            log_prob,
            c_prob,
            c_log_prob,
        ) = self(x, False)
        x = x.permute(0, 2, 3, 1)
        recon_x = recon_x.permute(0, 2, 3, 1)
        ze = ze.permute(0, 2, 3, 1)
        zq = zq.permute(0, 2, 3, 1)

        results = []
        for i in range(len(x)):
            data = {
                "x": x[i].detach().cpu().numpy(),
                "recon_x": recon_x[i].detach().cpu().numpy(),
                "ze": ze[i].detach().cpu().numpy(),
                "zq": zq[i].detach().cpu().numpy(),
                "book_prob": prob[i].detach().cpu().numpy(),
                "book_idx": prob[i].detach().cpu().numpy().argmax(axis=1),
                "label_prob": c_prob[i].detach().cpu().numpy(),
                "label": c_prob[i].detach().cpu().numpy().argmax(),
                "label_gt": labels[i].detach().cpu().numpy().item(),
            }
            results.append(data)

        return results

    def sample(self, c: int, nsamples: int, size: Tuple[int, int], seed: int = 42):
        torch.random.manual_seed(seed)

        if len(size) == 1 or isinstance(size, int):
            npts = size
        elif len(size) == 2:
            npts = size[0] * size[1]

        # sample zq from book
        book = self.quantizer.books[c]
        indices = torch.randint(0, len(book), (nsamples * npts,)).to(self.device)
        indices = indices.view(nsamples, npts, 1)
        encodings = torch.zeros(nsamples, npts, len(book)).to(self.device)
        encodings.scatter_(2, indices, 1)
        zq = torch.matmul(encodings, book.view(1, len(book), -1))
        zq = zq.view(nsamples, size[0], size[1], -1)

        # generate samples
        generated_x = self.decoder(zq.permute(0, 3, 1, 2))
        generated_x = generated_x.permute(0, 2, 3, 1)

        results = []
        for i in range(nsamples):
            data = {
                "gen_x": generated_x[i].detach().cpu().numpy(),
                "zq": zq[i].detach().cpu().numpy(),
                "book_idx": indices[i].detach().cpu().numpy(),
                "gt": c,
            }
            results.append(data)

        return results
