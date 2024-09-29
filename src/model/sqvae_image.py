from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from lightning.pytorch import LightningModule

from model.module.mnist import Decoder, Encoder
from model.module.quantizer import GaussianVectorQuantizer


class SQVAE(LightningModule):
    def __init__(self, config: SimpleNamespace):
        super().__init__()
        self.config = config
        self.temp_init = config.temp_init
        self.temp_decay = config.temp_decay
        self.temp_min = config.temp_min
        self.latent_ndim = config.latent_ndim
        self.n_clusters = config.n_clusters

        self.encoder = None
        self.decoder = None
        self.quantizer = None

    def configure_model(self):
        if self.encoder is not None:
            return
        self.encoder = Encoder(self.config)
        self.decoders = Decoder(self.config)
        self.quantizer = GaussianVectorQuantizer(self.config)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.config.lr)
        sch = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda epoch: self.config.lr_lmd**epoch
        )
        return [opt], [sch]

    def forward(self, x, quantizer_is_train):
        # x (b, nch, w, h)
        # encoding
        ze, c_logits = self.encoder(x)
        c_prob = F.softmax(c_logits, dim=-1)
        # ze (b, npts, latent_ndim)
        # c_prob (b, n_clusters)

        # quantization
        zq, precision_q, prob, log_prob = self.quantizer(
            ze, c_logits, quantizer_is_train
        )
        # zq (b, npts, latent_ndim)
        # prob (b, npts, book_size)

        # reconstruction
        recon_x = self.decoder(zq)

        return (
            recon_x,
            c_prob,
            ze,
            zq,
            precision_q,
            prob,
            log_prob,
        )

    def calc_temperature(self):
        return np.max(
            [
                self.temp_init * np.exp(-self.temp_decay * self.global_step),
                self.temp_min,
            ]
        )

    def loss_x(self, x, recon_x):
        return F.mse_loss(recon_x, x)

    def loss_kl_continuous(self, ze, zq, precision_q):
        return torch.sum(((ze - zq) ** 2) * precision_q, dim=(1, 2)).mean()

    def loss_kl_discrete(self, prob, log_prob):
        return torch.sum(prob * log_prob, dim=(0, 1)).mean()

    def training_step(self, batch, batch_idx):
        x, labels = batch

        # update temperature of gumbel softmax
        temp_cur = self.calc_temperature()
        self.quantizer.temperature = temp_cur

        # forward
        (
            recon_x,
            c_prob,
            ze,
            zq,
            precision_q,
            prob,
            log_prob,
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
            log_param_q_cls=self.quantizer.log_param_q_cls.item(),
        )

        # clustering loss
        c_prior = torch.full_like(c_prob, 1 / self.n_clusters)
        c_prob = torch.clamp(c_prob, min=1e-10)
        lc_psuedo = (c_prior * (c_prior.log() - c_prob.log())).mean()
        loss_dict["c_psuedo"] = lc_psuedo.item()

        if torch.all(torch.isnan(labels)):
            lc_real = 0.0
            loss_dict["c_real"] = 0.0
        else:
            mask_supervised = ~torch.isnan(labels)
            if torch.any(mask_supervised):
                lc_real = F.cross_entropy(
                    c_prob[mask_supervised], labels[mask_supervised], reduction="sum"
                )
                lc_real = lc_real / labels.size(0)
                loss_dict["c_real"] = lc_real.item()
            else:
                lc_real = 0.0
                loss_dict["c_real"] = 0.0

        loss_total = (
            lrc_x * self.config.lmd_lrc
            + kl_continuous * self.config.lmd_klc
            + kl_discrete * self.config.lmd_kld
            + lc_real * self.config.lmd_c_real
            + lc_psuedo * self.config.lmd_c_psuedo
        )
        loss_dict["total"] = loss_total.item()

        self.log_dict(loss_dict, prog_bar=True, logger=True)

        return loss_total

    def predict_step(self, batch):
        x, labels = batch

        (
            recon_x,
            c_prob,
            ze,
            zq,
            precision_q,
            prob,
            log_prob,
        ) = self(x, False)

        results = []
        for i in range(len(x)):
            data = {
                "x": x[i].cpu().numpy(),
                "recon_x": recon_x[i].cpu().numpy(),
                "ze": ze[i].cpu().numpy(),
                "zq": zq[i].cpu().numpy(),
                "book_prob": prob[i].cpu().numpy(),
                "book_idx": prob[i].cpu().numpy().argmax(axis=1),
                "label_prob": c_prob[i].cpu().numpy(),
                "label": c_prob[i].cpu().numpy().argmax(),
                "gt": labels[i].cpu().numpy().item(),
            }
            results.append(data)

        return results
