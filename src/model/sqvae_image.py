from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch import LightningModule

from src.model.module.mnist import ClassificationHead, Decoder, Encoder
from src.model.module.quantizer import GaussianVectorQuantizer, gumbel_softmax_sample


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class CSQVAE(LightningModule):
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
        size = np.sqrt(config.latent_npts).astype(int)
        self.latent_size = (size, size)

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
        sch = torch.optim.lr_scheduler.ExponentialLR(opt, self.config.lr_lmd)
        return [opt], [sch]

    def forward(self, x, is_train):
        ze = self.encoder(x)
        b, ndim, h, w = ze.size()
        ze = ze.permute(0, 2, 3, 1).contiguous()
        ze = ze.view(b, -1, ndim)

        c_logits = self.cls_head(ze)
        if is_train:
            c_probs = gumbel_softmax_sample(c_logits, self.cls_head.temperature)
        else:
            c_probs = F.softmax(c_logits, dim=-1)

        zq, precision_q, logits = self.quantizer(ze, is_train)

        ze = ze.view(b, h, w, ndim)
        ze = ze.permute(0, 3, 1, 2)
        zq = zq.view(b, h, w, ndim)
        zq = zq.permute(0, 3, 1, 2)

        recon_x = self.decoder(zq)

        if is_train:
            zq_sampled, logits_sampled, mu_sampled = self.quantizer.sample_from_c(
                c_probs, is_train
            )
            # zq_sampled = zq_sampled.view(b, h, w, ndim)
            # zq_sampled = zq_sampled.permute(0, 3, 1, 2)
        else:
            # zq_sampled = None
            logits_sampled = None
            # mu_sampled = None

        return (
            recon_x,
            ze,
            zq,
            precision_q,
            logits,
            logits_sampled,
            c_logits,
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

    def loss_kl_discrete(self, logits, logits_sampled):
        p = logits_sampled.softmax(dim=-1)
        p_log = logits_sampled.log_softmax(dim=-1)
        q_log = logits.log_softmax(dim=-1)
        kl = torch.sum(p * (p_log - q_log), dim=(1, 2)).mean()
        return kl

    def loss_c_elbo(self, c_logits):
        prob = F.softmax(c_logits, dim=-1)
        log_prob = F.log_softmax(c_logits, dim=-1)
        lc_elbo = torch.sum(prob * log_prob, dim=1).mean()
        return lc_elbo

    def loss_c_real(self, c_logits, labels):
        c_prob = F.softmax(c_logits, dim=-1)
        if torch.all(labels == -1):  # all samples are unlabeled
            lc_real = torch.Tensor([0.0]).to(self.device)
        else:
            mask_supervised = labels != -1
            lc_real = F.cross_entropy(
                c_prob[mask_supervised], labels[mask_supervised], reduction="sum"
            )
            lc_real = lc_real / c_prob[mask_supervised].size(0)

        return lc_real

    def training_step(self, batch, batch_idx):
        x, labels = batch

        # update temperature of gumbel softmax
        temp_cur = self.calc_temperature(
            self.temp_init_cls, self.temp_decay_cls, self.temp_min_cls
        )
        self.cls_head.temperature = temp_cur
        temp_cur = self.calc_temperature(self.temp_init, self.temp_decay, self.temp_min)
        self.quantizer.temperature = temp_cur

        # forward
        (
            recon_x,
            ze,
            zq,
            precision_q,
            logits,
            logits_sampled,
            c_logits,
        ) = self(x, True)

        # ELBO loss
        lrc_x = self.loss_x(x, recon_x)
        kl_continuous = self.loss_kl_continuous(ze, zq, precision_q)
        kl_discrete = self.loss_kl_discrete(logits, logits_sampled)

        # clustering loss
        lc_elbo = self.loss_c_elbo(c_logits)
        lc_real = self.loss_c_real(c_logits, labels)

        loss_total = (
            lrc_x * self.config.lmd_lrc
            + kl_continuous * self.config.lmd_klc
            + kl_discrete * self.config.lmd_kld
            + lc_elbo * self.config.lmd_c_elbo
            + lc_real * self.config.lmd_c_real
        )

        loss_dict = dict(
            x=lrc_x.item(),
            kl_discrete=kl_discrete.item(),
            kl_continuous=kl_continuous.item(),
            log_param_q=self.quantizer.log_param_q.item(),
            log_param_q_cls=self.cls_head.log_param_q_cls.item(),
            c_elbo=lc_elbo.item(),
            c_real=lc_real.item(),
            total=loss_total.item(),
        )
        self.log_dict(loss_dict, prog_bar=True, logger=True)

        return loss_total

    def predict_step(self, batch):
        x, labels = batch

        (
            recon_x,
            ze,
            zq,
            precision_q,
            logits,
            logits_sampled,
            c_logits,
        ) = self(x, False)
        x = x.permute(0, 2, 3, 1)
        recon_x = recon_x.permute(0, 2, 3, 1)
        ze = ze.permute(0, 2, 3, 1)
        zq = zq.permute(0, 2, 3, 1)
        prob = logits.softmax(dim=-1)
        c_prob = c_logits.softmax(dim=-1)

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

    def sample(self, c_probs: torch.Tensor):
        c_probs = c_probs.to(self.device)
        b = c_probs.size(0)
        # sample zq from book
        zq, logits, mu = self.quantizer.sample_from_c(c_probs, False)

        # generate samples
        h, w = self.latent_size
        zq = zq.view(b, h, w, self.latent_ndim)
        generated_x = self.decoder(zq.permute(0, 3, 1, 2))
        generated_x = generated_x.permute(0, 2, 3, 1)

        results = []
        for i in range(b):
            data = {
                "gen_x": generated_x[i].detach().cpu().numpy(),
                "zq": zq[i].detach().cpu().numpy(),
                "gt": c_probs[i].argmax(dim=-1).cpu(),
            }
            results.append(data)

        return results
