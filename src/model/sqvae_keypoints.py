from types import SimpleNamespace
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from lightning.pytorch import LightningModule
from model.module.quantizer import GaussianVectorQuantizer
from module.keypoibnts import Decoder, Encoder


class SQVAE(LightningModule):
    def __init__(
        self,
        config: SimpleNamespace,
        annotation_path: Optional[str] = None,
    ):
        super().__init__()
        self.config = config
        self.annotation_path = annotation_path
        self.temp_init = config.temp_init
        self.temp_decay = config.temp_decay
        self.temp_min = config.temp_min
        self.latent_ndim = config.latent_ndim
        self.n_clusters = config.n_clusters

        self.encoder = None
        self.decoder = None
        self.quantizer = None

        self.annotation_path = annotation_path

    def configure_model(self):
        if self.encoder is not None:
            return
        self.encoder = Encoder(self.config)
        self.decoders = Decoder(self.config)
        self.quantizer = GaussianVectorQuantizer(self.config)

        if self.annotation_path is not None:
            anns = np.loadtxt(self.annotation_path, str, delimiter=" ", skiprows=1)
            self.annotations = anns

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.config.lr)
        sch = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda epoch: self.config.lr_lmd**epoch
        )
        return [opt], [sch]

    def forward(self, x_vis, x_spc, quantizer_is_train):
        # x_vis (b, seq_len, 17, 2)
        # x_spc (b, seq_len, 2, 2)
        # encoding
        ze, c_logits = self.encoder(x_vis, x_spc)
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
        recon_x_vis, recon_x_spc = self.decoder(x_vis, x_spc, zq)

        return (
            ze,
            zq,
            precision_q,
            prob,
            log_prob,
            recon_x_vis,
            recon_x_spc,
            c_prob,
        )

    def calc_temperature(self):
        return np.max(
            [
                self.temp_init * np.exp(-self.temp_decay * self.global_step),
                self.temp_min,
            ]
        )

    def mse_x(self, x, recon_x, mask=None):
        if x.ndim == 4:
            return F.mse_loss(recon_x, x, reduction="none").mean(dim=(1, 2, 3))  # (b,)
        elif x.ndim == 3:
            return F.mse_loss(recon_x, x, reduction="none").mean(dim=(1, 2))  # (b,)

    def loss_x(self, x, recon_x, mask=None):
        mses = self.mse_x(x, recon_x, mask)
        return mses.mean()

    def loss_kl_continuous(self, ze, zq, precision_q):
        return torch.sum(((ze - zq) ** 2) * precision_q, dim=(1, 2)).mean()

    def loss_kl_discrete(self, prob, log_prob):
        return torch.sum(prob * log_prob, dim=(0, 1)).mean()

    def training_step(self, batch, batch_idx):
        keys, ids, x_vis, x_spc, mask = batch
        keys = np.array(keys).T[0]
        ids = ids[0]
        x_vis = x_vis[0]
        x_spc = x_spc[0]
        # mask = mask[0]

        # update temperature of gumbel softmax
        temp_cur = self.calc_temperature()
        self.quantizer.temperature = temp_cur

        # forward
        (
            ze,
            zq,
            precision_q,
            prob,
            log_prob,
            recon_x_vis,
            recon_x_spc,
            c_prob,
        ) = self(x_vis, x_spc, True)

        # ELBO loss
        lrc_x_vis = self.loss_x(x_vis, recon_x_vis)
        lrc_x_spc = self.loss_x(x_spc, recon_x_spc)
        kl_continuous = self.loss_kl_continuous(ze, zq, precision_q)
        kl_discrete = self.loss_kl_discrete(prob, log_prob)

        # clustering loss
        psuedo_labels_prob = torch.full_like(c_prob, 1 / self.n_clusters)
        keys = ["{}_{}".format(*key.split("_")[0::2]) for key in keys]
        for i, key in enumerate(keys):
            if key in self.annotations.T[0]:
                label = self.annotations.T[1][key == self.annotations.T[0]]
                psuedo_labels_prob[i] = F.one_hot(
                    torch.tensor(int(label)), self.n_clusters
                ).to(self.device, torch.float32)

        mask_supervised = np.isin(keys, self.annotations.T[0]).ravel()
        mask_supervised = torch.tensor(mask_supervised).to(self.device)
        lc = F.cross_entropy(c_prob, psuedo_labels_prob, reduction="none")
        lc_psuedo = (lc * ~mask_supervised).mean()
        lc = (lc * mask_supervised).mean()

        loss_total = (
            (lrc_x_vis + lrc_x_spc) * self.config.lmd_lrc
            + kl_continuous * self.config.lmd_klc
            + kl_discrete * self.config.lmd_kld
            + (lc * 10.0 + lc_psuedo * 0.01) * self.config.lmd_c
        )

        loss_dict = dict(
            x_vis=lrc_x_vis.item(),
            x_spc=lrc_x_spc.item(),
            kl_discrete=kl_discrete.item(),
            kl_continuous=kl_continuous.item(),
            log_param_q=self.quantizer.log_param_q.item(),
            log_param_q_cls=self.quantizer.log_param_q_cls.item(),
            c=lc.item(),
            c_psuedo=lc_psuedo.item(),
            total=loss_total.item(),
        )

        self.log_dict(loss_dict, prog_bar=True, logger=True)

        return loss_total

    def predict_step(self, batch):
        keys, ids, x_vis, x_spc, mask = batch
        x_vis = x_vis.to(next(self.parameters()).device)
        x_spc = x_spc.to(next(self.parameters()).device)
        # mask = mask.to(next(self.parameters()).device)
        if x_vis.ndim == 5:
            ids = ids[0]
            x_vis = x_vis[0]
            x_spc = x_spc[0]
            # mask = mask[0]

        # forward
        (
            ze,
            zq,
            precision_q,
            prob,
            log_prob,
            recon_x_vis,
            recon_x_spc,
            c_prob,
        ) = self(x_vis, x_spc, False)

        mse_x_vis = self.mse_x(x_vis, recon_x_vis)
        mse_x_spc = self.mse_x(x_spc, recon_x_spc)

        results = []
        for i in range(len(keys)):
            data = {
                "key": keys[i],
                "id": ids[i].cpu().numpy().item(),
                "x_vis": x_vis[i].cpu().numpy(),
                "recon_x_vis": recon_x_vis[i].cpu().numpy(),
                "mse_x_vis": mse_x_vis.item(),
                "x_spc": x_spc[i].cpu().numpy(),
                "recon_x_spc": recon_x_spc[i].cpu().numpy(),
                "mse_x_spc": mse_x_spc.item(),
                "ze": ze[i].cpu().numpy(),
                "zq": zq[i].cpu().numpy(),
                "book_prob": prob[i].cpu().numpy(),
                "book_idx": prob[i].cpu().numpy().argmax(axis=1),
                "label_prob": c_prob[i].cpu().numpy(),
                "label": c_prob[i].cpu().numpy().argmax(),
            }
            results.append(data)

        return results
