from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch import LightningModule

from src.model.module.cifar10 import ClassificationHead as ClassificationHead_CIFAR10
from src.model.module.cifar10 import Decoder as Decoder_CIFAR10
from src.model.module.cifar10 import Encoder as Encoder_CIFAR10
from src.model.module.diffusion import DiffusionModel
from src.model.module.mnist import ClassificationHead as ClassificationHead_MNIST
from src.model.module.mnist import Decoder as Decoder_MNIST
from src.model.module.mnist import Encoder as Encoder_MNIST
from src.model.module.pixelcnn import PixelCNN
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
        self.dataset_name = config.name
        self.n_clusters = config.n_clusters
        self.temp_init_cls = config.temp_init_cls
        self.temp_decay_cls = config.temp_decay_cls
        self.temp_min_cls = config.temp_min_cls

        self.temp_init = config.temp_init
        self.temp_decay = config.temp_decay
        self.temp_min = config.temp_min

        self.book_size = config.book_size
        self.latent_dim = config.latent_dim
        self.latent_size = config.latent_size
        self.bs = config.batch_size

        self.encoder = None
        self.decoder = None
        self.cls_head = None
        self.quantizer = None
        self.pixelcnn = None
        self.diffusion = None
        self.x_dim = config.x_dim
        self.flg_arelbo = True

    def configure_model(self):
        if self.encoder is not None:
            return

        if self.dataset_name == "mnist":
            self.encoder = Encoder_MNIST(self.config)
            self.decoder = Decoder_MNIST(self.config)
            self.cls_head = ClassificationHead_MNIST(self.config)
        elif self.dataset_name == "cifar10":
            self.encoder = Encoder_CIFAR10(self.config)
            self.decoder = Decoder_CIFAR10(self.config)
            self.cls_head = ClassificationHead_CIFAR10(self.config)

        self.quantizer = GaussianVectorQuantizer(self.config)
        if self.config.gen_model == "pixelcnn":
            self.pixelcnn = PixelCNN(self.config)
        elif self.config.gen_model == "diffusion":
            self.diffusion = DiffusionModel(self.config)
        else:
            raise NotImplementedError

        if not self.flg_arelbo:
            self.logvar_x = nn.Parameter(torch.tensor(np.log(0.1)))
        self.apply(weights_init)

    def configure_optimizers(self):
        opt = torch.optim.RAdam(self.parameters(), lr=self.config.lr)
        sch = torch.optim.lr_scheduler.ExponentialLR(opt, self.config.lr_lmd)
        # sch = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     opt, self.config.t_max, self.config.lr_min
        # )
        return [opt], [sch]

    def forward(self, x, is_train):
        ze = self.encoder(x)

        ze = ze.permute(0, 2, 3, 1).contiguous()
        ze = ze.view(self.bs, -1, self.latent_dim)

        c_logits = self.cls_head(ze)

        zq, precision_q, logits = self.quantizer(ze, is_train)

        h, w = self.latent_size
        ze = ze.view(self.bs, h, w, self.latent_dim)
        ze = ze.permute(0, 3, 1, 2)
        zq = zq.view(self.bs, h, w, self.latent_dim)
        zq = zq.permute(0, 3, 1, 2)

        recon_x = self.decoder(zq)

        return (
            recon_x,
            ze,
            zq,
            precision_q,
            logits,
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
            loss_x = self.x_dim * torch.log(mse) / 2
        else:
            loss_x = mse / (2 * self.logvar_x.exp()) + self.x_dim * self.logvar_x / 2

        return loss_x

    def loss_kl_continuous(self, ze, zq, precision_q):
        return torch.sum(((ze - zq) ** 2) * precision_q, dim=(1, 2)).mean()

    def loss_kl_discrete(self, logits):
        prob = F.softmax(logits, dim=-1)
        log_prob = F.log_softmax(logits, dim=-1)
        kl_discreate = torch.sum(prob * log_prob, dim=1).mean()
        return kl_discreate

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

    def loss_pixelcnn(self, logits, logits_prior, eps=1e-10):
        q = logits.softmax(dim=-1)
        q_log = logits.log_softmax(dim=-1)
        p_log = logits_prior.log_softmax(dim=-1)
        kl = torch.sum(q * (q_log - p_log), dim=(1, 2)).mean()

        return kl

    def loss_diffusion(self, predicted_noise, noise):
        return F.mse_loss(predicted_noise, noise)

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
            logits_prior,
            c_logits,
        ) = self(x, True)

        # ELBO loss
        lrc_x = self.loss_x(x, recon_x)
        kl_continuous = self.loss_kl_continuous(ze, zq, precision_q)
        kl_discrete = self.loss_kl_discrete(logits)

        # clustering loss
        lc_elbo = self.loss_c_elbo(c_logits)
        lc_real = self.loss_c_real(c_logits, labels)

        # generative loss
        c_probs = gumbel_softmax_sample(c_logits, self.cls_head.temperature)
        if self.config.gen_model == "pixelcnn":
            indices = logits.argmax(-1)
            indices = indices.view(self.bs, self.latent_size[0], self.latent_size[1])
            logits_prior = self.pixelcnn(indices, c_probs)
            logits_prior = logits_prior.view(self.bs, self.book_size, -1).permute(
                0, 2, 1
            )
            lg = self.loss_pixelcnn(logits, logits_prior)
        elif self.config.gen_model == "diffusion":
            predicted_noise, noise = self.diffusion.train_step(ze, c_probs)
            lg = self.loss_diffusion(predicted_noise, noise)
        else:
            raise NotImplementedError

        loss_total = (
            lrc_x * self.config.lmd_lrc
            + kl_continuous * self.config.lmd_klc
            + kl_discrete * self.config.lmd_kld
            + lc_elbo * self.config.lmd_c_elbo
            + lc_real * self.config.lmd_c_real
            + lg * self.config.lmd_lg
        )

        loss_dict = dict(
            x=lrc_x.item(),
            kl_discrete=kl_discrete.item(),
            kl_continuous=kl_continuous.item(),
            log_param_q=self.quantizer.log_param_q.item(),
            log_param_q_cls=self.cls_head.log_param_q_cls.item(),
            c_elbo=lc_elbo.item(),
            c_real=lc_real.item(),
            g=lg.item(),
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
        c_probs = c_probs.to(self.device, torch.float32)
        nsamples = c_probs.size(0)

        if self.config.gen_model == "pixelcnn":
            indices = self.pixelcnn.sample_prior(c_probs)
            indices = indices.view(nsamples, -1)
            zq = self.quantizer.sample_zq_from_indices(indices)
        elif self.config.gen_model == "diffusion":
            z = self.diffusion.sample(c_probs)
            zq, precision_q, logits = self.quantizer(z, False)
        else:
            raise NotImplementedError

        # generate samples
        h, w = self.latent_size
        zq = zq.view(nsamples, h, w, self.latent_dim)
        generated_x = self.decoder(zq.permute(0, 3, 1, 2))
        generated_x = generated_x.permute(0, 2, 3, 1)

        results = []
        for i in range(nsamples):
            data = {
                "gen_x": generated_x[i].detach().cpu().numpy(),
                "zq": zq[i].detach().cpu().numpy(),
                "gt": c_probs[i].argmax(dim=-1).cpu(),
            }
            results.append(data)

        return results
