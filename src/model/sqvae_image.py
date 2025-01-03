from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch import LightningModule
from timm.models.vision_transformer import VisionTransformer
from timm.scheduler.cosine_lr import CosineLRScheduler

from src.model.module.cifar10 import Decoder as Decoder_CIFAR10
from src.model.module.cifar10 import Encoder as Encoder_CIFAR10
from src.model.module.diffusion import DiffusionModule
from src.model.module.mnist import Decoder as Decoder_MNIST
from src.model.module.mnist import Encoder as Encoder_MNIST
from src.model.module.quantizer import GaussianVectorQuantizer, gumbel_softmax_sample


class CSQVAE(LightningModule):
    def __init__(self, config: SimpleNamespace, is_train_diffusion: bool = False):
        super().__init__()
        self.config = config
        self.is_train_diffusion = is_train_diffusion

        self.dataset_name = config.name
        self.x_dim = config.x_dim

        self.n_clusters = config.n_clusters
        self.temp_init_cls = config.temp_init_cls
        self.temp_decay_cls = config.temp_decay_cls
        self.temp_min_cls = config.temp_min_cls
        self.temperature_cls = None
        log_param_q_cls = np.log(config.param_q_init_cls)
        self.log_param_q_cls = nn.Parameter(
            torch.tensor(log_param_q_cls, dtype=torch.float32)
        )
        self.temp_init = config.temp_init
        self.temp_decay = config.temp_decay
        self.temp_min = config.temp_min
        self.temperature = None
        log_param_q = np.log(config.param_q_init)
        self.log_param_q = nn.Parameter(torch.tensor(log_param_q, dtype=torch.float32))

        self.book_size = config.book_size
        self.latent_dim = config.latent_dim
        self.latent_size = config.latent_size

        self.encoder = None
        self.decoder = None
        self.cls_head = None
        self.quantizer = None
        self.pixelcnn = None
        self.diffusion = None
        self.flg_arelbo = True

    def configure_model(self):
        if self.encoder is not None:
            return

        if self.dataset_name == "mnist":
            self.encoder = Encoder_MNIST(self.config)
            self.decoder = Decoder_MNIST(self.config)
        elif self.dataset_name == "cifar10":
            self.encoder = Encoder_CIFAR10(self.config)
            self.decoder = Decoder_CIFAR10(self.config)
        self.cls_head = VisionTransformer(
            self.config.x_shape[1:],
            self.config.patch_size,
            self.config.x_shape[0],
            self.config.n_clusters,
            embed_dim=self.config.latent_dim_cls,
            depth=self.config.n_blocks_cls,
            num_heads=self.config.n_heads_cls,
        )
        self.quantizer = GaussianVectorQuantizer(self.config)
        self.diffusion = DiffusionModule(self.config, with_condition=True)

        if not self.flg_arelbo:
            self.logvar_x = nn.Parameter(torch.tensor(np.log(0.1)))

        if self.is_train_diffusion:
            for p in self.parameters():
                p.requires_grad_(False)
            self.diffusion.requires_grad_(True)
            self.diffusion.model.rotary_emb.freqs.requires_grad_(False)
        else:
            self.init_weights()

    def init_weights(self):
        def _weights_init(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            classname = m.__class__.__name__
            if classname.find("Conv") != -1:
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif classname.find("BatchNorm") != -1:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)

        self.apply(_weights_init)

        # Zero-out adaLN modulation layers in DiT blocks:
        nn.init.constant_(self.diffusion.model.emb_c.weight, 0)
        nn.init.constant_(self.diffusion.model.emb_c.bias, 0)
        for block in self.diffusion.model.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.diffusion.model.fin.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.diffusion.model.fin.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.diffusion.model.fin.linear.weight, 0)
        nn.init.constant_(self.diffusion.model.fin.linear.bias, 0)

    def configure_optimizers(self):
        if not self.is_train_diffusion:
            opt = torch.optim.AdamW(self.parameters(), lr=self.config.optim.csqvae.lr)
            sch = CosineLRScheduler(
                opt,
                t_initial=self.config.optim.csqvae.epochs,
                lr_min=self.config.optim.csqvae.lr_min,
                warmup_t=self.config.optim.csqvae.warmup_t,
                warmup_lr_init=self.config.optim.csqvae.warmup_lr_init,
                warmup_prefix=True,
            )
            return [opt], [{"scheduler": sch, "interval": "epoch"}]
        else:
            opt = torch.optim.AdamW(
                self.parameters(), lr=self.config.optim.diffusion.lr
            )
            # sch = CosineLRScheduler(
            #     opt,
            #     t_initial=self.config.optim.diffusion.epochs,
            #     lr_min=self.config.optim.diffusion.lr_min,
            #     warmup_t=self.config.optim.diffusion.warmup_t,
            #     warmup_lr_init=self.config.optim.diffusion.warmup_lr_init,
            #     warmup_prefix=True,
            # )
            return opt

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step(epoch=self.current_epoch)

    def forward(self, x, is_train):
        b = x.size(0)

        # classification
        c_logits = self.cls_head(x)
        if is_train and not self.is_train_diffusion:
            param_q = self.log_param_q_cls.exp()
            precision_q = 0.5 / torch.clamp(param_q, min=1e-10)
            c_logits_scaled = c_logits * precision_q
            c_probs = gumbel_softmax_sample(c_logits_scaled, self.temperature_cls)
        else:
            c_probs = c_logits.softmax(-1)

        # encoding
        z = self.encoder(x)
        z = z.permute(0, 2, 3, 1).contiguous()
        z = z.view(b, -1, self.latent_dim)

        # quantization
        zq, precision_q, logits, mu = self.quantizer(
            z,
            c_probs,
            self.log_param_q,
            self.temperature,
            is_train and not self.is_train_diffusion,
        )

        h, w = self.latent_size
        z = z.view(b, h, w, self.latent_dim)
        z = z.permute(0, 3, 1, 2)
        zq = zq.view(b, h, w, self.latent_dim)
        zq = zq.permute(0, 3, 1, 2)

        # decoding
        recon_x = self.decoder(zq)

        if is_train:
            return recon_x, z, zq, logits, precision_q, c_logits, c_probs, mu
        else:
            return recon_x, z, zq, logits, c_probs

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

    def loss_c(self, c_logits, labels):
        c_prob = F.softmax(c_logits, dim=-1)
        if torch.all(labels == -1):  # all samples are unlabeled
            lc_labeled = torch.Tensor([0.0]).to(self.device)

            c_log_prob = F.log_softmax(c_logits, dim=-1)
            lc_unlabeled = torch.sum(c_prob * c_log_prob, dim=1).mean()
        elif torch.all(labels != -1):  # all samples are unlabeled
            lc_labeled = F.cross_entropy(c_prob, labels, reduction="sum")
            lc_labeled = lc_labeled / c_prob.size(0)

            lc_unlabeled = torch.Tensor([0.0]).to(self.device)
        else:
            mask_supervised = labels != -1
            lc_labeled = F.cross_entropy(
                c_prob[mask_supervised], labels[mask_supervised], reduction="sum"
            )
            lc_labeled = lc_labeled / c_prob[mask_supervised].size(0)

            c_log_prob = F.log_softmax(c_logits, dim=-1)
            lc_unlabeled = torch.sum(
                c_prob[~mask_supervised] * c_log_prob[~mask_supervised], dim=1
            ).mean()

        return lc_labeled, lc_unlabeled

    def loss_kl_logits(self, logits, logits_prior, eps=1e-10):
        q = logits.softmax(dim=-1)
        q_log = logits.log_softmax(dim=-1)
        p_log = logits_prior.log_softmax(dim=-1)
        kl = torch.sum(q * (q_log - p_log), dim=(1, 2)).mean()

        return kl

    def loss_diffusion(self, predicted_noise, noise):
        return F.mse_loss(predicted_noise, noise, reduction="sum") / noise.size(0)

    def training_step(self, batch, batch_idx):
        if not self.is_train_diffusion:
            return self.training_step_csqvae(batch)
        else:
            return self.training_step_diffusion(batch)

    def on_train_epoch_end(self):
        if not self.is_train_diffusion:
            epochs = self.config.optim.csqvae.epochs
        else:
            epochs = self.config.optim.diffusion.epochs
        print(f"\nEpoch {self.current_epoch} / {epochs}: Done.\n")

    def training_step_csqvae(self, batch):
        x, labels = batch
        b = x.size(0)

        # update temperature of gumbel softmax
        temp_cur = self.calc_temperature(
            self.temp_init_cls, self.temp_decay_cls, self.temp_min_cls
        )
        self.temperature_cls = temp_cur
        temp_cur = self.calc_temperature(self.temp_init, self.temp_decay, self.temp_min)
        self.temperature = temp_cur

        recon_x, z, zq, logits, precision_q, c_logits, c_probs, mu = self(x, True)

        # samplig z_prior from diffusion
        zq_flat = zq.view(b, self.latent_dim, -1).permute(0, 2, 1)
        pred_noise, noise, zq_prior = self.diffusion.train_step(
            zq_flat, c_probs, mu, is_c_onehot=False
        )
        logits_prior = self.quantizer.calc_distance(
            zq_prior.view(-1, self.latent_dim), precision_q
        )
        logits_prior = logits_prior.view(b, -1, self.book_size)

        # scaling logits_prior
        logits_prior = logits_prior * precision_q

        # ELBO loss
        lrc_x = self.loss_x(x, recon_x)
        kl_continuous = self.loss_kl_continuous(z, zq, precision_q)
        kl_discrete = self.loss_kl_logits(logits, logits_prior)
        ldt = self.loss_diffusion(pred_noise, noise)
        # lc_elbo = self.loss_c_elbo(c_logits)

        # clustering loss of labeled data
        # lc_real = self.loss_c_real(c_logits, labels)
        lc_real, lc_elbo = self.loss_c(c_logits, labels)

        loss_total = (
            lrc_x * self.config.loss.csqvae.lmd_x
            + kl_continuous * self.config.loss.csqvae.lmd_klc
            + kl_discrete * self.config.loss.csqvae.lmd_kld
            + ldt * self.config.loss.csqvae.lmd_ldt
            + lc_elbo * self.config.loss.csqvae.lmd_c_elbo
            + lc_real * self.config.loss.csqvae.lmd_c_real
        )
        loss_dict = dict(
            x=lrc_x.item(),
            klc=kl_continuous.item(),
            kld=kl_discrete.item(),
            ldt=ldt.item(),
            log_param_q=self.log_param_q.item(),
            log_param_q_cls=self.log_param_q_cls.item(),
            c_elbo=lc_elbo.item(),
            c_real=lc_real.item(),
            total=loss_total.item(),
        )
        self.log_dict(loss_dict, prog_bar=True, logger=True)

        return loss_total

    def training_step_diffusion(self, batch):
        x, labels = batch
        b = x.size(0)
        recon_x, z, zq, logits, precision_q, c_logits, c_probs, mu = self(x, True)

        # samplig z_prior from diffusion
        self.diffusion.send_sigma_to_device(self.device)
        zq_flat = zq.view(b, self.latent_dim, -1).permute(0, 2, 1)
        pred_noise, noise, zq_prior = self.diffusion.train_step(
            zq_flat, c_probs, mu, is_c_onehot=True
        )
        logits_prior = self.quantizer.calc_distance(
            zq_prior.view(-1, self.latent_dim), precision_q
        )
        logits_prior = logits_prior.view(b, -1, self.book_size)

        # scaling logits_prior
        logits_prior = logits_prior * precision_q

        # loss
        kl_discrete = self.loss_kl_logits(logits, logits_prior)
        ldt = self.loss_diffusion(pred_noise, noise)

        loss_total = (
            kl_discrete * self.config.loss.diffusion.lmd_kld
            + ldt * self.config.loss.diffusion.lmd_ldt
        )
        loss_dict = dict(
            kld=kl_discrete.item(),
            ldt=ldt.item(),
            total=loss_total.item(),
        )
        self.log_dict(loss_dict, prog_bar=True, logger=True)
        return loss_total

    def predict_step(self, batch):
        x, labels = batch

        recon_x, z, zq, logits, c_probs = self(x, False)

        x = x.permute(0, 2, 3, 1)
        recon_x = recon_x.permute(0, 2, 3, 1)
        z = z.permute(0, 2, 3, 1)
        zq = zq.permute(0, 2, 3, 1)
        zq_probs = logits.softmax(dim=-1)

        results = []
        for i in range(len(x)):
            data = {
                "x": x[i].detach().cpu().numpy(),
                "recon_x": recon_x[i].detach().cpu().numpy(),
                "ze": z[i].detach().cpu().numpy(),
                "zq": zq[i].detach().cpu().numpy(),
                "book_prob": zq_probs[i].detach().cpu().numpy(),
                "book_idx": zq_probs[i].detach().cpu().numpy().argmax(axis=1),
                "label_prob": c_probs[i].detach().cpu().numpy(),
                "label": c_probs[i].detach().cpu().numpy().argmax(),
                "label_gt": labels[i].detach().cpu().numpy().item(),
            }
            results.append(data)

        return results

    @torch.no_grad()
    def sample(self, c_probs: torch.Tensor, with_diffusion: bool = True):
        c_probs = c_probs.to(self.device, torch.float32)
        nsamples = c_probs.size(0)
        h, w = self.latent_size

        if with_diffusion:
            z = self.diffusion.sample(c_probs, self.quantizer.mu)
            zq = self.quantizer.z_to_zq(z, self.log_param_q)
        else:
            z = torch.randn((nsamples, h * w, self.latent_dim)).to(self.device)
            zq, precision_q, logits, mu = self.quantizer(
                z, c_probs, self.log_param_q, self.temperature, False
            )

        # generate samples
        z = z.view(nsamples, h, w, self.latent_dim)
        zq = zq.view(nsamples, h, w, self.latent_dim).permute(0, 3, 1, 2)
        generated_x = self.decoder(zq)
        generated_x = generated_x.permute(0, 2, 3, 1)

        zq = zq.permute(0, 2, 3, 1)
        results = []
        for i in range(nsamples):
            data = {
                "gen_x": generated_x[i].detach().cpu().numpy(),
                "z": z[i].detach().cpu().numpy(),
                "zq": zq[i].detach().cpu().numpy(),
                "gt": c_probs[i].argmax(dim=-1).cpu(),
            }
            results.append(data)

        return results
