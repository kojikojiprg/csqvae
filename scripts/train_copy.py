import argparse
import os
import shutil
import sys
from glob import glob

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy
from torch.utils.data import DataLoader

sys.path.append(".")
from src.data.cifar10 import CIFAR10
from src.data.mnist import MNIST
from src.model.sqvae_image_copy import CSQVAE
from src.utils import yaml_handler

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["mnist", "cifar10"], type=str)
    parser.add_argument("-g", "--gpu_ids", type=int, nargs="*", default=None)
    parser.add_argument("-up", "--use_pretrained", action="store_true", default=False)
    args = parser.parse_args()
    dataset_name = args.dataset
    gpu_ids = args.gpu_ids
    use_pretrained = args.use_pretrained

    # load config
    config_name = dataset_name
    config_path = f"configs/{config_name}.yaml"
    config = yaml_handler.load(config_path)

    # create checkpoint directory of this version
    checkpoint_dir = f"models/{dataset_name}"
    ckpt_dirs = glob(os.path.join(checkpoint_dir, "*/"))
    ckpt_dirs = [d for d in ckpt_dirs if os.path.basename(d[:-1]).isdecimal()]
    if len(ckpt_dirs) > 0:
        max_v_num = 0
        for d in ckpt_dirs:
            last_ckpt_dir = os.path.dirname(d)
            v_num = int(last_ckpt_dir.split("/")[-1])
            if v_num > max_v_num:
                max_v_num = v_num
        v_num = max_v_num + 1
    else:
        v_num = 0
    version = str(v_num)
    checkpoint_dir = os.path.join(checkpoint_dir, version)

    if "WORLD_SIZE" not in os.environ:
        # copy config
        os.makedirs(checkpoint_dir, exist_ok=False)
        copy_config_path = os.path.join(checkpoint_dir, f"{config_name}.yaml")
        shutil.copyfile(config_path, copy_config_path)

    # load dataset
    summary_path = f"{checkpoint_dir}/summary_train_labels.tsv"
    if dataset_name == "mnist":
        dataset = MNIST(True, config.n_labeled_samples, 42, "data/", True, summary_path)
    elif dataset_name == "cifar10":
        dataset = CIFAR10(
            True, config.n_labeled_samples, 42, "data/", True, summary_path
        )

    # ====================================================================================================
    #  Training SQ-VAE
    # ====================================================================================================
    if not use_pretrained:
        print("Training SQ-VAE")
        dataloader = DataLoader(
            dataset,
            config.optim.sqvae.batch_size,
            shuffle=True,
            num_workers=config.optim.num_workers,
            pin_memory=True,
        )

        # create model
        model = CSQVAE(config, "sqvae")

        # model checkpoint callback
        filename = f"sqvae-{dataset_name}-d{config.latent_dim}-bs{config.book_size}"
        model_checkpoint = ModelCheckpoint(
            checkpoint_dir,
            filename=filename + "-best-{epoch}",
            monitor="loss",
            mode="min",
            save_last=True,
        )
        model_checkpoint.CHECKPOINT_NAME_LAST = filename + "-last-{epoch}"

        ddp = DDPStrategy(find_unused_parameters=False, process_group_backend="nccl")
        logger = TensorBoardLogger(
            "logs", name=dataset_name, version=version + "_sqvae"
        )
        trainer = Trainer(
            accelerator="cuda",
            strategy=ddp,
            devices=gpu_ids,
            logger=logger,
            callbacks=[model_checkpoint],
            max_epochs=config.optim.sqvae.epochs,
            accumulate_grad_batches=config.optim.sqvae.accumulate_grad_batches,
            benchmark=True,
        )
        trainer.fit(model, train_dataloaders=dataloader)
        torch.cuda.empty_cache()
    else:
        # copy checkpoint
        checkpoint_dir_pre = f"models/{dataset_name}/pretrain"
        checkpoint_path_pre = sorted(glob(f"{checkpoint_dir_pre}/sqvae-*.ckpt"))[-1]
        checkpoint_path = f"{checkpoint_dir}/{os.path.basename(checkpoint_path_pre)}"
        shutil.copyfile(checkpoint_path_pre, checkpoint_path)

    # ====================================================================================================
    #  Training Diffusion
    # ====================================================================================================
    if not use_pretrained:
        print("Training Diffusion")
        dataloader = DataLoader(
            dataset,
            config.optim.diffusion.batch_size,
            shuffle=True,
            num_workers=config.optim.num_workers,
            pin_memory=True,
        )

        # load model
        checkpoint_path = sorted(glob(f"{checkpoint_dir}/sqvae-*.ckpt"))[-1]
        model = CSQVAE.load_from_checkpoint(
            checkpoint_path,
            map_location=f"cuda:{gpu_ids[0]}",
            config=config,
            train_stage="diffusion",
        )
        model.configure_model()

        # model checkpoint callback
        filename = f"diffusion-{dataset_name}-d{config.latent_dim}-bs{config.book_size}"
        model_checkpoint = ModelCheckpoint(
            checkpoint_dir,
            filename=filename + "-best-{epoch}",
            monitor="loss",
            mode="min",
            save_last=True,
        )
        model_checkpoint.CHECKPOINT_NAME_LAST = filename + "-last-{epoch}"

        ddp = DDPStrategy(find_unused_parameters=False, process_group_backend="nccl")
        logger = TensorBoardLogger(
            "logs", name=dataset_name, version=version + "_diffusion"
        )
        trainer = Trainer(
            accelerator="cuda",
            strategy=ddp,
            devices=gpu_ids,
            logger=logger,
            callbacks=[model_checkpoint],
            max_epochs=config.optim.diffusion.epochs,
            accumulate_grad_batches=config.optim.diffusion.accumulate_grad_batches,
            benchmark=True,
        )
        trainer.fit(model, train_dataloaders=dataloader)
        torch.cuda.empty_cache()
    else:
        # copy checkpoint
        checkpoint_dir_pre = f"models/{dataset_name}/pretrain"
        checkpoint_path_pre = sorted(glob(f"{checkpoint_dir_pre}/diffusion-*.ckpt"))[-1]
        checkpoint_path = f"{checkpoint_dir}/{os.path.basename(checkpoint_path_pre)}"
        shutil.copyfile(checkpoint_path_pre, checkpoint_path)

    # ====================================================================================================
    #  Training CSQVAE
    # ====================================================================================================
    print("Training CSQ-VAE")
    dataloader = DataLoader(
        dataset,
        config.optim.csqvae.batch_size,
        shuffle=True,
        num_workers=config.optim.num_workers,
        pin_memory=True,
    )

    # load model
    checkpoint_path = sorted(glob(f"{checkpoint_dir}/diffusion-*.ckpt"))[-1]
    model = CSQVAE.load_from_checkpoint(
        checkpoint_path,
        map_location=f"cuda:{gpu_ids[0]}",
        config=config,
        train_stage="csqvae",
    )
    model.configure_model()
    print("Initiallize CSQ-VAE")
    model.init_mu(dataset)

    # model checkpoint callback
    filename = f"csqvae-{dataset_name}-d{config.latent_dim}-bs{config.book_size}"
    model_checkpoint = ModelCheckpoint(
        checkpoint_dir,
        filename=filename + "-best-{epoch}",
        monitor="loss",
        mode="min",
        save_last=True,
    )
    model_checkpoint.CHECKPOINT_NAME_LAST = filename + "-last-{epoch}"

    ddp = DDPStrategy(find_unused_parameters=False, process_group_backend="nccl")
    logger = TensorBoardLogger("logs", name=dataset_name, version=version + "_csqvae")
    trainer = Trainer(
        accelerator="cuda",
        strategy=ddp,
        devices=gpu_ids,
        logger=logger,
        callbacks=[model_checkpoint],
        max_epochs=config.optim.csqvae.epochs,
        accumulate_grad_batches=config.optim.csqvae.accumulate_grad_batches,
        benchmark=True,
    )
    trainer.fit(model, train_dataloaders=dataloader)
