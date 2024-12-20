import argparse
import os
import shutil
import sys
from glob import glob

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy
from torch.utils.data import DataLoader

sys.path.append(".")
from src.data.cifar10 import CIFAR10
from src.data.mnist import MNIST
from src.model.sqvae_image import CSQVAE
from src.utils import yaml_handler

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["mnist", "cifar10"], type=str)
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("-g", "--gpu_ids", type=int, nargs="*", default=None)
    args = parser.parse_args()
    dataset_name = args.dataset
    pre_checkpoint_path = args.checkpoint
    gpu_ids = args.gpu_ids

    # load config
    config_name = dataset_name + "_finetuning"
    config_path = f"configs/{config_name}.yaml"
    config = yaml_handler.load(config_path)

    # create checkpoint directory of this version
    checkpoint_dir = f"models/{dataset_name}"
    ckpt_dirs = glob(os.path.join(checkpoint_dir, "*/"))
    version_prefix = "finetuning_"
    ckpt_dirs = [d for d in ckpt_dirs if version_prefix in d]
    if len(ckpt_dirs) > 0:
        max_v_num = 0
        for d in ckpt_dirs:
            last_ckpt_dir = os.path.dirname(d)
            v_num = int(last_ckpt_dir.split("/")[-1].replace(version_prefix, ""))
            if v_num > max_v_num:
                max_v_num = v_num
        v_num = max_v_num + 1
    else:
        v_num = 0
    version = f"{version_prefix}{v_num}"
    checkpoint_dir = os.path.join(checkpoint_dir, version)

    if "WORLD_SIZE" not in os.environ:
        # copy config
        os.makedirs(checkpoint_dir, exist_ok=False)
        copy_config_path = os.path.join(checkpoint_dir, f"{config_name}.yaml")
        shutil.copyfile(config_path, copy_config_path)

    # model checkpoint callback
    filename = f"csqvae-{dataset_name}-d{config.latent_dim}-bs{config.book_size}_finetuned.ckpt"
    model_checkpoint = ModelCheckpoint(
        checkpoint_dir,
        filename=filename + "-best-{epoch}",
        monitor="loss",
        mode="min",
        save_last=True,
    )
    model_checkpoint.CHECKPOINT_NAME_LAST = filename + "-last-{epoch}"

    # load dataset
    summary_path = f"{checkpoint_dir}/summary_train_labels.tsv"
    if dataset_name == "mnist":
        dataset = MNIST(True, config.n_labeled_samples, 42, "data/", True, summary_path)
    elif dataset_name == "cifar10":
        dataset = CIFAR10(
            True, config.n_labeled_samples, 42, "data/", True, summary_path
        )
    dataloader = DataLoader(
        dataset,
        config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    # load model
    model = CSQVAE.load_from_checkpoint(
        pre_checkpoint_path,
        map_location=f"cuda:{gpu_ids[0]}",
        config=config,
        is_finetuning=True,
    )
    model.configure_model()

    ddp = DDPStrategy(find_unused_parameters=False, process_group_backend="nccl")
    logger = TensorBoardLogger("logs", name=dataset_name, version=version)
    trainer = Trainer(
        accelerator="cuda",
        strategy=ddp,
        devices=gpu_ids,
        logger=logger,
        callbacks=[model_checkpoint],
        max_epochs=config.epochs,
        accumulate_grad_batches=config.accumulate_grad_batches,
        benchmark=True,
    )
    trainer.fit(model, train_dataloaders=dataloader)
