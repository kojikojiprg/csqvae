import argparse
import os
import shutil
import sys
from glob import glob

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy

sys.path.append(".")
from src.data import individual_train_dataloader
from src.model import SQVAE, VAE
from src.utils import yaml_handler

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root", type=str)
    parser.add_argument("-g", "--gpu_ids", type=int, nargs="*", default=None)
    parser.add_argument(
        "-mt", "--model_type", required=False, type=str, default="sqvae"
    )
    parser.add_argument("-ckpt", "--checkpoint", required=False, type=str, default=None)
    args = parser.parse_args()
    data_root = args.data_root
    model_type = args.model_type
    gpu_ids = args.gpu_ids
    pre_checkpoint_path = args.checkpoint

    # load config
    config_path = f"configs/individual-{model_type}.yaml"
    config = yaml_handler.load(config_path)

    # create checkpoint directory of this version
    checkpoint_dir = f"models/individual/{model_type}"
    ckpt_dirs = glob(os.path.join(checkpoint_dir, "*/"))
    ckpt_dirs = [d for d in ckpt_dirs if "version_" in d]
    if len(ckpt_dirs) > 0:
        max_v_num = 0
        for d in ckpt_dirs:
            last_ckpt_dir = os.path.dirname(d)
            v_num = int(last_ckpt_dir.split("/")[-1].replace("version_", ""))
            if v_num > max_v_num:
                max_v_num = v_num
        v_num = max_v_num + 1
    else:
        v_num = 0
    checkpoint_dir = os.path.join(checkpoint_dir, f"version_{v_num}")

    if "WORLD_SIZE" not in os.environ:
        # copy config
        os.makedirs(checkpoint_dir, exist_ok=False)
        copy_config_path = os.path.join(checkpoint_dir, f"individual-{model_type}.yaml")
        shutil.copyfile(config_path, copy_config_path)

    # model checkpoint callback
    h, w = config.img_size
    filename = f"{model_type}-seq_len{config.seq_len}-stride{config.stride}-{h}x{w}"
    model_checkpoint = ModelCheckpoint(
        checkpoint_dir,
        filename=filename + "-best-{epoch}",
        monitor="loss",
        mode="min",
        save_last=True,
    )
    model_checkpoint.CHECKPOINT_NAME_LAST = filename + "-last-{epoch}"

    # load dataset
    dataloader, n_batches = individual_train_dataloader(
        data_root, "individual", config, gpu_ids
    )

    # create model
    ann_path = "../datasets/dataset03/annotation/role.txt"
    if model_type == "vae":
        model = VAE(config, n_batches, annotation_path=ann_path)
        # model = VAE(config, n_batches)
        ddp = DDPStrategy(find_unused_parameters=True, process_group_backend="nccl")
    elif model_type == "sqvae":
        model = SQVAE(config, annotation_path=ann_path)
        ddp = DDPStrategy(find_unused_parameters=False, process_group_backend="nccl")
    accumulate_grad_batches = config.accumulate_grad_batches

    logger = TensorBoardLogger("logs/individual", name=model_type)
    trainer = Trainer(
        accelerator="cuda",
        strategy=ddp,
        devices=gpu_ids,
        logger=logger,
        callbacks=[model_checkpoint],
        max_epochs=config.epochs,
        accumulate_grad_batches=accumulate_grad_batches,
        benchmark=True,
    )
    trainer.fit(model, train_dataloaders=dataloader, ckpt_path=pre_checkpoint_path)
