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
from src.data.mnist import MNIST
from src.model.sqvae_image import CSQVAE
from src.utils import yaml_handler

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gpu_ids", type=int, nargs="*", default=None)
    parser.add_argument(
        "-ut",
        "--unsupervised_training",
        required=False,
        action="store_true",
        default=False,
    )
    parser.add_argument("-ckpt", "--checkpoint", required=False, type=str, default=None)
    args = parser.parse_args()
    unsupervised_training = args.unsupervised_training
    gpu_ids = args.gpu_ids
    pre_checkpoint_path = args.checkpoint

    # load config
    config_path = "configs/mnist.yaml"
    config = yaml_handler.load(config_path)

    # create checkpoint directory of this version
    checkpoint_dir = "models/mnist"
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
        copy_config_path = os.path.join(checkpoint_dir, "mnist.yaml")
        shutil.copyfile(config_path, copy_config_path)

    # model checkpoint callback
    filename = f"sqvae-mnist-d{config.latent_ndim}-bs{config.book_size}.ckpt"
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
    mnist = MNIST(True, config.n_labeled_samples, 42, "data/", True, summary_path)
    dataloader = DataLoader(
        mnist,
        config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    # create model
    model = CSQVAE(config)
    ddp = DDPStrategy(find_unused_parameters=True, process_group_backend="nccl")
    accumulate_grad_batches = config.accumulate_grad_batches

    logger = TensorBoardLogger("logs", name="mnist")
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
