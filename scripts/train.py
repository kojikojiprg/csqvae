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
from src.model.sqvae_image import CSQVAE
from src.utils import yaml_handler
from src.utils.train_stage import TrainStages


def create_checkpoint_dir(dataset_name):
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
        if "WORLD_SIZE" not in os.environ:
            v_num = max_v_num + 1
    else:
        v_num = 0
    checkpoint_dir = os.path.join(checkpoint_dir, str(v_num))
    if "WORLD_SIZE" not in os.environ:
        # copy config
        os.makedirs(checkpoint_dir, exist_ok=False)
        copy_config_path = os.path.join(checkpoint_dir, f"{config_name}.yaml")
        shutil.copyfile(config_path, copy_config_path)
    return checkpoint_dir, v_num


def load_dataset(dataset_name, config, checkpoint_dir, train_stage):
    summary_path = f"{checkpoint_dir}/summary_train_labels.tsv"
    if dataset_name == "mnist":
        dataset = MNIST(
            train_stage,
            config.n_labeled_samples,
            summary_path=summary_path,
        )
    elif dataset_name == "cifar10":
        dataset = CIFAR10(
            train_stage,
            config.n_labeled_samples,
            summary_path=summary_path,
        )
    return dataset


def get_gpu_index():
    if "LOCAL_RANK" in os.environ:
        gpu_index = int(os.environ["LOCAL_RANK"])
    else:
        gpu_index = 0
    return gpu_index


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["mnist", "cifar10"], type=str)
    parser.add_argument("-g", "--gpu_ids", type=int, nargs="*", default=None)
    parser.add_argument("-pre", "--pretrain", action="store_true", default=False)
    parser.add_argument("-tc", "--train_csqvae", action="store_true", default=False)
    parser.add_argument("-ft", "--finetuning", action="store_true", default=False)
    args = parser.parse_args()
    dataset_name = args.dataset
    gpu_ids = args.gpu_ids
    pretrain = args.pretrain
    train_csqvae = args.train_csqvae
    finetuning = args.finetuning

    # load config
    config_name = dataset_name
    config_path = f"configs/{config_name}.yaml"
    config = yaml_handler.load(config_path)

    # create checkpoint directory of this version
    checkpoint_dir, v_num = create_checkpoint_dir(dataset_name)

    # get gpu index
    gpu_index = get_gpu_index()

    train_stages = TrainStages(pretrain, train_csqvae, finetuning)
    train_stage_last = train_stages[0][0]
    for train_stage, train_flag in train_stages:
        print(train_stage, train_flag)
        if train_flag or v_num == 0:
            print(f"Training {train_stage}")

            # load dataset
            dataset = load_dataset(dataset_name, config, checkpoint_dir, train_stage)
            dataloader = DataLoader(
                dataset,
                eval(f"config.optim.{train_stage}.batch_size"),
                shuffle=True,
                num_workers=config.optim.num_workers,
                pin_memory=True,
            )

            if train_stage == "sqvae":
                # create model
                model = CSQVAE(config, train_stage)
            else:
                # load model
                checkpoint_path = sorted(
                    glob(f"{checkpoint_dir}/{train_stage_last}*.ckpt")
                )[-1]
                model = CSQVAE.load_from_checkpoint(
                    checkpoint_path,
                    map_location=f"cuda:{gpu_ids[gpu_index]}",
                    config=config,
                    train_stage=train_stage,
                )
                model.configure_model()
                if train_stage == "classification":
                    print("Initiallize mu and sigma of Classification")
                    model.init_mu_and_sigma(dataset)
                elif train_stage == "csqvae":
                    # print("Initiallize log_sigma_q of CSQ-VAE")
                    # model.init_log_sigma_q()
                    print("Initiallize mu and sigma of Classification")
                    model.init_mu_and_sigma(dataset)

            # model checkpoint callback
            filename = f"{train_stage}-v{v_num}-{dataset_name}"
            model_checkpoint = ModelCheckpoint(
                checkpoint_dir,
                filename=filename + "-best-{epoch}",
                monitor="loss",
                mode="min",
                save_last=True,
            )
            model_checkpoint.CHECKPOINT_NAME_LAST = filename + "-last-{epoch}"

            epochs = eval(f"config.optim.{train_stage}.epochs")
            agb = eval(f"config.optim.{train_stage}.accumulate_grad_batches")

            ddp = DDPStrategy(
                find_unused_parameters=False, process_group_backend="nccl"
            )
            logger = TensorBoardLogger(
                "logs", name=dataset_name, version=f"{train_stage}_{v_num}"
            )
            trainer = Trainer(
                accelerator="cuda",
                strategy=ddp,
                devices=gpu_ids,
                logger=logger,
                callbacks=[model_checkpoint],
                max_epochs=epochs,
                accumulate_grad_batches=agb,
                benchmark=True,
            )
            trainer.fit(model, train_dataloaders=dataloader)
            del dataset, dataloader, trainer
            torch.cuda.empty_cache()

        else:
            if "WORLD_SIZE" not in os.environ:
                # copy pretrained checkpoint
                for v_num_pre in reversed(range(v_num)):
                    checkpoint_dir_pre = f"models/{dataset_name}/{v_num_pre}"
                    checkpoint_path_pre = sorted(
                        glob(f"{checkpoint_dir_pre}/{train_stage}-*.ckpt")
                    )
                    if len(checkpoint_path_pre) >= 1:
                        checkpoint_path_pre = checkpoint_path_pre[-1]
                        checkpoint_path = (
                            f"{checkpoint_dir}/{os.path.basename(checkpoint_path_pre)}"
                        )
                        shutil.copyfile(checkpoint_path_pre, checkpoint_path)
                        break

        train_stage_last = train_stage
