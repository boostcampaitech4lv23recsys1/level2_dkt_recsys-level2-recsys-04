import pandas as pd
import torch
import config
from config import CFG, logging_conf
from lightgcn.datasets import prepare_dataset
from lightgcn.models import build, train
from lightgcn.utils import class2dict, get_logger
import wandb


logger = get_logger(logging_conf)
use_cuda = torch.cuda.is_available() and CFG.use_cuda_if_available
device = torch.device("cuda" if use_cuda else "cpu")
print(device)


def main():
    wandb.init(config=class2dict(CFG))

    logger.info("Task Started")

    logger.info("[1/1] Data Preparing - Start")
    train_data, valid_data, test_data, n_node = prepare_dataset(
        device, CFG.basepath, verbose=CFG.loader_verbose, logger=logger.getChild("data")
    )
    logger.info("[1/1] Data Preparing - Done")

    logger.info("[2/2] Model Building - Start")
    model = build(
        n_node,
        embedding_dim=wandb.config.embedding_dim,
        num_layers=wandb.config.num_layers,
        alpha=wandb.config.alpha,
        logger=logger.getChild("build"),
        **CFG.build_kwargs
    )
    model.to(device)

    if CFG.user_wandb:
        wandb.watch(model)

    logger.info("[2/2] Model Building - Done")

    logger.info("[3/3] Model Training - Start")
    train(
        model,
        train_data,
        valid_data = valid_data, # 베이스라인 대비 추가
        n_epoch=wandb.config.n_epoch,
        learning_rate=wandb.config.learning_rate,
        lr_decay = wandb.config.lr_decay,
        gamma = wandb.config.gamma,
        use_wandb=CFG.user_wandb,
        weight=CFG.weight_basepath,
        logger=logger.getChild("train"),
    )
    logger.info("[3/3] Model Training - Done")

    logger.info("Task Complete")


if __name__ == "__main__":
    if CFG.user_wandb:
        wandb.login()
        sweep_id = wandb.sweep(sweep=config.sweep_configuration)
        wandb.agent(sweep_id=sweep_id, function=main)
    else:
        main()
