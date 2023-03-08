import os
import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelSummary
from model import convnet_sc,LitConvNet
from data import MagnetogramDataModule
import pandas as pd
import numpy as np
import wandb
from pytorch_lightning.loggers import WandbLogger

def main():
    # set dataset file and parameters
    datafile = 'Data/labels_MDI.csv'
    window = 24     # forecast window (hours)
    dim = 128
    lr = 1e-4
    wd = 1e-3
    dropout_ratio = 0
    batch = 32

    # set seeds
    pl.seed_everything(42,workers=True)

    # define data module
    data = MagnetogramDataModule(data_file=datafile,forecast_window=window,dim=dim,batch=batch)

    # define model
    model = convnet_sc(dim=dim,length=1,dropoutRatio=dropout_ratio)
    classifier = LitConvNet(model,lr,wd)

    # initialize wandb logger
    wandb_logger = WandbLogger(project='flare-forecast')

    # add parameters to wandb config
    wandb_logger.experiment.config.update({'datafile':datafile,
                                           'forecast_window':window,
                                           'dim':dim,
                                           'dropout_ratio':dropout_ratio,
                                           'batch':batch})

    # train model
    trainer = pl.Trainer(deterministic=True,
                         max_epochs=50,
                         log_every_n_steps=4,
                         callbacks=[ModelSummary(max_depth=2)],
                         limit_train_batches=12,
                         limit_val_batches=5,
                         logger=wandb_logger)
    trainer.fit(model=classifier,datamodule=data)

    # evaluate model

if __name__ == "__main__":
    main()
