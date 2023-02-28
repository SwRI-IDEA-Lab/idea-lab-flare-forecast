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
    datafile = 'Data/labels_mdi_small.csv'
    window = 24     # forecast window (hours)
    dim = 256

    # set seeds
    pl.seed_everything(42,workers=True)

    # define data module
    data = MagnetogramDataModule(data_file=datafile,forecast_window=window,dim=dim)

    # define model
    model = convnet_sc(dim=dim,length=1,dropoutRatio=0)
    classifier = LitConvNet(model)

    # initialize wandb logger
    wandb_logger = WandbLogger(project='flare-forecast')

    # add parameters to wandb config
    wandb_logger.experiment.config['forecast_window'] = window

    # train model
    trainer = pl.Trainer(deterministic=True,max_epochs=100,callbacks=[ModelSummary(max_depth=2)],log_every_n_steps=1,logger=wandb_logger)
    trainer.fit(model=classifier,datamodule=data)

    # evaluate model
    predictions = trainer.predict(classifier,data.val_dataloader())
    y_pred = predictions[0][0]
    y_true = predictions[0][1]
    print('Predicted: ',torch.transpose(y_pred,0,1))
    print('True:', torch.transpose(y_true,0,1))

if __name__ == "__main__":
    main()
