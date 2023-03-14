import os
import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelSummary, ModelCheckpoint 
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from model import convnet_sc,LitConvNet
from data import MagnetogramDataModule
import pandas as pd
import numpy as np
import wandb
from pytorch_lightning.loggers import WandbLogger
import yaml

def main():
    # read in config file
    with open('experiment_config.yml') as config_file:
        config = yaml.safe_load(config_file.read())
    
    wandb.init(config=config,project=config['meta']['project'])
    config = wandb.config

    data_file = config.data['data_file']
    window = config.data['forecast_window']
    dim = config.data['dim']
    split_type = config.data['split_type']
    balance_ratio = config.data['balance_ratio']
    label = config.data['label']
    dropout_ratio = config.model['dropout_ratio']
    lr = config.training['lr']
    wd = config.training['wd']
    batch = config.training['batch_size']
    epochs = config.training['epochs']

    # set seeds
    pl.seed_everything(42,workers=True)

    # define data module
    data = MagnetogramDataModule(data_file=data_file,
                                 label=label,
                                 balance_ratio=balance_ratio,
                                 split_type=split_type,
                                 forecast_window=window,dim=dim,batch=batch)

    # define model
    model = convnet_sc(dim=dim,length=1,dropoutRatio=dropout_ratio)
    classifier = LitConvNet(model,lr,wd,epochs=epochs)

    # initialize wandb logger
    wandb_logger = WandbLogger(log_model='all')
    checkpoint_callback = ModelCheckpoint(monitor='val_tss',mode='max')
    early_stop_callback = EarlyStopping(monitor='val_loss',min_delta=0.0,patience=10,mode='min')

    # train model
    trainer = pl.Trainer(deterministic=True,
                         max_epochs=epochs,
                        #  log_every_n_steps=4,
                         callbacks=[ModelSummary(max_depth=2),checkpoint_callback,early_stop_callback],
                        #  limit_train_batches=15,
                        #  limit_val_batches=5,
                         logger=wandb_logger)
    trainer.fit(model=classifier,datamodule=data)

    # evaluate model

if __name__ == "__main__":
    main()
