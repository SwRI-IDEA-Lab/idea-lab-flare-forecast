import sys,os
sys.path.append(os.getcwd())

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
from pathlib import Path
import numpy as np
import wandb
from pytorch_lightning.loggers import WandbLogger
import yaml

def main():    
    # read in config file
    with open('experiment_config.yml') as config_file:
        config = yaml.safe_load(config_file.read())
    
    run = wandb.init(project=config['meta']['project'],resume='must',id=config['meta']['id'])
    config = wandb.config

    dim = config.data['dim']
    lr = config.training['lr']
    wd = config.training['wd']
    batch = config.training['batch_size']
    epochs = config.training['epochs']

    # set seeds
    pl.seed_everything(42,workers=True)

    # define data module
    data = MagnetogramDataModule(data_file=config.data['data_file'],
                                 label=config.data['label'],
                                 balance_ratio=config.data['balance_ratio'],
                                 split_type=config.data['split_type'],
                                 val_split=config.data['val_split'],
                                 forecast_window=config.data['forecast_window'],
                                 dim=dim,
                                 batch=batch,
                                 augmentation=config.data['augmentation'],
                                 flare_thresh=config.data['flare_thresh'],
                                 flux_thresh=config.data['flux_thresh'])

    # define model
    model = convnet_sc(dim=dim,length=1,dropoutRatio=config.model['dropout_ratio'])
    classifier = LitConvNet(model,lr,wd,epochs=epochs)

    # load checkpoint
    print('Loading model checkpoint from ', 'kierav/'+config.meta['project']+'/model-'+config.meta['id']+':best_k')
    artifact = run.use_artifact('kierav/'+config.meta['project']+'/model-'+config.meta['id']+':best_k',type='model')
    artifact_dir = artifact.download()
    classifier = LitConvNet.load_from_checkpoint(Path(artifact_dir)/'model.ckpt',model=model)
    
    # initialize wandb logger
    wandb_logger = WandbLogger(log_model='all')

    # evaluate model
    trainer = pl.Trainer(accelerator='cpu',
                         devices=1,
                         deterministic=False,
                         logger=wandb_logger)
    
    trainer.test(model=classifier,dataloaders=data.pseudotest_dataloader())

    wandb.finish()


if __name__ == "__main__":
    main()
