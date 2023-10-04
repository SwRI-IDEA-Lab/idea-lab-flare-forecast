import sys,os
sys.path.append(os.getcwd())

import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelSummary, ModelCheckpoint 
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from src.model_classifier import convnet_sc,LitConvNet
from model_regressor import convnet_sc_regressor,LitConvNetRegressor
from data import MagnetogramDataModule
from data_zarr import AIAHMIDataModule
from utils.model_utils import *
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

    
    # load config from specified wandb run
    run = wandb.init(project=config['meta']['project'],resume='must',id=config['meta']['id'])
    config = wandb.config

    if config.data['regression']:
        litclass = LitConvNetRegressor
        modelclass = convnet_sc_regressor
    else:
        litclass = LitConvNet
        modelclass = convnet_sc

    dim = config.data['dim']
    batch = config.training['batch_size']
    test = config.data['test']

    # set seeds
    pl.seed_everything(42,workers=True)

    # define data module
    if not config.data['use_zarr_dataset']:
        data = MagnetogramDataModule(data_file=config.data['data_file'],
                                    label=config.data['label'],
                                    balance_ratio=config.data['balance_ratio'],
                                    regression=config.data['regression'],
                                    val_split=config.data['val_split'],
                                    forecast_window=config.data['forecast_window'],
                                    dim=config.data['dim'],
                                    batch=config.training['batch_size'],
                                    augmentation=config.data['augmentation'],
                                    flare_thresh=config.data['flare_thresh'],
                                    flux_thresh=config.data['flux_thresh'],
                                    feature_cols=config.data['feature_cols'],
                                    test=config.data['test'],
                                    maxval=config.data['maxval'],
                                    file_col=config.data['file_col'])
    else:
        data = AIAHMIDataModule(zarr_file=config.data['zarr_file'],
                            val_split=config.data['val_split'],
                            data_file=config.data['data_file'],
                            regression=config.data['regression'],
                            forecast_window=config.data['forecast_window'],
                            dim=config.data['dim'],
                            batch=config.training['batch_size'],
                            augmentation=config.data['augmentation'],
                            flare_thresh=config.data['flare_thresh'],
                            feature_cols=config.data['feature_cols'],
                            test=config.data['test'],
                            channels=config.data['channels'],
                            maxvals=config.data['maxval'],)
    
    # define model
    model = modelclass(dim=config.data['dim'],length=len(config.data['channels']),
                                 len_features=len(config.data['feature_cols']),
                                 weights=[],dropoutRatio=config.model['dropout_ratio'])

    # load checkpoint
    classifier = load_model(run, 'kierav/'+config.meta['project']+'/model-'+run.id+':latest', model,litclass=litclass)

    for name, layer in model.named_modules():
        if isinstance(layer, torch.nn.Linear):
            print(name, layer)        
            print(layer.weight)
            print(layer.bias)

    # evaluate model
    trainer = pl.Trainer(accelerator='gpu',
                         devices=1,
                         deterministic=False,
                         logger=False)
    
    data.prepare_data()
    data.setup('test')

    print('------Train/val predictions------')
    preds = trainer.predict(model=classifier,dataloaders=data.trainval_dataloader())
    save_preds(preds,wandb.run.dir,'trainval_results.csv',config.data['regression'])

    print('------Pseudotest predictions------')
    preds = trainer.predict(model=classifier,dataloaders=data.pseudotest_dataloader())
    save_preds(preds,wandb.run.dir,'pseudotest_results.csv',config.data['regression'])

    print('------Test predictions------')
    preds = trainer.predict(model=classifier,dataloaders=data.test_dataloader())
    save_preds(preds,wandb.run.dir,'test_results.csv',config.data['regression'])

    wandb.finish()

if __name__ == "__main__":
    main()
