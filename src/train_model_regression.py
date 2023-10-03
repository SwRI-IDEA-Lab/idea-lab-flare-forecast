import sys,os
sys.path.append(os.getcwd())

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelSummary, ModelCheckpoint 
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from model_regressor import convnet_sc_regressor,LitConvNetRegressor
from data import MagnetogramDataModule
from src.mlp_model_regressor import MLPModel
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
    
    if config['meta']['resume']:
        run = wandb.init(config=config,project=config['meta']['project'],resume='must',id=config['meta']['id'])
    else:
        run = wandb.init(config=config,project=config['meta']['project'])
    config = wandb.config

    # set seeds
    pl.seed_everything(42,workers=True)

    #
    print('Features:',config.data['feature_cols'])

    # define data module
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

    # train MLP model to obtain weights for final layers of CNN+MLP
    if len(config.data['feature_cols'])>0:
        mlp_model = MLPModel(data_file=config.data['data_file'],
                            window=config.data['forecast_window'],
                            val_split=config.data['val_split'],
                            flare_thresh=config.data['flare_thresh'],
                            features=config.data['feature_cols'],
                            hidden_layer_sizes=(100,))
        mlp_model.prepare_data()
        mlp_model.setup()
        mlp_model.train()
        weights = [mlp_model.model.coefs_[0],mlp_model.model.intercepts_[0],
                   mlp_model.model.coefs_[1],mlp_model.model.intercepts_[1]]
    else:
        weights = []
    
    # initialize model
    model = convnet_sc_regressor(dim=config.data['dim'],length=1,
                                 len_features=len(config.data['feature_cols']),
                                 weights=weights,dropoutRatio=config.model['dropout_ratio'])
    classifier = LitConvNetRegressor(model,config.training['lr'],config.training['wd'],epochs=config.training['epochs'])

    # load checkpoint
    if wandb.run.resumed:
        classifier = load_model(run, config.meta['user']+'/'+config.meta['project']+'/model-'+config.meta['id']+':latest',
                                model, litclass=LitConvNetRegressor)
    elif config.model['load_checkpoint']:
        classifier = load_model(run, config.model['checkpoint_location'], model, 
                                litclass=LitConvNetRegressor, strict=False)


    # initialize wandb logger
    wandb_logger = WandbLogger(log_model='all')
    checkpoint_callback = ModelCheckpoint(monitor='val_loss',
                                          mode='min',
                                          save_top_k=1,
                                          save_last=True,
                                          save_weights_only=True,
                                          verbose=False)
    early_stop_callback = EarlyStopping(monitor='val_loss',min_delta=0.0002,patience=10,mode='min',strict=False,check_finite=False)

    # train model
    trainer = pl.Trainer(accelerator=config.training['device'],
                         devices=[3],
                         deterministic=False,
                         max_epochs=config.training['epochs'],
                         callbacks=[ModelSummary(max_depth=2),early_stop_callback,checkpoint_callback],
                         logger=wandb_logger,
                         precision=16)
    trainer.fit(model=classifier,datamodule=data)

    # test trained model
    if config.testing['eval']:
        # load best checkpoint
        classifier = load_model(run, config.meta['user']+'/'+config.meta['project']+'/model-'+run.id+':best_k', model,
                                litclass=LitConvNetRegressor)

        # save predictions locally
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
