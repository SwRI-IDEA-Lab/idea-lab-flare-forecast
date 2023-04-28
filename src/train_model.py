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

def load_model(run,ckpt_path,model):
    """
    Load model into wandb run by downloading and initializing weights

    Parameters:
        run:        wandb run object
        ckpt_path:  wandb path to download model checkpoint from
        model:      model class
    Returns:
        classifier: LitConvNet object with loaded weights
    """
    print('Loading model checkpoint from ', ckpt_path)
    artifact = run.use_artifact(ckpt_path,type='model')
    artifact_dir = artifact.download()
    classifier = LitConvNet.load_from_checkpoint(Path(artifact_dir)/'model.ckpt',model=model)
    return classifier

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

    # define data module
    data = MagnetogramDataModule(data_file=config.data['data_file'],
                                 label=config.data['label'],
                                 balance_ratio=config.data['balance_ratio'],
                                 split_type=config.data['split_type'],
                                 val_split=config.data['val_split'],
                                 forecast_window=config.data['forecast_window'],
                                 dim=config.data['dim'],
                                 batch=config.training['batch_size'],
                                 augmentation=config.data['augmentation'],
                                 flare_thresh=config.data['flare_thresh'],
                                 flux_thresh=config.data['flux_thresh'])

    # define model
    model = convnet_sc(dim=config.data['dim'],length=1,dropoutRatio=config.model['dropout_ratio'])
    classifier = LitConvNet(model,config.training['lr'],config.training['wd'],epochs=config.training['epochs'])

    # load checkpoint
    if wandb.run.resumed:
        classifier = load_model(run, 'kierav/'+config.meta['project']+'/model-'+config.meta['id']+':latest',model)
    elif config.model['load_checkpoint']:
        classifier = load_model(run, config.model['checkpoint_location'], model)

    # initialize wandb logger
    wandb_logger = WandbLogger(log_model='all')
    checkpoint_callback = ModelCheckpoint(monitor='val_loss',
                                          mode='min',
                                          save_top_k=1,
                                          save_last=True,
                                          save_weights_only=True,
                                          verbose=False)
    early_stop_callback = EarlyStopping(monitor='val_loss',min_delta=0.0,patience=10,mode='min')

    # train model
    trainer = pl.Trainer(accelerator='cpu',
                         devices=1,
                         deterministic=False,
                         max_epochs=config.training['epochs'],
                        #  log_every_n_steps=4,
                         callbacks=[ModelSummary(max_depth=2),early_stop_callback,checkpoint_callback],
                        #  limit_train_batches=10,
                        #  limit_val_batches=5,
                         logger=wandb_logger)
    trainer.fit(model=classifier,datamodule=data)

    # test trained model
    if config.testing['eval']:
        # load best checkpoint
        classifier = load_model(run, 'kierav/'+config.meta['project']+'/model-'+run.id+':best_k', model)
    
        # run test to log metrics to wandb
        trainer.test(model=classifier,dataloaders=data.pseudotest_dataloader())

        # save predictions locally
        preds = trainer.predict(model=classifier,dataloaders=data.pseudotest_dataloader())

        file = []
        ytrue = []
        ypred = []
        for predbatch in preds:
            file.extend(predbatch[0])
            ytrue.extend(np.array(predbatch[1]).flatten())
            ypred.extend(np.array(predbatch[2]).flatten())
        df = pd.DataFrame({'filename':file,'ytrue':ytrue,'ypred':ypred})
        df.to_csv(wandb.run.dir+'/pseudotest_results.csv',index=False)
        wandb.save('pseudotest_results.csv')

    wandb.finish()

if __name__ == "__main__":
    main()
