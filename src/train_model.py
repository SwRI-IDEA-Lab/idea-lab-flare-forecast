import os
import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelSummary
from model import convnet_sc,LitConvNet
from data import MagnetogramDataSet
import pandas as pd
import numpy as np

def main():
    # set dataset file and parameters
    df = pd.read_csv('../Data/labels_mdi_small.csv')
    window = 12     # forecast window (hours)

    # set seeds
    pl.seed_everything(42,workers=True)

    # define dataloaders
    df['flare'] = df['flare_intensity_in_'+str(window)+'h']>=1e-5
    dataset = MagnetogramDataSet(df)
    train_loader = DataLoader(dataset,batch_size=5,shuffle=True,num_workers=4)
    test_loader = DataLoader(dataset,batch_size=20,shuffle=False)

    # define model
    model = convnet_sc(dim=256,length=1,dropoutRatio=0)
    classifier = LitConvNet(model)

    # train model
    trainer = pl.Trainer(deterministic=True,max_epochs=5,callbacks=[ModelSummary(max_depth=2)])
    trainer.fit(model=classifier,train_dataloaders=train_loader)

    # evaluate model
    predictions = trainer.predict(classifier,test_loader)
    y_pred = predictions[0][0]
    y_true = predictions[0][1]
    print('Predicted: ',torch.transpose(y_pred,0,1))
    print('True:', torch.transpose(y_true,0,1))

if __name__ == "__main__":
    main()
