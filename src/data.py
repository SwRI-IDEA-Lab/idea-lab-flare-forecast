import torch
from torchvision import transforms
import h5py
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import random
from torch.utils.data import Dataset,DataLoader
from src.utils.transforms import RandomPolaritySwitch
from datetime import datetime,timedelta

class MagnetogramDataSet(Dataset):
    """
        Pytorch dataset for handling magnetogram data 
        
        Parameters:
            df (dataframe):     Pandas dataframe containing filenames and labels 
            label (str):        Name of column with label data
            transform:          torchvision transform to apply to data (default is ToTensor())
    """
    def __init__(self,df,label: str ='flare',transform=transforms.ToTensor()):
        self.name_frame = df.loc[:,'filename']
        self.label_frame = df.loc[:,label]
        self.dataset_frame = df.loc[:,'dataset']
        self.transform = transform

    def __len__(self):
        return len(self.name_frame)

    def __getitem__(self,idx):
        """
            Get item function for dataset 

            Parameters:
                idx (int):      Index into dataset
            
            Returns:
                List:           Filename, magnetogram image array, and label
        """
        filename = self.name_frame.iloc[idx]
        img = np.array(h5py.File(filename,'r')['magnetogram']).astype(np.float32)
        img = np.nan_to_num(img)
        label = self.label_frame.iloc[idx]

        # Normalize magnetogram data
        # clip magnetogram data within max value
        maxval = 1000  # Gauss
        img[np.where(img>maxval)] = maxval
        img[np.where(img<-maxval)] = -maxval
        # scale between -1 and 1
        img = img/maxval

        # transform image
        img = self.transform(img)

        return [filename,img,label]
    

class MagnetogramDataModule(pl.LightningDataModule):
    """
        Collection of dataloaders for training, validation, test
        and prediction for magnetograms. Contains data processing and data splitting.
    
        Parameters:
            data_file (str):        file containing magnetogram filenames and labels
            label (str):            name of column for label data (i.e., 'flare' or 'high_flux')
            balance_ratio (int):    ratio of negatives to positives to impose, None for unbalanced
            split_type (str):       method of splitting training and validation data ('random' or 'temporal')
            forecast_window (int):  number of hours for forecast
            dim (int):              dimension for scaling data
            batch (int):            batch size for all dataloaders
            augmentation (str):     option to choose between None, conservative, or full data augmentation
            flare_thresh (float):   threshold for peak flare intensity to label as positive (default 1e-5, M flare)
            flux_thresh (float):    threshold for total unsigned flux to label as positive (default 4e7)
    """
    def __init__(self, data_file:str, label:str, balance_ratio:int=None, split_type:str='random', forecast_window: int = 24, dim: int = 256, batch: int = 32, augmentation: str = None, flare_thresh: float = 1e-5, flux_thresh: float = 4e7):
        super().__init__()
        self.data_file = data_file
        self.label = label
        self.flux_thresh = flux_thresh
        self.flare_thresh = flare_thresh
        self.forecast_window = forecast_window
        self.split_type = split_type
        self.balance_ratio = balance_ratio
        self.batch_size = batch

        # define data transforms
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(dim,transforms.InterpolationMode.BILINEAR,antialias=True)
        ])

        if augmentation == 'conservative':
            self.training_transform =  transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(dim,transforms.InterpolationMode.BILINEAR,antialias=True),
                transforms.RandomVerticalFlip(p=0.5),
                RandomPolaritySwitch(p=0.5),
            ])
        elif augmentation == 'full':
            self.training_transform =  transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(dim,transforms.InterpolationMode.BILINEAR,antialias=True),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(20),
                RandomPolaritySwitch(p=0.5)
            ])
        else:
            self.training_transform = self.transform

    def prepare_data(self):
        # load dataframe
        self.df = pd.read_csv(self.data_file)
        self.df['sample_time'] = pd.to_datetime(self.df['sample_time'])
        # define high flux based on total unsigned flux threshold (for pretraining)
        self.df['high_flux'] = (self.df['tot_us_flux'] >= self.flux_thresh).astype(int)
        # define flare label based on desired forecast window
        self.df['flare'] = (self.df['flare_intensity_in_'+str(self.forecast_window)+'h']>=self.flare_thresh).astype(int)

    def setup(self,stage: str):
        # performs data splitting and initializes datasets

        # hold out test set
        inds_test = (self.df['sample_time']>datetime(2016,1,1)) | (self.df['sample_time'].dt.month >= 11)
        df_test = self.df.loc[inds_test,:]
        df_full = self.df.loc[~inds_test,:]
        print('Bounds of test set:',df_test['sample_time'].min(),df_test['sample_time'].max())
        print('Bounds of training set:',df_full['sample_time'].min(),df_full['sample_time'].max())
        
        # perform splitting of training and validation set
        if self.split_type == 'random':
            df_train = df_full.sample(frac=0.7,random_state=42)
            df_val = df_full.drop(df_train.index)
        elif self.split_type == 'temporal':
            inds_train = df_full['sample_time'].dt.month < 9
            df_train = df_full.loc[inds_train,:]
            df_val = df_full.loc[~inds_train,:]

        # balance training data
        if self.balance_ratio != None:
            df_train = df_train
            inds_train = np.array(df_train[self.label]==1)
            inds_neg = np.where(df_train[self.label]==0)[0]
            random.shuffle(inds_neg)
            inds_train[inds_neg[:self.balance_ratio*np.sum(inds_train)]] = 1
            df_train = df_train.iloc[inds_train,:]

        self.train_set = MagnetogramDataSet(df_train,self.label,self.training_transform)
        self.val_set = MagnetogramDataSet(df_val,self.label,self.transform)
        self.test_set = MagnetogramDataSet(df_test,self.label,self.transform)
        print('Train:',len(self.train_set),
              'Valid:',len(self.val_set),
              'Test:',len(self.test_set))
        print('Flare/no-flare ratio in training:',sum(df_train['flare']==1),sum(df_train['flare']==0))
        print('Flare/no-flare ratio in validation:',sum(df_val['flare']==1),sum(df_val['flare']==0))
        print('Flare/no-flare ratio in test:',sum(df_test['flare']==1),sum(df_test['flare']==0))

    def train_dataloader(self):
        return DataLoader(self.train_set,batch_size=self.batch_size,num_workers=4,shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_set,batch_size=self.batch_size,num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_set,batch_size=self.batch_size,num_workers=4)

    def predict_dataloader(self):
        return DataLoader(self.test_set,batch_size=self.batch_size,num_workers=4)



