import torch
from torchvision import transforms
import h5py
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import Dataset,DataLoader
from datetime import datetime,timedelta

class MagnetogramDataSet(Dataset):
    """
        Pytorch dataset for handling magnetogram data 
        
        Parameters:
            df (dataframe):     Pandas dataframe containing filenames and labels 
            dim (int):          Square dimension for resized magnetograms
            transform:          torchvision transform to apply to data (default is ToTensor())
    """
    def __init__(self,df,transform=transforms.ToTensor()):
        self.name_frame = df.loc[:,'filename']
        self.label_frame = df.loc[:,'flare']
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
        # 1.3 calibration factor for MDI data
        if self.dataset_frame.iloc[idx] == 'mdi':
            img = img/1.3
        # clip magnetogram data within max value
        maxval = 2000  # Gauss
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
            forecast_window (int):  number of hours for forecast
    """
    def __init__(self, data_file: str, forecast_window: int = 24):
        super().__init__()
        self.data_file = data_file
        self.forecast_window = forecast_window
        self.split_type = 'random'
        self.batch_size = 32
        # define data transforms
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(256,transforms.InterpolationMode.BILINEAR,antialias=True),
        ])

    def prepare_data(self):
        # load dataframe
        self.df = pd.read_csv(self.data_file)
        self.df['sample_time'] = pd.to_datetime(self.df['sample_time'])
        # define label
        self.df['flare'] = self.df['flare_in_'+str(self.forecast_window)+'h']

    def setup(self,stage: str):
        # performs data splitting and initializes datasets

        # hold out test set
        inds_test = self.df['sample_time']>datetime(2016,1,1)
        df_test = self.df.loc[inds_test,:]
        df_full = self.df.loc[~inds_test,:]
        print('Bounds of test set:',df_test['sample_time'].min(),df_test['sample_time'].max())
        print('Bounds of training set:',df_full['sample_time'].min(),df_full['sample_time'].max())
        # perform random splitting of training and validation set
        if self.split_type == 'random':
            data_full = MagnetogramDataSet(df_full,self.transform)
            self.train_set, self.val_set = torch.utils.data.random_split(data_full,[0.7,0.3])
        self.test_set = MagnetogramDataSet(df_test,self.transform)
        print('Train:',len(self.train_set),'Valid:',len(self.val_set),'Test:',len(self.test_set))

    def train_dataloader(self):
        return DataLoader(self.train_set,batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_set,batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_set,batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.test_set,batch_size=self.batch_size)



