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
        # 1.3 calibration factor for MDI data
        if self.dataset_frame.iloc[idx] == 'mdi':
            img = img/1.3
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
            forecast_window (int):  number of hours for forecast
            dim (int):              dimension for scaling data
            batch (int):            batch size for all dataloaders
    """
    def __init__(self, data_file: str, label: str, split_type: str = 'random', forecast_window: int = 24, dim: int = 256, batch: int = 32):
        super().__init__()
        self.data_file = data_file
        self.label = label
        self.flux_thresh = 4e7
        self.flare_thresh = 1e-5    # M flare
        self.forecast_window = forecast_window
        self.split_type = split_type
        self.batch_size = batch
        # define data transforms
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(dim,transforms.InterpolationMode.BILINEAR,antialias=True),
        ])

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
            data_full = MagnetogramDataSet(df_full,self.label,self.transform)
            self.train_set, self.val_set = torch.utils.data.random_split(data_full,[0.7,0.3])
        elif self.split_type == 'temporal':
            inds_train = df_full['sample_time'].dt.month < 9
            df_train = df_full.loc[inds_train,:]
            df_val = df_full.loc[~inds_train,:]
            self.train_set = MagnetogramDataSet(df_train,self.label,self.transform)
            self.val_set = MagnetogramDataSet(df_val,self.label,self.transform)
        
        self.test_set = MagnetogramDataSet(df_test,self.label,self.transform)
        print('Train:',len(self.train_set),
              'Valid:',len(self.val_set),
              'Test:',len(self.test_set))
        print('Flare/no-flare ratio in training+val:',sum(df_full['flare']==1),sum(df_full['flare']==0))

    def train_dataloader(self):
        return DataLoader(self.train_set,batch_size=self.batch_size,num_workers=4,shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_set,batch_size=self.batch_size,num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_set,batch_size=self.batch_size,num_workers=4)

    def predict_dataloader(self):
        return DataLoader(self.test_set,batch_size=self.batch_size,num_workers=4)



