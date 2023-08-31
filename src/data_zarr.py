import sys,os
sys.path.append(os.getcwd())

import torch
from torchvision import transforms
import xarray as xr
import h5py
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import random
from torch.utils.data import Dataset,DataLoader
from sklearn.preprocessing import MaxAbsScaler,StandardScaler
from src.utils.transforms import RandomPolaritySwitch
from datetime import datetime,timedelta
from src.data import split_data

class ZarrDataSet(Dataset):
    """
        Pytorch dataset for handling multichannel AIA and HMI data contained in a zarr array 
        
        Parameters:
            df (dataframe):         Pandas dataframe containing filenames and labels 
            zarr_file (str):        Path to zarr data
            indices (list):     	List of indices within zarr file to get items from
            label (str):            Name of column with label data
            zarr_group (str):       Subhierarchy label within zarr dataset
            transform:              torchvision transforms to apply to data (default is ToTensor())
            feature_cols (list):    names of scalar feature columns
	        channels (list or int):	indices of channels to extract
            maxvals (list or tuple):        scaling values for data, must be same length as channels

    """
    def __init__(self,df,zarr_file:str,indices:list,label:str='flare',
                 zarr_group:str='aia_hmi',transform=transforms.ToTensor(),
                 feature_cols:list=None,channels=7,maxvals=(300,)):
        self.data = xr.open_zarr(zarr_file)
        self.zarr_group = zarr_group
        self.indices = indices
        self.label_frame = df.loc[:,label]
        self.dataset_frame = df.loc[:,'dataset']
        self.transform = transform
        if feature_cols == None:
            feature_cols = []
        self.features = df.loc[:,feature_cols]
        if isinstance(channels,int):
            channels = [channels]
        self.channels = self.data.channel[channels].data
        self.maxvals = maxvals
        
    def __len__(self):
        return len(self.indices)

    def __getitem__(self,idx):
        """
            Get item function for dataset 

            Parameters:
                idx (int):      Index into dataset
            
            Returns:
                List:           index, image array, and label
        """
        zarr_idx = self.data.t_obs[self.indices[idx]].data
        img = self.data[self.zarr_group].loc[zarr_idx,self.channels,:,:].load()
        if img.ndim == 2:
            img = np.expand_dims(img,axis=0)

        img = np.nan_to_num(img)
        # max scaling, TODO: change this to a torch transform
        for i in range(np.shape(img)[0]):
            if self.channels[i] != 'hmilos':
                img[i,img[i,:,:]<1] = 1
                img[i,img[i,:,:]>self.maxvals[i]] = self.maxvals[i]
                img[i,:,:] = np.log10(img[i,:,:])/self.maxvals[i]
            else:
                img[i,img[i,:,:]<-self.maxvals[i]] = -self.maxvals[i]
                img[i,img[i,:,:]>self.maxvals[i]] = self.maxvals[i]
                img[i,:,:] = (img[i,:,:]+self.maxvals[i])/2/self.maxvals[i]
            
        label = self.label_frame.iloc[idx]
        features = torch.Tensor(self.features.iloc[idx])

        # transform image
        img = np.transpose(img,(1,2,0))
        img = self.transform(img)

        return [self.indices[idx],img,features,label]

class AIAHMIDataModule(pl.LightningDataModule):
    """
        Collection of dataloaders for training, validation, test and 
        prediction for multichannel AIA and HMI data. Contains data processing 
        and data splitting.
    
        Parameters:
            zarr_file (str):        path to zarr dataset
            data_file (str):        path to labels data
            regression (bool):      regression task if True else classification
            forecast_window (int):  number of hours for forecast
            dim (int):              dimension for scaling data
            batch (int):            batch size for all dataloaders
            augmentation (str):     option to choose between None, conservative, or full data augmentation
            flare_thresh (float):   threshold for peak flare intensity to label as positive (default 1e-5, M flare)
            feature_cols (list):    list of columns names for scalar features
            test (str):             which test set to choose (test_a or test_b else both)
            channels (list or int):       channels in zarr dataset
            maxvals (tuple):        min-max scaling parameters
    """
    def __init__(self, zarr_file:str, data_file:str, regression:bool=False, 
                 val_split:int=1, forecast_window: int = 24, dim: int = 256, batch: int = 32, 
                 augmentation: str = None, flare_thresh: float = 1e-5, feature_cols:list =None, 
                 test: str = '', file_col:str='filename', channels=7,maxvals=(300,)):
        super().__init__()
        self.zarr_file = zarr_file
        self.data_file = data_file
        self.flare_thresh = flare_thresh
        self.forecast_window = forecast_window
        self.regression = regression
        self.val_split = val_split
        self.batch_size = batch
        if feature_cols == None:
            feature_cols = []
        self.feature_cols = feature_cols
        self.test = test
        self.file_col = file_col
        self.channels = channels
        self.maxvals = maxvals
        self.label='flare'

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
                # RandomPolaritySwitch(p=0.5),
            ])
        elif augmentation == 'full':
            self.training_transform =  transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(dim,transforms.InterpolationMode.BILINEAR,antialias=True),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(20),
                # RandomPolaritySwitch(p=0.5)
            ])
        else:
            self.training_transform = self.transform

    def prepare_data(self):
        # load dataframe
        self.df = pd.read_csv(self.data_file)
        self.df['sample_time'] = pd.to_datetime(self.df['sample_time'],format='mixed')
        # define flare label based on desired forecast window and regression or classification task
        if self.regression:
            self.df['flare'] = self.df['xrsb_max_in_'+str(self.forecast_window)+'h']
            self.df['flare'] = (np.log10(self.df['flare'])+8.5)/6
            self.p_thresh = (np.log10(self.flare_thresh)+8.5)/6
            self.df.loc[self.df['flare']<0,'flare'] = 0           
        else:
            self.df['flare'] = (self.df['xrsb_max_in_'+str(self.forecast_window)+'h']>=self.flare_thresh).astype(int)
            self.p_thresh = 0.5
        self.df = self.df.dropna(axis=0,subset='flare')

    def setup(self,stage: str):
        # performs data splitting and initializes datasets
        df_test,df_pseudotest,df_train,df_val = split_data(self.df,self.val_split,self.test)
        inds_test = np.where(self.df['sample_time'].isin(df_test['sample_time']))[0]
        inds_pseudotest = np.where(self.df['sample_time'].isin(df_pseudotest['sample_time']))[0]
        inds_train = np.where(self.df['sample_time'].isin(df_train['sample_time']))[0]
        inds_val = np.where(self.df['sample_time'].isin(df_val['sample_time']))[0]

        # scale input features
        if len(self.feature_cols)>0:
            self.scaler = StandardScaler()
            self.scaler.fit(df_train.loc[:,self.feature_cols])
            df_train.loc[:,self.feature_cols] = self.scaler.transform(df_train.loc[:,self.feature_cols])
            df_val.loc[:,self.feature_cols] = self.scaler.transform(df_val.loc[:,self.feature_cols])
            df_pseudotest.loc[:,self.feature_cols] = self.scaler.transform(df_pseudotest.loc[:,self.feature_cols])
            df_test.loc[:,self.feature_cols] = self.scaler.transform(df_test.loc[:,self.feature_cols])

        self.train_set = ZarrDataSet(df_train,self.zarr_file,inds_train,
                                     transform=self.training_transform,
                                     feature_cols=self.feature_cols,
                                     channels=self.channels,
                                     maxvals=self.maxvals)
        self.val_set = ZarrDataSet(df_val,self.zarr_file,inds_val,
                                     transform=self.transform,
                                     feature_cols=self.feature_cols,
                                     channels=self.channels,
                                     maxvals=self.maxvals)
        self.trainval_set = ZarrDataSet(pd.concat([df_train,df_val]),self.zarr_file,
                                     np.append(inds_train,inds_val),
                                     transform=self.transform,
                                     feature_cols=self.feature_cols,
                                     channels=self.channels,
                                     maxvals=self.maxvals)
        self.pseudotest_set = ZarrDataSet(df_pseudotest,self.zarr_file,inds_pseudotest,
                                     transform=self.transform,
                                     feature_cols=self.feature_cols,
                                     channels=self.channels, maxvals=self.maxvals)
        self.test_set = ZarrDataSet(df_test,self.zarr_file,inds_test, 
                                     transform=self.transform,
                                     feature_cols=self.feature_cols, 
                                     channels=self.channels, maxvals=self.maxvals)
        print('Train:',len(self.train_set),
              'Valid:',len(self.val_set),
              'Pseudo-test:',len(self.pseudotest_set),
              'Test:',len(self.test_set))
        self.train_p = sum(df_train[self.label]>self.p_thresh)
        self.train_n = sum(df_train[self.label]<=self.p_thresh)
        print('P/N ratio in training:',sum(df_train[self.label]>self.p_thresh),sum(df_train[self.label]<=self.p_thresh))
        print('P/N ratio in validation:',sum(df_val[self.label]>self.p_thresh),sum(df_val[self.label]<=self.p_thresh))
        print('P/N ratio in pseudotest:',sum(df_pseudotest[self.label]>self.p_thresh),sum(df_pseudotest[self.label]<=self.p_thresh))
        print('P/N ratio in test:',sum(df_test[self.label]>self.p_thresh),sum(df_test[self.label]<=self.p_thresh))

    def train_dataloader(self):
        return DataLoader(self.train_set,batch_size=self.batch_size,num_workers=4,drop_last=True,shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_set,batch_size=self.batch_size,num_workers=4,drop_last=True)

    def trainval_dataloader(self,shuffle=False):
        return DataLoader(self.trainval_set,batch_size=self.batch_size,num_workers=4,shuffle=shuffle,drop_last=True)

    def pseudotest_dataloader(self):
        return DataLoader(self.pseudotest_set,batch_size=self.batch_size,num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_set,batch_size=self.batch_size,num_workers=4)

    def predict_dataloader(self):
        return DataLoader(self.test_set,batch_size=self.batch_size,num_workers=4)



