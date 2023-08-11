import sys,os
sys.path.append(os.getcwd())

import torch
from torchvision import transforms
import h5py
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import random
from torch.utils.data import Dataset,DataLoader
from sklearn.preprocessing import MaxAbsScaler,StandardScaler
from src.utils.transforms import RandomPolaritySwitch
from datetime import datetime,timedelta

def split_data(df,val_split,test=''):
    """
        Split dataset into training, validation, hold-out (pseudotest) and test sets.
        The test set can be either 'test_a' which is all data from November and December,
        or 'test_b' which is 2016-2017, or both combined. The hold-out set is data from
        Sept 15-Oct 31. The remaining data is split temporally 4:1 into training and 
        validation.

        Parameters:
            df (dataframe):     Pandas dataframe containing all the data
            val_split (0-4):    Number between 0-4 indicating which temporal training/validation split to select    
            test (str):         Which test set to choose ('test_a' or 'test_b', otherwise both)

        Returns:
            df_test (dataframe):        Test set
            df_pseudotest (dataframe):  Hold-out set
            df_train (dataframe):       Training set
            df_val (dataframe):         Validation set
    """

    # hold out test sets
    inds_test_a = (df['sample_time'].dt.month >= 11)
    inds_test_b = (df['sample_time']>=datetime(2016,1,1))&(df['sample_time']<datetime(2018,1,1)) 
    
    # select test set
    if test == 'test_a':
        inds_test = inds_test_a
    elif test == 'test_b':
        inds_test = inds_test_b
    else:
        inds_test = inds_test_a | inds_test_b

    df_test = df.loc[inds_test,:]
    df_full = df.loc[~inds_test,:]

    # select pseudotest/hold-out set
    if test == 'test_a':
        inds_pseudotest = ((df_full['sample_time'].dt.month==10)&(df_full['sample_time'].dt.day>26)) | ((df_full['sample_time'].dt.month==1)&(df_full['sample_time'].dt.day<6))
    elif test == 'test_b':
        inds_pseudotest = (df['sample_time']>=datetime(2015,12,26))
    else:
        inds_pseudotest = (df_full['sample_time'].dt.month==10) | ((df_full['sample_time'].dt.month==9)&(df_full['sample_time'].dt.day>15))
    # inds_pseudotest = (df['sample_time']<datetime(1996,1,1)) | ((df_full['sample_time'].dt.month==10)&(df_full['sample_time'].dt.day>26)) | ((df_full['sample_time'].dt.month==1)&(df_full['sample_time'].dt.day<6))
    df_pseudotest = df_full.loc[inds_pseudotest,:]

    # split training and validation
    df_train = df_full.loc[~inds_pseudotest,:]
    df_train = df_train.reset_index(drop=True)
    n_val = int(np.floor(len(df_train)/5))
    df_val = df_train.iloc[val_split*n_val:(val_split+1)*n_val,:]
    df_train = df_train.drop(df_val.index)

    return df_test,df_pseudotest,df_train,df_val

class MagnetogramDataSet(Dataset):
    """
        Pytorch dataset for handling magnetogram data 
        
        Parameters:
            df (dataframe):     Pandas dataframe containing filenames and labels 
            label (str):        Name of column with label data
            transform:          torchvision transform to apply to data (default is ToTensor())
    """
    def __init__(self,df,label: str ='flare',transform=transforms.ToTensor(),feature_cols=[]):
        self.name_frame = df.loc[:,'filename']
        self.label_frame = df.loc[:,label]
        self.dataset_frame = df.loc[:,'dataset']
        self.transform = transform
        self.features = df.loc[:,feature_cols]

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
        if self.name != 'filename':
            # embedding is in shape (dim x dim x channels)
            img = np.load('../solar-similarity-search/'+filename).astype(np.float32)
            img = img/self.maxval
        else:
            img = np.array(h5py.File(filename,'r')['magnetogram']).astype(np.float32)
            img = np.nan_to_num(img)

            # Normalize magnetogram data
            # clip magnetogram data within max value
            img[np.where(img>self.maxval)] = self.maxval
            img[np.where(img<-self.maxval)] = -self.maxval
            # scale between -1 and 1
            img = img/self.maxval

        label = self.label_frame.iloc[idx]
        features = torch.Tensor(self.features.iloc[idx])

        # Normalize magnetogram data
        # clip magnetogram data within max value
        maxval = 1000  # Gauss
        img[np.where(img>maxval)] = maxval
        img[np.where(img<-maxval)] = -maxval
        # scale between -1 and 1
        img = img/maxval

        # transform image
        img = self.transform(img)

        return [filename,img,features,label]
    

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
            feature_cols (list):    list of columns names for scalar features
            test (str):             which test set to choose (test_a or test_b else both)
    """
    def __init__(self, data_file:str, label:str, balance_ratio:int=None, split_type:str='random', 
                 val_split:int=1, forecast_window: int = 24, dim: int = 256, batch: int = 32, 
                 augmentation: str = None, flare_thresh: float = 1e-5, flux_thresh: float = 1.5e7,
                 feature_cols=['tot_us_flux'], test: str = ''):
        super().__init__()
        self.data_file = data_file
        self.label = label
        self.flux_thresh = flux_thresh
        self.flare_thresh = flare_thresh
        self.forecast_window = forecast_window
        self.split_type = split_type
        self.val_split = val_split
        self.balance_ratio = balance_ratio
        self.batch_size = batch
        self.feature_cols = feature_cols
        self.test = test

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
        # self.df['flare'] = (self.df['xrsb_max_in_'+str(self.forecast_window)+'h']>=self.flare_thresh).astype(int)
        self.df['flare'] = self.df['xrsb_max_in_'+str(self.forecast_window)+'h']
        self.df.loc[self.df['sample_time']<datetime(2001,3,1),'flare'] =  self.df.loc[self.df['sample_time']<datetime(2001,3,1),'flare']/0.7
        self.df['flare'] = (np.log10(self.df['flare'])+8.5)/6
        self.df.loc[self.df['flare']<0,'flare'] = 0
        self.df = self.df.dropna(axis=0,subset='flare')
        # # define label based on linear relationship between flare intensity and flux
        # self.df['flare_flux'] = ((self.df['flare_intensity_in_'+str(self.forecast_window)+'h']==0) & self.df['tot_us_flux']>=self.flux_thresh) + (np.log10(self.df['flare_intensity_in_'+str(self.forecast_window)+'h'])>(-3 - 3/self.flux_thresh*self.df['tot_us_flux']))
        # self.df['flare_flux'] = self.df['flare_flux'].astype(int)

    def setup(self,stage: str):
        # performs data splitting and initializes datasets
        df_test,df_pseudotest,df_train,df_val = split_data(self.df,self.val_split,self.test)

        # balance training data
        if self.balance_ratio > 0:
            df_train = df_train
            inds_train = np.array(df_train[self.label]==1)
            inds_neg = np.where(df_train[self.label]==0)[0]
            random.shuffle(inds_neg)
            inds_train[inds_neg[:self.balance_ratio*np.sum(inds_train)]] = 1
            df_train = df_train.iloc[inds_train,:]

        # scale input features
        self.scaler = StandardScaler()
        self.scaler.fit(df_train.loc[:,self.feature_cols])
        df_train.loc[:,self.feature_cols] = self.scaler.transform(df_train.loc[:,self.feature_cols])
        df_val.loc[:,self.feature_cols] = self.scaler.transform(df_val.loc[:,self.feature_cols])
        df_pseudotest.loc[:,self.feature_cols] = self.scaler.transform(df_pseudotest.loc[:,self.feature_cols])
        df_test.loc[:,self.feature_cols] = self.scaler.transform(df_test.loc[:,self.feature_cols])

        self.train_set = MagnetogramDataSet(df_train,self.label,self.training_transform,self.feature_cols)
        self.val_set = MagnetogramDataSet(df_val,self.label,self.transform,self.feature_cols)
        self.trainval_set = MagnetogramDataSet(pd.concat([df_train,df_val]),self.label,self.transform,self.feature_cols)
        self.pseudotest_set = MagnetogramDataSet(df_pseudotest,self.label,self.transform,self.feature_cols)
        self.test_set = MagnetogramDataSet(df_test,self.label,self.transform,self.feature_cols)
        print('Train:',len(self.train_set),
              'Valid:',len(self.val_set),
              'Pseudo-test:',len(self.pseudotest_set),
              'Test:',len(self.test_set))
        self.train_p = sum(df_train[self.label]>3.5/6)
        self.train_n = sum(df_train[self.label]<=3.5/6)
        print('P/N ratio in training:',sum(df_train[self.label]>3.5/6),sum(df_train[self.label]<=3.5/6))
        print('P/N ratio in validation:',sum(df_val[self.label]>3.5/6),sum(df_val[self.label]<=3.5/6))
        print('P/N ratio in pseudotest:',sum(df_pseudotest[self.label]>3.5/6),sum(df_pseudotest[self.label]<=3.5/6))
        print('P/N ratio in test:',sum(df_test[self.label]>3.5/6),sum(df_test[self.label]<=3.5/6))

    def train_dataloader(self):
        return DataLoader(self.train_set,batch_size=self.batch_size,num_workers=4,shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_set,batch_size=self.batch_size,num_workers=4)

    def trainval_dataloader(self):
        return DataLoader(self.trainval_set,batch_size=self.batch_size,num_workers=4)

    def pseudotest_dataloader(self):
        return DataLoader(self.pseudotest_set,batch_size=self.batch_size,num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_set,batch_size=self.batch_size,num_workers=4)

    def predict_dataloader(self):
        return DataLoader(self.test_set,batch_size=self.batch_size,num_workers=4)



