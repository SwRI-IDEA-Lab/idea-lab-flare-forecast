import torch
import h5py
import numpy as np

class MagnetogramDataSet(torch.utils.data.Dataset):
    """
        Pytorch dataset for handling magnetogram data 
        
        Parameters:
            df (dataframe):     Pandas dataframe containing filenames and labels 
    """
    def __init__(self,df):
        self.name_frame = df.loc[:,'filename']
        self.label_frame = df.loc[:,'flare']

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

        return [filename,img,label]
    