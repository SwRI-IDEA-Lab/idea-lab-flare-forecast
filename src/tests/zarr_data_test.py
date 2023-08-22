import sys,os
sys.path.append(os.getcwd())

import unittest
from src.data_zarr import ZarrDataSet,AIAHMIDataModule
import pandas as pd
import numpy as np
import torch
import pytorch_lightning as pl
from pathlib import Path
from torchvision import transforms
import matplotlib.pyplot as plt


class ZarrDataTest(unittest.TestCase):

    def setUp(self):
        self.datafile = 'Data/labels_regression_aiahmi.csv'
        self.zarrfile = '/d0/euv/aia/preprocessed/aia_hmi_stacks_2010_2023_1d_full.zarr/'
        self.df = pd.read_csv(self.datafile)
        self.label = 'flare'
        self.df['flare'] = (self.df['xrsb_max_in_24h']>=1e-5).astype(int)
        self.labels = [0,1]
        self.dim = 256
        self.channels=[0,3,7]
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(self.dim,transforms.InterpolationMode.BILINEAR,antialias=True),
        ])
        self.hmidataset = ZarrDataSet(self.df,self.zarrfile,
                                      label=self.label,transform=self.transform,
                                      channels=7,maxvals=300)
        self.aiahmidataset = ZarrDataSet(self.df,self.zarrfile,label=self.label,
                                         transform=self.transform,channels=self.channels,maxvals=[500,1000,300])
        self.idx = 0        # index into dataset (must be within length of self.dataset)
        self.datamodule = AIAHMIDataModule(self.zarrfile,self.datafile,regression=True,
                                           val_split=0,dim=self.dim,channels=self.channels,maxvals=[500,1000,300])

    def test_datasetExists(self):
        self.assertGreaterEqual(len(self.hmidataset),0)
    
    def test_getItem(self):
        self.assertIsNotNone(self.hmidataset[0])

    def test_itemType(self):
        item = self.dataset[self.idx]
        self.assertIsInstance(item,list)        
        self.assertEqual(len(item),4)

    def test_itemContents(self):
        item = self.hmidataset[self.idx]
        self.assertIsInstance(item[0],str)
        self.assertIsInstance(item[1],torch.Tensor)
        self.assertTrue(item[1].dtype==torch.float32)
        self.assertTrue(item[1].shape==(1,self.dim,self.dim))
        self.assertIsInstance(item[2],torch.Tensor)
        self.assertTrue(item[3] in self.labels)
        item2 = self.aiahmidataset[self.idx]
        self.assertTrue(item[2].shape==(len(self.channels),self.dim,self.dim))


    # def test_dataNormalization(self):
    #     for idx in range(np.min([len(self.dataset),10])):
    #         item = self.dataset[idx]
    #         data = item[1]
    #         self.assertLessEqual(torch.max(torch.abs(data)),1)
    #         plt.figure()
    #         plt.imshow(data[0,:,:],vmin=-1,vmax=1)
    #         plt.colorbar()
    #         plt.title(item[0].split('/')[-1]+' flare label: '+str(item[2]))
    #         plt.savefig('src/tests/test_img_'+str(idx)+'.png')
    #         plt.close()

    def test_datamoduleLoadData(self):
        self.datamodule.prepare_data()
        self.assertIsInstance(self.datamodule.df,pd.DataFrame)

    def test_datamoduleSetup(self):
        self.datamodule.prepare_data()
        self.datamodule.setup('fit')
        self.assertIsInstance(self.datamodule.train_dataloader(),torch.utils.data.DataLoader)
        self.assertIsInstance(self.datamodule.val_dataloader(),torch.utils.data.DataLoader)
        self.assertIsInstance(self.datamodule.test_dataloader(),torch.utils.data.DataLoader)
        self.assertIsInstance(self.datamodule.train_set,torch.utils.data.Dataset)
        self.assertIsInstance(self.datamodule.val_set,torch.utils.data.Dataset)
        self.assertIsInstance(self.datamodule.test_set,torch.utils.data.Dataset)
# self.assertEqual(sum_label,len(self.datamodule.train_set)/2)


if __name__ == "__main__":
    unittest.main()


