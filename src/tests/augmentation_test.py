import sys,os
sys.path.append(os.getcwd())

import unittest
from src.utils.transforms import RandomPolaritySwitch
from src.data import MagnetogramDataSet, MagnetogramDataModule
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import h5py
import pytorch_lightning as pl


class AugmentationTest(unittest.TestCase):

    def setUp(self):
        img = np.array(h5py.File('Data/hdf5/MDI/2011/MDI_magnetogram.20110101_000000_TAI.h5','r')['magnetogram'])
        self.img = torch.Tensor(img)
        self.dim = 128
        self.data_noaugment = MagnetogramDataModule('Data/labels_MDI.csv','flare',dim=self.dim,augmentation=None)
        self.data_simpleaugment = MagnetogramDataModule('Data/labels_MDI.csv','flare',dim=self.dim,augmentation='conservative')
        self.data_fullaugment = MagnetogramDataModule('Data/labels_MDI.csv','flare',dim=self.dim,augmentation='full')
        pl.seed_everything(42)

    def test_polaritySwitch(self):
        negtransform = RandomPolaritySwitch(p=1)
        transformed_img = negtransform(self.img)
        self.assertEqual(transformed_img.shape,self.img.shape)
        self.assertEqual(torch.max(transformed_img+self.img),0)
        self.assertEqual(torch.min(transformed_img+self.img),0)
        notransform = RandomPolaritySwitch(p=0)
        transformed_img = notransform(self.img)
        self.assertEqual(transformed_img.shape,self.img.shape)
        self.assertEqual(torch.max(transformed_img-self.img),0)
        self.assertEqual(torch.min(transformed_img-self.img),0)

    def test_trainingTransform(self):
        self.data_noaugment.prepare_data()
        self.data_noaugment.setup('fit')
        noaug_img = self.data_noaugment.train_set[0][1]
        self.data_simpleaugment.prepare_data()
        self.data_simpleaugment.setup('fit')
        simpleaug_img = self.data_simpleaugment.train_set[0][1]
        self.data_fullaugment.prepare_data()
        self.data_fullaugment.setup('fit')
        fullaug_img = self.data_fullaugment.train_set[0][1]
        self.assertEqual(noaug_img.shape,(1,self.dim,self.dim))
        self.assertEqual(simpleaug_img.shape,(1,self.dim,self.dim))        
        self.assertEqual(fullaug_img.shape,(1,self.dim,self.dim))
        for idx in range(np.min([len(self.data_noaugment.train_set),5])):
            item1 = self.data_noaugment.train_set[idx]
            item2 = self.data_simpleaugment.train_set[idx]
            item3 = self.data_fullaugment.train_set[idx]
            fig,ax = plt.subplots(1,3,figsize=(15,4))
            ax[0].imshow(item1[1][0,:,:],vmin=-1,vmax=1)
            ax[0].set_title(item1[0].split('/')[-1]+' label: '+str(item1[2]))
            ax[1].imshow(item2[1][0,:,:],vmin=-1,vmax=1)
            ax[1].set_title(item2[0].split('/')[-1]+' label: '+str(item2[2]))
            ax[2].imshow(item3[1][0,:,:],vmin=-1,vmax=1)
            ax[2].set_title(item3[0].split('/')[-1]+' label: '+str(item3[2]))
            plt.savefig('src/tests/test_augmentimg_'+str(idx)+'.png')
            plt.close()

if __name__ == "__main__":
    unittest.main()