import sys,os
sys.path.append(os.getcwd())

import unittest
from src.utils.transforms import RandomPolaritySwitch
from src.data import MagnetogramDataSet, MagnetogramDataModule
import torch
import numpy as np
import h5py

class AugmentationTest(unittest.TestCase):

    def setUp(self):
        img = np.array(h5py.File('Data/hdf5/MDI/2011/MDI_magnetogram.20110101_000000_TAI.h5','r')['magnetogram'])
        self.img = torch.Tensor(img)
    
    def testPolaritySwitch(self):
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

if __name__ == "__main__":
    unittest.main()