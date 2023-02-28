import sys,os
sys.path.append(os.getcwd())
import unittest
from src.model import convnet_sc
import numpy as np
import torch
import h5py
import glob
from pathlib import Path


class ModelTest(unittest.TestCase):

    def setUp(self):
        self.files = glob.glob('Data/MDI_small/*')
        filename = self.files[0]
        self.x = torch.tensor(np.array(h5py.File(filename, 'r')['magnetogram']).astype(np.float32))[None,None,:,:]
        self.x[torch.isnan(self.x)]=0
        self.dim = self.x.shape[-1]
        self.model = convnet_sc(dim=self.dim)

    def test_modelexists(self):
        self.assertIsNotNone(self.model)
    
    def test_data(self):
        self.assertTrue(len(self.files)>0)
        print(self.x.shape)
        self.assertTrue(self.x.shape==(1,1,self.dim,self.dim))

    def test_imgNaN(self):
        self.assertTrue(not torch.isnan(self.x).any())

    def test_modelforward(self):
        output = self.model.forward(self.x)
        print(output)
        self.assertTrue(output.shape[0] == 1)
        self.assertTrue(output >= 0 and output <= 1)

    def test_weightint(self):
        for name, layer in self.model.named_modules():
            if isinstance(layer, torch.nn.Linear):
                print(name, layer)        
                print(layer.weight)
                print(layer.bias)


if __name__ == "__main__":
    unittest.main()

