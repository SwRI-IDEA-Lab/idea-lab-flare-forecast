import sys,os
sys.path.append(os.getcwd())
import unittest
from src.model_classifier import convnet_sc, LitConvNet
from src.data import MagnetogramDataSet
import numpy as np
import torch
import h5py
import pandas as pd
import glob

class LightningModuleTest(unittest.TestCase):
    def setUp(self):
        self.files = glob.glob('Data/MDI_small/*')
        filename = self.files[0]
        self.x = torch.tensor(np.array(h5py.File(filename, 'r')['magnetogram']).astype(np.float32))[None,None,:,:]
        self.x[torch.isnan(self.x)]=0
        self.dim = self.x.shape[-1]
        self.model = convnet_sc(dim=self.dim)
        self.litmodel = LitConvNet(self.model)
        self.df = pd.read_csv('Data/labels_mdi_small.csv')
        self.df['flare'] = self.df['flare_in_24h']
        self.dataset = MagnetogramDataSet(self.df)
        self.train_loader = torch.utils.data.DataLoader(self.dataset,batch_size=10)

    def test_litmodelExists(self):
        self.assertIsNotNone(self.litmodel)

    def test_trainModel(self):
        batch = next(iter(self.train_loader))
        batch_idx = 0
        self.assertTrue(torch.is_tensor(self.litmodel.training_step(batch,batch_idx)))

    def test_litmodelOptimizer(self):
        self.assertIsInstance(self.litmodel.configure_optimizers(),torch.optim.Optimizer)

    def test_predictModel(self):
        batch = next(iter(self.train_loader))
        batch_idx = 0
        pred = self.litmodel.predict_step(batch,batch_idx)
        self.assertEqual(len(pred),3)
        self.assertTrue(torch.is_tensor(pred[1]))
        self.assertTrue(torch.is_tensor(pred[2]))

if __name__ == "__main__":
    unittest.main()

