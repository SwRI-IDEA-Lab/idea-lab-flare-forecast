import unittest
from model import convnet_sc, LitConvNet
from data import MagnetogramDataSet
import numpy as np
import torch
import h5py
import pandas as pd

class LightningModuleTest(unittest.TestCase):
    def setUp(self):
        self.model = convnet_sc()
        self.litmodel = LitConvNet(self.model)
        self.df = pd.read_csv('../Data/labels_mdi_small.csv')
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
        self.assertEqual(len(pred),2)
        self.assertTrue(torch.is_tensor(pred[0]))
        self.assertTrue(torch.is_tensor(pred[1]))

if __name__ == "__main__":
    unittest.main()

