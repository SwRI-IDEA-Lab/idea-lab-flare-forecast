import sys,os
sys.path.append(os.getcwd())

import unittest
from src.losses import WeightedMSELoss
import pandas as pd
import numpy as np
import torch
import pytorch_lightning as pl
from pathlib import Path
from torchvision import transforms
import matplotlib.pyplot as plt

class LossTest(unittest.TestCase):
    def setUp(self):
        self.alpha = [0.5,0.8,0.97]
        self.loss = WeightedMSELoss()
        self.input = torch.normal(0,1,(10,3))   # batch size x number of classes
        self.output = torch.randint(0,2,(10,3))

    def test_LossDim(self):
        loss = self.loss(self.input,self.output)
        self.assertIsInstance(loss,torch.Tensor)
        self.assertEqual(loss.shape,torch.Size([]))
        self.assertGreater(loss,0)
    
    def test_LossZero(self):
        loss = self.loss(self.input,self.input+1e-5)
        self.assertAlmostEqual(loss,0,delta=1e-4)
    
    def test_LossWeighted(self):
        loss1 = self.loss(torch.Tensor([0.5]),torch.Tensor([0.6]))
        loss2 = self.loss(torch.Tensor([0.5]),torch.tensor([0.4]))
        self.assertGreater(loss1,loss2)
        self.assertAlmostEqual(loss1,(0.5-0.6)**2*0.6,delta=1e-1)

if __name__ == "__main__":
    unittest.main()