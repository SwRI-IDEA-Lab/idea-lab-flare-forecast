import sys,os
sys.path.append(os.getcwd())

import unittest
from src.losses import FocalLoss
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
        self.loss = FocalLoss(gamma=2,alpha=self.alpha)
        self.input = torch.normal(0,1,(10,3))   # batch size x number of classes
        self.output = torch.randint(0,2,(10,3))

    def test_LossDim(self):
        loss = self.loss(self.input,self.output)
        self.assertIsInstance(loss,torch.Tensor)
        self.assertGreater(loss,0)
    
    def test_LossZero(self):
        loss = self.loss(10*(self.output.to(torch.float32)-0.5),self.output)
        self.assertAlmostEqual(loss,0,delta=1e-4)

if __name__ == "__main__":
    unittest.main()
