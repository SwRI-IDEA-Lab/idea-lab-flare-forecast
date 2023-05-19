import sys,os
sys.path.append(os.getcwd())

import unittest
from src.linear_model import LinearModel
import pandas as pd
import numpy as np
import torch
import pytorch_lightning as pl
from pathlib import Path
from torchvision import transforms
import matplotlib.pyplot as plt




class LinearModelTest(unittest.TestCase):

    def setUp(self):
        self.datafile = 'Data/labels_all_smoothed2.csv'
        self.window=24
        self.model = LinearModel(self.datafile,self.window)

    def test_modelInit(self):
        self.assertIsInstance(self.model.features,list)
        self.assertIsInstance(self.model.val_split,int)

    def test_prepareData(self):
        self.model.prepare_data()
        self.assertIsInstance(self.model.df,pd.DataFrame)
        self.assertIsInstance(self.model.df['sample_time'],pd.Series)

    def test_setupData(self):
        self.model.prepare_data()
        self.model.setup()
        self.assertIsInstance(self.model.df_train,pd.DataFrame)
        self.assertLessEqual(abs(self.model.X_train).max().max(),1)
        self.assertLessEqual(abs(self.model.X_test).max().max(),1)

    def test_train(self):
        self.model.prepare_data()
        self.model.setup()
        self.model.train()
        self.assertIsNotNone(self.model.model.coef_)

    def test_test(self):
        self.model.prepare_data()
        self.model.setup()
        self.model.train()
        ypred = self.model.test(self.model.X_val,self.model.df_val['flare'])
        self.assertIsInstance(ypred,np.ndarray)
        self.assertLessEqual(np.max(ypred),1)

if __name__ == "__main__":
    unittest.main()
