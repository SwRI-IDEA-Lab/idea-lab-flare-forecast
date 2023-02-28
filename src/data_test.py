import unittest
from data import MagnetogramDataSet, generateTrainValidData
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

class DataTest(unittest.TestCase):

    def setUp(self):
        self.df = pd.read_csv('../Data/labels_mdi_small.csv')
        self.df['flare'] = self.df['flare_in_24h']
        self.labels = [0,1]
        self.dim = 256
        self.dataset = MagnetogramDataSet(self.df,self.dim)
        self.idx = 0        # index into dataset (must be within length of self.dataset)
        self.trainsplit = 0.7

    def test_datasetExists(self):
        self.assertGreaterEqual(len(self.dataset),0)
    
    def test_getItem(self):
        self.assertIsNotNone(self.dataset[0])

    def test_itemType(self):
        item = self.dataset[self.idx]
        self.assertIsInstance(item,list)        
        self.assertEqual(len(item),3)

    def test_itemContents(self):
        item = self.dataset[self.idx]
        self.assertIsInstance(item[0],str)
        self.assertIsInstance(item[1],torch.Tensor)
        self.assertTrue(item[1].dtype==torch.float32)
        self.assertTrue(item[1].shape==(1,self.dim,self.dim))
        self.assertTrue(item[2] in self.labels)

    def test_dataNormalization(self):
        for idx in range(np.max([len(self.dataset),20])):
            item = self.dataset[idx]
            data = item[1]
            self.assertLessEqual(torch.max(torch.abs(data)),1)
            plt.figure()
            plt.imshow(data[0,:,:],vmin=-1,vmax=1)
            plt.colorbar()
            plt.title(item[0].split('/')[-1]+' flare label: '+str(item[2]))
            plt.savefig('../test_img_'+str(idx)+'.png')


    def test_trainvalidSplitExists(self):
        trainDataSet, validDataSet = generateTrainValidData(self.df,self.trainsplit)
        self.assertIsInstance(trainDataSet,torch.utils.data.Dataset)
        self.assertIsInstance(validDataSet,torch.utils.data.Dataset)


if __name__ == "__main__":
    unittest.main()
