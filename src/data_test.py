import unittest
from data import MagnetogramDataSet
import pandas as pd
import numpy as np

class DataTest(unittest.TestCase):

    def setUp(self):
        self.df = pd.read_csv('../Data/labels_mdi_small.csv')
        self.df['flare'] = self.df['flare_in_24h']
        self.dataset = MagnetogramDataSet(self.df)
        self.labels = [0,1]
        self.dim = 256
        self.idx = 0        # index into dataset (must be within length of self.dataset)

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
        self.assertIsInstance(item[1],np.ndarray)
        self.assertTrue(item[1].dtype==np.dtype('float32'))
        self.assertTrue(np.shape(item[1])==(self.dim,self.dim))
        self.assertTrue(item[2] in self.labels)
    
    def test_dataNormalization(self):
        for idx in range(len(self.dataset)):
            item = self.dataset[idx]
            data = item[1]
            self.assertLessEqual(np.max(np.abs(data)),1)


if __name__ == "__main__":
    unittest.main()
