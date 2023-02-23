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

    def test_datasetExists(self):
        self.assertGreaterEqual(len(self.dataset),0)
    
    def test_getItem(self):
        self.assertIsNotNone(self.dataset[0])

    def test_itemType(self):
        item = self.dataset[0]
        self.assertIsInstance(item,list)        
        self.assertEqual(len(item),3)
        self.assertIsInstance(item[0],str)
        self.assertIsInstance(item[1],np.ndarray)
        self.assertTrue(item[1].dtype==np.dtype('float32'))
        self.assertTrue(not np.isnan(item[1]).any())
        self.assertTrue(np.shape(item[1])==(self.dim,self.dim))
        self.assertTrue(item[2] in self.labels)


if __name__ == "__main__":
    unittest.main()
