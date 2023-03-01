import sys,os
sys.path.append(os.getcwd())

import unittest
from src.data_preprocessing.index_clean_magnetograms import *
import numpy as np
import pandas as pd

class IndexingTest(unittest.TestCase):

    def setUp(self):
        self.single_data = ['mdi']
        self.default_data = None
        self.multiple_data = ['mdi','hmi']
        self.invalid_data = ['blah']
        self.root_dir = Path('Data')
        self.data = 'MDI'
        self.year = '1999'
        self.filename='src/tests/index_MDI_test.csv'
        self.csv_file = open(self.filename,'w')
        self.csv_writer = csv.writer(self.csv_file,delimiter=',')
        self.csv_writer.writerow(['filename','date','time','timestamp'])

    def test_parser(self):
        parser = parse_args(self.single_data)
        self.assertIsInstance(parser.data,list)
        self.assertEqual(len(parser.data),1)
        parser = parse_args(self.default_data)
        self.assertIsInstance(parser.data,list)
        self.assertEqual(len(parser.data),1)
        parser = parse_args(self.multiple_data)
        self.assertIsInstance(parser.data,list)
        self.assertEqual(len(parser.data),len(self.multiple_data))
    
    def test_checkQuality(self):
        self.assertTrue(check_quality(data='MWO',header={}))
        self.assertFalse(check_quality(data='MDI',header={'MISSVALS':5000}))
        self.assertTrue(check_quality(data='MDI',header={'MISSVALS':0}))
        self.assertFalse(check_quality(data='HMI',header={'QUALITY':1}))
        self.assertTrue(check_quality(data='HMI',header={'QUALITY':0}))

    def test_indexYear(self):
        n = index_year(self.root_dir,self.data,self.year,self.csv_writer)
        self.csv_file.close()
        df = pd.read_csv(self.filename)
        print(df.head(5))
        self.assertEqual(n,len(df))

    def tearDown(self):
        self.csv_file.close()

if __name__ == "__main__":
    unittest.main()