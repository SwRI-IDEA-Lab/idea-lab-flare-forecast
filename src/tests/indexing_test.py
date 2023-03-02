import sys,os
sys.path.append(os.getcwd())

import unittest
from src.data_preprocessing.index_clean_magnetograms import *
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class IndexingTest(unittest.TestCase):

    def setUp(self):
        self.single_data = ['mdi']
        self.default_data = None
        self.multiple_data = ['MDI','HMI']
        self.invalid_data = ['blah']
        self.root_dir = Path('Data')
        self.data = 'MDI'
        self.year = '1999'
        self.filename='src/tests/index_MDI.csv'
        self.csv_file = open(self.filename,'w')
        self.csv_writer = csv.writer(self.csv_file,delimiter=',')
        self.csv_writer.writerow(['fname_MDI','date','time_MDI','timestamp_MDI'])
        self.df_HMI = pd.DataFrame({'fname_HMI':['hmi.fits'],
                                    'date':[19990204],
                                    'time_HMI':[000000],
                                    'timestamp_HMI':[datetime(1999,2,1,0)]})
        self.fileHMI = self.df_HMI.to_csv('src/tests/index_HMI.csv',index=False)

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
        print(self.filename)
        print(df.head(5))
        self.assertEqual(n,len(df))

    def test_mergeIndices(self):
        n = index_year(self.root_dir,self.data,self.year,self.csv_writer)
        self.csv_file.close()
        df_merged = merge_indices_by_date(Path('src/tests'),self.multiple_data)
        self.assertIsInstance(df_merged,pd.DataFrame)
        self.assertTrue('date' in df_merged.columns)
        for data in self.multiple_data:
            cols = ['fname_'+data,'time_'+data,'timestamp_'+data]
            self.assertTrue(all(col in df_merged.columns for col in cols))
        self.assertGreaterEqual(len(df_merged),len(self.df_HMI))
        df_MDI = pd.read_csv(self.filename)
        self.assertGreaterEqual(len(df_merged),len(df_MDI))

    def tearDown(self):
        self.csv_file.close()

if __name__ == "__main__":
    unittest.main()