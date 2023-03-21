import sys,os
sys.path.append(os.getcwd())

from astropy.io import fits
import unittest
from src.data_preprocessing.index_clean_magnetograms import *
from src.data_preprocessing.helper import *
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import csv

# Remove Warnings
import warnings
warnings.filterwarnings('ignore')

class IndexingTest(unittest.TestCase):

    def setUp(self):
        self.single_data = ['mdi']
        self.default_data = None
        self.multiple_data = ['SPMG','HMI']
        self.invalid_data = ['blah']
        self.root_dir = Path('Data')
        self.data = 'SPMG'
        self.year = '1992'
        self.fitsfile = os.listdir(self.root_dir/self.data/self.year)[0]
        self.new_dir = Path('Data/hdf5')/self.data
        self.filename='src/tests/index_SPMG.csv'
        self.csv_file = open(self.filename,'w')
        self.csv_writer = csv.writer(self.csv_file,delimiter=',')
        self.header = ['filename','fits_file','timestamp','t_obs','tot_us_flux','tot_flux','datamin','datamax']
        self.header = [key+'_'+self.data for key in self.header]
        self.header.insert(2,'date')
        self.csv_writer.writerow(self.header)
        self.df_HMI = pd.DataFrame({'filename_HMI':['hmi.h5'],
                                    'fits_file_HMI':['hmi.fits'],
                                    'date':[20110102],
                                    'timestamp_HMI':[datetime(2011,2,1,0)]})
        self.fileHMI = self.df_HMI.to_csv('src/tests/index_HMI.csv',index=False)
        self.cols = ['t_obs']
        self.indexer = Indexer(self.data,data_dir='Data',save_dir='Data/hdf5',index_dir='src/tests',metadata_cols=self.cols)

    def test_indexerExists(self):
        self.assertIsNotNone(self.indexer)
        self.assertIsInstance(self.indexer.data,str)
        self.assertIsInstance(self.indexer.root_dir,Path)        
        self.assertIsInstance(self.indexer.new_dir,Path)
        self.assertIsInstance(self.indexer.file,str)
        self.assertIsInstance(self.indexer.metadata_cols,list)
        index = pd.read_csv(self.indexer.file)
        self.assertEqual(len(index.columns),8+len(self.cols))

    def test_indexData(self):
        self.indexer.index_data()
        self.assertIsInstance(self.indexer.error_files,list)
        index = pd.read_csv(self.indexer.file)
        self.assertGreaterEqual(len(index),0)

    def test_parser(self):
        parser = parse_args(self.single_data)
        self.assertIsInstance(parser.data,list)
        self.assertEqual(len(parser.data),1)
        parser = parse_args(self.default_data)
        self.assertIsInstance(parser.data,list)
        self.assertIsInstance(parser.root,str)
        self.assertIsInstance(parser.newdir,str)
        self.assertEqual(len(parser.data),1)
        parser = parse_args(self.multiple_data)
        self.assertIsInstance(parser.data,list)
        self.assertEqual(len(parser.data),len(self.multiple_data))

    def test_indexItem(self):
        file = self.root_dir/self.data/self.year/self.fitsfile
        date,timestamp = extract_date_time(self.fitsfile,self.data,self.year)
        with fits.open(file,cache=False) as data_fits:
            data_fits.verify('fix')
            img,header = extract_fits(data_fits,self.data)
        index_data = self.indexer.index_item(file,img,header,date,timestamp,self.indexer.new_dir/self.year)    
        self.assertIsInstance(index_data,list)
        self.assertEqual(len(index_data),8+len(self.cols))
        print(index_data)

    def test_mergeIndices(self):
        index, errors = self.indexer.index_year(self.year,test=True)
        df = pd.DataFrame(index,columns=self.header)
        df.to_csv(self.filename,index=False)
        df_MDI = pd.read_csv(self.filename)
        self.assertEqual(len(index),len(df_MDI))
        df_merged = merge_indices_by_date('src/tests',self.multiple_data)
        self.assertIsInstance(df_merged,pd.DataFrame)
        self.assertTrue('date' in df_merged.columns)
        for data in self.multiple_data:
            cols = ['filename_'+data,'timestamp_'+data]
            self.assertTrue(all(col in df_merged.columns for col in cols))
        self.assertGreaterEqual(len(df_merged),len(self.df_HMI))
        self.assertGreaterEqual(len(df_merged),len(df_MDI))

    def tearDown(self):
        self.csv_file.close()

if __name__ == "__main__":
    unittest.main()