import sys,os
sys.path.append(os.getcwd())

import unittest
from src.data_preprocessing.regression_labeller import *
from src.data_preprocessing.helper import *
from sunpy import timeseries as ts
import numpy as np
import pandas as pd
import os
from datetime import datetime,timedelta

class LabelingTest(unittest.TestCase):

    def setUp(self):
        self.goes_dir = '/home/kvandersande/sunpy/data/goes/'
        self.sample_time = datetime(2020,1,1,0)
        self.windows=[24,48]
        self.flare_filename = 'Data/hek_flare_catalog.csv'
        self.flare_catalog = pd.read_csv(self.flare_filename)
        self.index_file = 'Data/index_MDIsmoothed.csv'
        self.out_file = 'src/tests/labels_MDI.csv'
        self.header = ['filename','sample_time','dataset','tot_us_flux','tot_flux','datamin','datamax','xrsb_max_in_24h','xrsb_max_in_48h']
        self.labeler = Labeler(self.index_file,self.out_file,self.flare_filename,
                               goes_dir=self.goes_dir,flare_windows=self.windows)

    def test_retrieveGOESdata(self):
        goes_data = self.labeler.retrieve_goes_data(self.sample_time)
        self.assertIsInstance(goes_data,pd.DataFrame)
        self.assertLessEqual(goes_data.index.max(),self.sample_time+timedelta(hours=max(self.windows)))
        self.assertGreater(goes_data.index.min(),self.sample_time)

    def test_addRegressionLabels(self):
        goes_data = self.labeler.retrieve_goes_data(self.sample_time)
        label = self.labeler.add_regression_data(goes_data,self.sample_time,self.windows[0])
        self.assertIsInstance(label,np.float32)    
        nolabel = self.labeler.add_regression_data([],self.sample_time,self.windows[0])
        self.assertTrue(np.isnan(nolabel))

    def test_labellerExists(self):
        self.assertIsNotNone(self.labeler)
        self.assertIsInstance(self.labeler.samples,pd.DataFrame)
        self.assertIsInstance(self.labeler.flares,pd.DataFrame)
        self.assertIsInstance(self.labeler.flare_windows,list)
        self.assertIsInstance(self.labeler.file,str)
        self.assertEqual(np.sum(self.labeler.samples.duplicated(subset='date')),0)
        start_date = int(datetime.strftime(self.labeler.flares['start_time'][0]-timedelta(hours = max(self.labeler.flare_windows)),'%Y%m%d'))
        self.assertEqual(np.sum(self.labeler.samples['date']<start_date),0)

    def test_parser(self):
        parser = parse_args([self.index_file,self.out_file,'-w 12'])
        self.assertIsInstance(parser.index_file,str)
        self.assertTrue(os.path.exists(parser.index_file))
        self.assertIsInstance(parser.out_file,str)
        self.assertIsInstance(parser.flare_windows,list)
        self.assertIsInstance(parser.flare_windows[0],int)

    def test_writeHeader(self):
        self.labeler.write_header()
        df = pd.read_csv(self.out_file)
        self.assertTrue(all(x in df.columns for x in self.header))

if __name__ == "__main__":
    unittest.main()