import sys,os
sys.path.append(os.getcwd())

import unittest
from src.data_preprocessing.label_dataset import *
from src.data_preprocessing.helper import *
import numpy as np
import pandas as pd
import os
from datetime import datetime,timedelta

class LabelingTest(unittest.TestCase):

    def setUp(self):
        self.flare_filename = 'Data/hek_flare_catalog.csv'
        self.flare_catalog = pd.read_csv(self.flare_filename)
        self.index_file = 'Data/index_MDI.csv'
        self.out_file = 'src/tests/labels_MDI.csv'
        self.mdi_dir = 'Data/MDI/1999'
        self.mdi_file = self.mdi_dir+os.listdir(self.mdi_dir)[0]
        if not os.path.exists(self.index_file):
            df = pd.DataFrame({'fname_MDI':[self.mdi_file],
                                    'date':[19990204],
                                    'time_MDI':[000000],
                                    'timestamp_MDI':[datetime(1999,2,1,0)]})
            df.to_csv(self.index_file,index=False)
        self.header = ['filename','sample_time','dataset','tot_us_flux','tot_flux','datamin','datamax','C_flare_in_24h','M_flare_in_24h','X_flare_in_24h','flare_intensity_in_24h']
        self.labeler = Labeler(self.index_file,self.out_file,self.flare_filename)

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

    def test_flareLabels(self):
        file_data = add_label_data(self.flare_catalog)
        self.assertEqual(file_data,[1,1,1,'{:0.1e}'.format(self.flare_catalog['intensity'].max())])

    def test_readCatalog(self):
        df = read_catalog(self.flare_filename)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df['start_time']))

    def test_writeHeader(self):
        self.labeler.write_header()
        df = pd.read_csv(self.out_file)
        self.assertTrue(all(x==y for x,y in zip(df.columns,self.header)))

    def test_fileData(self):
        samples = read_catalog(self.index_file,na_values=' ')
        sample = samples.iloc[0]
        file_data = self.labeler.generate_file_data(sample)
        print(file_data)
        self.assertIsInstance(file_data,list)

    def test_labelData(self):
        self.labeler.label_data()
        df = pd.read_csv(self.out_file)
        samples = read_catalog(self.index_file,na_values=' ')
        self.assertGreaterEqual(len(df),1)
        self.assertLessEqual(len(df),len(samples))

if __name__ == "__main__":
    unittest.main()