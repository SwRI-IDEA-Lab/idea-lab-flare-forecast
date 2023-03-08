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
        self.out_file = 'Data/labels_MDI.csv'
        self.mdi_dir = 'Data/MDI/1999'
        self.mdi_file = self.mdi_dir+os.listdir(self.mdi_dir)[0]
        if not os.path.exists(self.index_file):
            df = pd.DataFrame({'fname_MDI':[self.mdi_file],
                                    'date':[19990204],
                                    'time_MDI':[000000],
                                    'timestamp_MDI':[datetime(1999,2,1,0)]})
            df.to_csv(self.index_file,index=False)
        self.header = ['filename','sample_time','dataset','tot_flux','C_flare_in_12h','M_flare_in_12h','X_flare_in_12h','flare_intensity_in_12h']

    def test_parser(self):
        parser = parse_args([self.index_file,self.out_file,'-w 12'])
        self.assertIsInstance(parser.index_file,str)
        self.assertTrue(os.path.exists(parser.index_file))
        self.assertIsInstance(parser.out_file,str)
        self.assertIsInstance(parser.flare_windows,list)
        self.assertIsInstance(parser.flare_windows[0],int)

    def test_flareLabels(self):
        file_data = add_label_data(self.flare_catalog,[])
        self.assertEqual(file_data,[1,1,1,'{:0.1e}'.format(self.flare_catalog['intensity'].max())])

    def test_readCatalog(self):
        df = read_catalog(self.flare_filename)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df['start_time']))

    def test_writeHeader(self):
        out_file = open(self.out_file,'w')
        out_writer = csv.writer(out_file,delimiter=',')
        write_header([12],out_writer,['tot_flux'])
        out_file.close()
        df = pd.read_csv(self.out_file)
        self.assertTrue(all(x==y for x,y in zip(df.columns,self.header)))

    def test_fileData(self):
        samples = read_catalog(self.index_file,na_values=' ')
        sample = samples.iloc[0]
        flares = read_catalog(self.flare_filename)
        file_data = generate_file_data(sample,flares,[12])
        print(file_data)
        self.assertIsInstance(file_data,list)

if __name__ == "__main__":
    unittest.main()