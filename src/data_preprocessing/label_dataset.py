import sys,os
sys.path.append(os.getcwd())

from astropy.io import fits
import pandas as pd
import argparse
from datetime import datetime,timedelta
import csv
from src.data_preprocessing.helper import read_catalog, add_label_data, calculate_flaring_rate

class Labeler():
    def __init__(self,index_file:str = None,out_file:str = None,flare_catalog:str = None,flare_windows:list=[24]):
        """
        Initialize a labeling class to select best available data and add flare labels

        Parameters:
            index_file (str):       Path to index of data
            out_file (str):         Filename to save labeled dataset
            flare_catalog (str):    Path to index of flares
            flare_windows (list):   Forecast windows in hours
        """
        self.flare_windows = flare_windows   
        self.file = out_file

        # read in flare catalog
        self.flares = read_catalog(flare_catalog)

        # read in index file
        self.samples = read_catalog(index_file,na_values=' ')
        # drop samples with the same date, keeping the first entry
        self.samples.drop_duplicates(subset='date',keep='first',ignore_index=True,inplace=True)

        # set start date for dataset as the max forecast window before the first flare in the catalog
        start_date = int(datetime.strftime(self.flares['start_time'][0]-timedelta(hours = max(self.flare_windows)),'%Y%m%d'))
        # discard samples earlier than the start date
        self.samples = self.samples[self.samples['date']>=start_date]
        self.samples.reset_index(drop=True,inplace=True)

    def write_header(self):
        """
        Writes header columns to labels file
        """
        # header columns
        header_row = ['filename','sample_time','dataset']
        cols = [col.rstrip('_MDIHWOSPG512') for col in self.samples.columns if not col.rstrip('_MDIHWOSPG512') in ['filename','fits_file','date','timestamp','t_obs']]
        cols = list(dict.fromkeys(cols))  # filter only unique values
        header_row.extend(cols)
        header_row.extend(['flare_rate_y','flare_rate_m','flare_rate_w','max_flare_72h','max_flare_48h','max_flare_24h'])
        for window in self.flare_windows:
            header_row.append('C_flare_in_'+str(window)+'h')
            header_row.append('M_flare_in_'+str(window)+'h')
            header_row.append('X_flare_in_'+str(window)+'h')
            header_row.append('flare_intensity_in_'+str(window)+'h')

        # write header to file
        print(header_row)
        with open(self.file,'w') as out_file:
            out_writer = csv.writer(out_file,delimiter=',')
            out_writer.writerow(header_row)

        return 
    
    def label_data(self):
        """
        Iterate through index and write flare label data to file
        """
        out_file = open(self.file,'a')
        out_writer = csv.writer(out_file,delimiter=',')

        for i in self.samples.index:
            sample = self.samples.iloc[i]
            file_data = self.generate_file_data(sample)
            out_writer.writerow(file_data)

        out_file.close()
    
    def generate_file_data(self,sample):
        """
        For a given data sample, generates list of information for labels file

        Parameters:
            sample:     pandas series with filenames and times for this days sample

        Returns:
            file_data:  list of data to write to labels file for this sample
        """
        # order of preference of datasets
        datasets = ['HMI','MDI','SPMG','512','MWO']   

        # find prefered dataset for that day out of those available
        for dataset in datasets:
            if 'filename_'+dataset not in sample:
                continue
            if pd.notna(sample['filename_'+dataset]):
                fname = sample['filename_'+dataset]
                sample_time = sample['timestamp_'+dataset]
                data = dataset
                file_data = [fname,sample_time,data]
                file_data.extend(list(sample.loc[sample.index.str.endswith('_'+dataset)])[4:])
                break

        # calculate flaring rates
        for window in [365,30,7]:
            # filter flares
            flare_data = self.flares[(self.flares['peak_time']>=sample_time-timedelta(days=window))&(self.flares['peak_time']<=sample_time)]
            file_data.append(calculate_flaring_rate(flare_data,window))
        
        # calculate max historical flares
        for window in [72,48,24]:
            flare_data = self.flares[(self.flares['peak_time']>=sample_time-timedelta(hours=window))&(self.flares['peak_time']<=sample_time)]
            if len(flare_data) == 0:
                file_data.append(0)
            else:
                file_data.append(flare_data['intensity'].max())

        # add flare labels for each forecast window
        for window in self.flare_windows:
            flare_data = self.flares[(self.flares['peak_time']>=sample_time)&(self.flares['peak_time']<=sample_time+timedelta(hours=window))]
            file_data.extend(add_label_data(flare_data))

        return file_data
    

def parse_args(args=None):
    """
    Parses command line arguments to script. Sets up argument for which 
    dataset to label.

    Parameters:
        args (list):    defaults to parsing any command line arguments
    
    Returns:
        parser args:    Namespace from argparse
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('index_file',
                        type=str,
                        help='path to index file for labeling')
    parser.add_argument('out_file',
                        type=str,
                        help='filename to save labels data'
                        )
    parser.add_argument('-w','--flare_windows',
                        type=int,
                        nargs='*',
                        default=[12,24,48],
                        help='forecast windows for labeling, in hours'
                        )

    return parser.parse_args(args)


def main():
    # parse command line arguments
    parser = parse_args()
    flare_catalog = 'Data/hek_flare_catalog.csv'

    # create labeler instance
    labeler = Labeler(parser.index_file, parser.out_file,flare_catalog,parser.flare_windows)
    labeler.write_header()
    labeler.label_data()

if __name__ == '__main__':
    main()