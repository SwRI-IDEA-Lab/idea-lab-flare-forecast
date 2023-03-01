"""
Script to index and clean directories of magnetogram fits files that are 
organized by year. Works for MDI, HMI, SPMG, 512 and MWO files. Does not yet 
extract any metadata from the fits headers. 
"""

import sys,os
sys.path.append(os.getcwd())

from astropy.io import fits
import os
import csv
import tarfile
from datetime import datetime,timedelta
from pathlib import Path
from src.data_preprocessing.helper import extract_date_time
import argparse
import sys
import pandas as pd
# Remove Warnings
import warnings
warnings.filterwarnings('ignore')

def parse_args(args=None):
    """
    Parses command line arguments to script. Sets up argument for which 
    dataset to index.

    Parameters:
        args (list):    defaults to parsing any command line arguments
    
    Returns:
        parser args:    Namespace from argparse
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('data',
                        type=str,
                        nargs='*',
                        default=['mdi'],
                        help='dataset to index')
    return parser.parse_args(args)

def check_quality(data,header):
    """
    Check fits header files for quality keyword and handles MDI or HMI 
    bad quality values.

    Parameters:
        data (str):     which dataset we are checking
        header (dict or fits header)
    
    Returns:
        bool:           True if no bad quality flags for MDI or HMI data
    """
    if data == 'MDI':
        # for MDI, check for many missing values
        if header['MISSVALS'] > 2000:
            return False
    elif data == 'HMI':
        # for HMI, good quality images have quality keyword 0
        if header['QUALITY'] != 0:
            return False
    return True

def index_year(root_dir,data,year,out_writer):
    """
    Indexes fits files within a specified directory by writing filenames, date,
    time and timestamp information to a csv writer. Nothing will be written to 
    the index if the file has a bad quality flag.

    Parameters:
        root_dir (Path):    root directory for data
        data (str):         which dataset (MDI,HMI,SPMG,512,MWO...)
        year (str):         subdirectory containing files, sorted by year
        out_writer (csv writer):    where to write index to
    
    Returns:
        n (int):            number of files written to index
    """
    n = 0
    for file in sorted(os.listdir(root_dir/data/year)):
        # extract date and time from filename
        date,time,timestamp = extract_date_time(file,data,year)
        if date == None:
            continue
        
        # store filename, date, time and timestamp 
        index_data = [root_dir/data/year/file,date,time,timestamp]

        # open file
        with fits.open(root_dir/data/year/file,cache=False) as data_fits:
        # data_fits = fits.open(root_dir/data/year/file,cache=False)
            data_fits.verify('fix')
            if data in ['HMI','MDI']:
                header = data_fits[1].header
            elif data in ['SPMG','512','MWO']:
                header = data_fits[0].header

        # clean (don't add file to index) by checking quality flag
        if not check_quality(data,header):
            continue

        data_fits.close()

        # write metadata to file
        out_writer.writerow(index_data)
        n += 1

    return n

def merge_indices_by_date(root_dir,datasets):
    """
    Merge generated indices across datasets by date

    Parameters:
        root_dir (Path):    location of index files
        datasets (list):    names of data being indexed and merged
    """
    df_merged = pd.DataFrame({'date':[]})
    for data in datasets:
        filename = root_dir/('index_'+data+'.csv')
        df = pd.read_csv(filename)
        df.rename(columns={'filename':'fname_'+data,'time':'time_'+data,'timestamp':'timestamp_'+data},inplace=True)
        df_merged = df_merged.merge(df,how='outer',on='date',sort=True)
    print(len(df_merged),'entries in merged index')
    return df_merged

def main():
    datasets = parse_args().data
    root_dir = Path('Data')

    for data in datasets:
        data = data.upper()
        filename = root_dir / ('index_'+data+'.csv')

        # header data for csv file
        header = ['filename','date','time','timestamp']

        # open csv file writer
        out_file = open(filename,'w')
        out_writer = csv.writer(out_file,delimiter=',')
        out_writer.writerow(header)

        # iterate through files and add to index
        N = 0
        for year in sorted(os.listdir(root_dir / data)):
            n = index_year(root_dir,data,year,out_writer)
            print(n,'files for',data,year)
            N += n

        out_file.close()
        print(N,'files for',data)
    
    if len(datasets)>0:
        df_merged = merge_indices_by_date(root_dir,datasets)
        filename_merged = '_'.join([data for data in datasets])
        df_merged.to_csv(root_dir/('index_'+filename_merged),index=False)

if __name__ == '__main__':
    main()





