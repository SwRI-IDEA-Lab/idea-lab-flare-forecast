"""
Script to index and clean directories of magnetogram fits files that are 
organized by year. Works for MDI, HMI, SPMG, 512 and MWO files. Does not yet 
extract any metadata from the fits headers. 
"""

import sys,os
sys.path.append(os.getcwd())

from astropy.io import fits
import astropy.units as u
import os
import csv
import tarfile
from datetime import datetime,timedelta
import h5py
from pathlib import Path
from src.data_preprocessing.helper import *
from src.utils.utils import reprojectToVirtualInstrument, scale_rotate, zeroLimbs
import argparse
from sunpy.map import Map
import sys
from multiprocessing import Pool
import pandas as pd
import time
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
    parser.add_argument('-r','--root',
                        type=str,
                        default='Data',
                        help='root directory for fits files'
                        )
    parser.add_argument('-n','--newdir',
                    type=str,
                    default='Data/hdf5',
                    help='directory to save cleaned hdf5 files'
                    )
    return parser.parse_args(args)

def index_item(file,img,header,data,date,timestamp,metadata_cols,new_dir):
    """
    Obtains index data for a given fits file

    Parameters:
        file (Path):    full path to data
        img (array):    data from fits file
        header (dict):  header from fits file
        data (str):     dataset (MWO,SPMG,512,MDI,HMI)
        date (str):     in %Y%m%d format
        timestamp (datetime):   as extracted from filename
        metadata_cols:  list of keys to index from fits header
        new_dir:        directory to save new cleaned file
    
    Returns:
        index_data (list):  data to save about file, None if file is bad quality
    """    

    # clean (don't add file to index) by checking quality flag
    if not check_quality(data,header):
        return None

    # fix historical data headers
    if data in ['MWO','512','SPMG']:
        header = fix_header(header,data,timestamp)

    # calibrate MDI
    if data == 'MDI':
        img = img/1.3
    
    # create sunpy map, reproject and calculate total unsigned flux on reprojection
    map = Map(img,header)
    rot_map = scale_rotate(map)
    reproject_map = reprojectToVirtualInstrument(rot_map,dim=1024,radius=1*u.au,scale=4*u.Quantity([0.55,0.55],u.arcsec/u.pixel))
    
    # zero out limbs for computing flux statistics
    out_map = zeroLimbs(reproject_map,radius=0.95,fill_value=0)
    tot_flux, tot_us_flux = compute_tot_flux(out_map,Blim=30,area_threshold=64)

    # save new hdf5 file
    if not os.path.exists(new_dir):
        os.mkdir(new_dir)
    new_file = new_dir / (data +'_magnetogram.' + datetime.strftime(timestamp,'%Y%m%d_%H%M%S') +'_TAI.h5')
    with h5py.File(new_file,'w') as h5:
        h5.create_dataset('magnetogram',data=reproject_map.data,compression='gzip')

    # store metadata
    index_data = [str(new_file),str(file),date,timestamp]
    index_data.extend([header[key] for key in metadata_cols])
    index_data.extend([tot_us_flux,tot_flux,np.nanmin(out_map.data),np.nanmax(out_map.data)])

    return index_data

def index_year(root_dir,data,year,metadata_cols,new_dir,onefileperday=True,test=False):
    """
    Indexes fits files within a specified directory by writing metadata
    to a csv writer. Nothing will be written to the index if the file has 
    a bad quality flag.

    Parameters:
        root_dir (Path):    root directory for data
        data (str):         which dataset (MDI,HMI,SPMG,512,MWO...)
        year (str):         subdirectory containing files, sorted by year
        metadata_cols (list):   list of keys to index from the fits header
        new_dir (Path):     directory to save cleaned files
        onefileperday (bool):   option to only index the first file from each day
        test (bool):        option to stop indexing after 5 files
    
    Returns:
        index:              list of index data for all files in year
    """
    n = 0
    index = []
    t0 = time.time()
    lastdate = '19600101'

    for file in sorted(os.listdir(root_dir/data/year)):
        # extract date and time from filename
        date,timestamp = extract_date_time(file,data,year)

        # only index the first file per date
        if date == None or (onefileperday and date == lastdate):
            continue

        # open file
        with fits.open(root_dir/data/year/file,cache=False) as data_fits:
            data_fits.verify('fix')
            img,header = extract_fits(data_fits,data)           

        # index_file
        index_data = index_item(root_dir/data/year/file,img,header,data,date,timestamp,metadata_cols,new_dir/year)
        if index_data == None:
            continue
        # file was indexed so save last date as current date
        lastdate = date

        # add metadata to list
        index.append(index_data)
        n += 1

        # cut off indexing after 5 files if testing
        if test and n>=5:
            break

    t1 = time.time()
    print(n, 'files indexed for ',data,year,'in',round((t1-t0)/60,2),'minutes')
    return index

def merge_indices_by_date(root_dir,datasets):
    """
    Merge generated indices across datasets by date

    Parameters:
        root_dir (Path):    location of index files
        datasets (list):    names of data being indexed and merged
    """
    df_merged = pd.DataFrame({'date':[]})
    for data in datasets:
        filename = root_dir/('index_'+data.upper()+'.csv')
        df = pd.read_csv(filename)
        df_merged = df_merged.merge(df,how='outer',on='date',sort=True)
    print(len(df_merged),'entries in merged index')
    print(df_merged.head)
    return df_merged


def main():
    parser = parse_args()
    datasets = parser.data
    root_dir = Path(parser.root)
    new_dir = Path(parser.newdir)

    for data in datasets:
        data = data.upper()
        filename = 'Data/index_'+data+'.csv'

        # header data for csv file
        header = ['filename','fits_file','timestamp']
        metadata_cols = ['t_obs']
        header.extend(metadata_cols)
        header.extend(['tot_us_flux','tot_flux','datamin','datamax'])
        header = [key+'_'+data for key in header]
        header.insert(2,'date')

        # open csv file writer
        out_file = open(filename,'w')
        out_writer = csv.writer(out_file,delimiter=',')
        out_writer.writerow(header)

        # iterate through files and add to index
        years = sorted(os.listdir(root_dir/data))
        args = [(root_dir,data,year,metadata_cols,new_dir/data) for year in years]

        # index years in parallel and write results to csv
        with Pool(8) as pool:
            for index in pool.starmap(index_year,args):
                out_writer.writerows(index) 

        out_file.close()
        print('Finished indexing',data)    

    if len(datasets)>1:
        df_merged = merge_indices_by_date(Path('Data'),datasets)
        filename_merged = '_'.join([data for data in datasets])
        df_merged.to_csv('Data/index_'+filename_merged,index=False)

if __name__ == '__main__':
    main()





