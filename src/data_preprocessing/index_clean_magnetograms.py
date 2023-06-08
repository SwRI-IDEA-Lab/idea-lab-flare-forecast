"""
Script to index and clean directories of magnetogram fits files that are 
organized by year. Works for MDI, HMI, SPMG, 512 and MWO files. Does not yet 
extract any metadata from the fits headers. 
"""

import sys,os
sys.path.append(os.getcwd())

from astropy.convolution import convolve, convolve_fft, Gaussian2DKernel
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

class Indexer:
    def __init__(self, data:str, data_dir:str='Data', save_dir:str='Data/hdf5', index_dir:str='Data', metadata_cols:list=[], nworkers:int=4):
        """
        Initialize an indexing class to iterate through data, cleaning and indexing
        
        Parameters:
            data (str):             magnetogram instrument (HMI,MDI,512,SPMG or MWO)
            data_dir (str):         root dir where the data is stored 
            save_dir (str):         dir for new hdf5 files to be saved
            index_dir (str):        location for generated index to be saved
            metadata_cols (list):   list of fits header keys to save 
        """
        self.data = data
        self.root_dir = Path(data_dir)
        self.new_dir = Path(save_dir)/self.data
        if not os.path.exists(self.new_dir):
            os.mkdir(self.new_dir)
        self.file = index_dir +'/index_'+data+'smoothed.csv'
        self.metadata_cols = metadata_cols
        self.nworkers = nworkers

        # header data for csv file
        header = ['filename','fits_file','timestamp']
        header.extend(metadata_cols)
        header.extend(['tot_us_flux','tot_flux','datamin','datamax'])
        header = [key+'_'+data for key in header]
        header.insert(2,'date')

        # write header to csv file
        with open(self.file,'w') as out_file:
            out_writer = csv.writer(out_file,delimiter=',')
            out_writer.writerow(header)
    
    def index_data(self):
        """
        Clean and index data in root dir by year. Save index to csv file.
        
        Parameters:
            nworkers (int):     Number of processes to index years in parallel
        """
        # open csv file writer
        out_file = open(self.file,'a')
        out_writer = csv.writer(out_file,delimiter=',')

        # iterate through files and add to index
        years = sorted(os.listdir(self.root_dir/self.data))
        args = [(year,True,False) for year in years]

        # index years in parallel and write results to csv
        self.error_files = []
        with Pool(self.nworkers) as pool:
            for result in pool.starmap(self.index_year,args):
                index = result[0]
                self.error_files.extend(result[1])
                out_writer.writerows(index) 

        out_file.close()
        print('Finished indexing',self.data)    
        print('Errors on:',self.error_files)

    def index_year(self,year,onefileperday=True,test=False):
        """
        Indexes fits files within a specified directory by writing metadata
        to a csv writer. Nothing will be written to the index if the file has 
        a bad quality flag.

        Parameters:
            year (str):         subdirectory containing files, sorted by year
            onefileperday (bool):   option to only index the first file from each day
            test (bool):        option to stop indexing after 5 files
        
        Returns:
            index:              list of index data for all files in year
            error_files:        list of any files which threw errors
        """
        n = 0
        index = []
        t0 = time.time()
        lastdate = '19600101'
        error_files = []

        if not os.path.isdir(self.root_dir/self.data/year):
            # check that this is acually a directory
            return index,error_files
        if not os.path.exists(self.new_dir/year):
            # create new year directory if needed
            os.mkdir(self.new_dir/year)
        
        for file in sorted(os.listdir(self.root_dir/self.data/year)):
            # extract date and time from filename
            date,timestamp = extract_date_time(file,self.data,year)

            # only index the first file per date
            if date == None or (onefileperday and date == lastdate):
                continue

            try:
                # open file
                with fits.open(self.root_dir/self.data/year/file,cache=False) as data_fits:
                    data_fits.verify('fix')

                    img,header = extract_fits(data_fits,self.data)           

                # index_file
                index_data = self.index_item(self.root_dir/self.data/year/file,img,header,date,timestamp,self.new_dir/year)
            except ValueError:
                error_files.append(str(self.root_dir/self.data/year/file))
                continue

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
        print(n, 'files indexed for ',self.data,year,'in',round((t1-t0)/60,2),'minutes')
        return index, error_files
    
    def index_item(self,file,img,header,date,timestamp,new_dir):
        """
        Obtains index data for a given fits file

        Parameters:
            file (Path):    full path to data
            img (array):    data from fits file
            header (dict):  header from fits file
            date (str):     in %Y%m%d format
            timestamp (datetime):   as extracted from filename
            new_dir:        path to save file at
        
        Returns:
            index_data (list):  data to save about file, None if file is bad quality
        """    

        # clean (don't add file to index) by checking quality flag
        if not check_quality(self.data,header):
            return None

        # fix historical data headers
        if self.data in ['MWO','512','SPMG']:
            header = fix_header(header,self.data,timestamp)

        # calibrate and smooth based on instrument
        img = calibrate_img(img,self.data)

        # create sunpy map, reproject and calculate total unsigned flux on reprojection
        map = Map(img,header)
        rot_map = scale_rotate(map)
        reproject_map = reprojectToVirtualInstrument(rot_map,dim=1024,radius=1*u.au,scale=4*u.Quantity([0.55,0.55],u.arcsec/u.pixel))
        
        # zero out limbs for computing flux statistics
        out_map = zeroLimbs(reproject_map,radius=0.95,fill_value=0)
        tot_flux, tot_us_flux = compute_tot_flux(out_map,Blim=30,area_threshold=64)

        # save new hdf5 file
        new_file = new_dir /(self.data +'_magnetogram.' + datetime.strftime(timestamp,'%Y%m%d_%H%M%S') +'_TAI.h5')
        with h5py.File(new_file,'w') as h5:
            h5.create_dataset('magnetogram',data=reproject_map.data,compression='gzip')

        # store metadata
        index_data = [str(new_file),str(file),date,timestamp]
        index_data.extend([header[key] for key in self.metadata_cols])
        index_data.extend([tot_us_flux,tot_flux,np.nanmin(out_map.data),np.nanmax(out_map.data)])

        return index_data

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
    parser.add_argument('-i','--indexdir',
                        type=str,
                        default='Data',
                        help='directory to save index file'
                        )
    return parser.parse_args(args)


def merge_indices_by_date(root_dir,datasets):
    """
    Merge generated indices across datasets by date

    Parameters:
        root_dir (str):     location of index files
        datasets (list):    names of data being indexed and merged
    """
    df_merged = pd.DataFrame({'date':[]})
    for data in datasets:
        filename = Path(root_dir)/('index_'+data.upper()+'.csv')
        df = pd.read_csv(filename)
        df_merged = df_merged.merge(df,how='outer',on='date',sort=True)
    print(len(df_merged),'entries in merged index')
    print(df_merged.head)
    return df_merged

def main():
    parser = parse_args()
    datasets = parser.data
    root_dir = parser.root
    new_dir = parser.newdir
    index_dir = parser.indexdir

    if not os.path.exists(new_dir):
        os.mkdir(new_dir)

    for data in datasets:
        data = data.upper()
        indexer = Indexer(data,root_dir,new_dir,index_dir,['t_obs'],nworkers=8)
        indexer.index_data()

    if len(datasets)>1:
        df_merged = merge_indices_by_date(index_dir,datasets)
        filename_merged = '_'.join([data for data in datasets])
        df_merged.to_csv(index_dir+'/index_'+filename_merged,index=False)

if __name__ == '__main__':
    main()





