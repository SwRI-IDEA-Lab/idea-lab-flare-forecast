import sys,os
sys.path.append(os.getcwd())

from astropy.io import fits
import glob
import numpy as np
import pandas as pd
import argparse
from datetime import datetime,timedelta
import csv
import matplotlib.pyplot as plt
from multiprocessing import Pool
import scipy as sp
from src.data_preprocessing.helper import read_catalog, add_label_data, calculate_flaring_rate
from sunpy import timeseries as ts
import time

def fix_artifacts(goes_ts):
    # fix values for detecting artifacts
    deriv_max = 1.5/3
    deriv_min = -1/3
    thresh_min = -7.8
    thresh_max = -3.5
    diff_normal = 0.3
    tback = 120

    dt = np.median((goes_ts.index[1:]-goes_ts.index[:-1]).total_seconds())

    size_filt = np.min([int(np.floor(12/dt)),12])
    origin_filt =int(np.floor((size_filt-1)/2))

    goes_ts_new = np.log10(sp.ndimage.median_filter(goes_ts['xrsb'].copy(),size_filt,origin=origin_filt))
    goes_ts_new[np.isnan(goes_ts_new)] = np.nanmin(goes_ts_new)

    mask = np.zeros(len(goes_ts_new))

    # backwards differences to estimate derivatives
    ts_diff = np.diff(goes_ts_new,prepend=goes_ts_new[0])/dt
    ts_diff2 = np.diff(goes_ts_new,prepend=[goes_ts_new[0],goes_ts_new[0]])

    # find critical points
    inds_start = np.where(((ts_diff>deriv_max)&(goes_ts_new>thresh_max))|((goes_ts_new<thresh_min)&(ts_diff<0)))[0]

    inds_further=0
    for ind in inds_start:
        if (ind < inds_further) | (ind == 0):
            continue

        if (goes_ts_new[ind]<thresh_min):
            # if np.median(goes_ts_new[np.max([ind-30*60*24*20,0]):ind])<thresh_min:
            #     # all recent data is low noise, so ignore
            #     continue
            # look for previous inflection point
            ind = np.argmin(ts_diff2[np.max([ind-tback,0]):ind]) + np.max([ind-tback,0]) - 1
            # if too far from local median then just pick closest point to median
            if np.abs(goes_ts_new[ind-1]-np.median(goes_ts_new[np.max([0,ind-12000]):ind-1])) > 0.2*np.median(goes_ts_new[np.max([0,ind-12000]):ind-1]):
                ind = np.argmin(np.abs(goes_ts_new[np.max([ind-tback,0]):ind]-np.median(goes_ts_new[np.max([0,ind-12000]):ind])))+np.max([ind-tback,0])+1 

        elif (goes_ts_new[ind]>thresh_max):
            # look for previous inflection point
            ind = np.argmax(ts_diff2[np.max([ind-tback,0]):ind]) + np.max([ind-tback,0]) - 1
            # if too far from local median then just pick closest point to median
            if np.abs(goes_ts_new[ind-1]-np.median(goes_ts_new[np.max([0,ind-12000]):ind-1])) > 0.2*np.median(goes_ts_new[np.max([0,ind-12000]):ind-1]):
                ind = np.argmin(np.abs(goes_ts_new[np.max([ind-tback,0]):ind]-np.median(goes_ts_new[np.max([0,ind-12000]):ind])))+np.max([ind-tback,0])+1 

        # last normal value
        y_last = goes_ts_new[ind-1]
        # if there are more critical values in the next 30 minutes then search after them
        inds_further = np.max(np.append(inds_start[(goes_ts.index[inds_start]-goes_ts.index[ind])<=timedelta(minutes=30)],ind))
        # find next normal value
        ind_next = np.where(np.abs(goes_ts_new[inds_further:]-y_last)<diff_normal)
        # handle case where no normal values 
        if len(ind_next[0])==0:
            goes_ts_new[ind:] = goes_ts_new[ind-1]
            mask[ind:] = 1
            break
        ind_next = ind_next[0][0]+inds_further
        # handle case where nothing to interpolate
        if ind_next == ind:
            continue

        # linearly interpolate between last and next normal value
        goes_ts_new[ind:ind_next] = goes_ts_new[ind-1]+(goes_ts_new[ind_next]-goes_ts_new[ind-1])*(goes_ts.index[ind:ind_next]-goes_ts.index[ind-1])/(goes_ts.index[ind_next]-goes_ts.index[ind-1])
        # set mask
        mask[ind:ind_next] = 1

    mask = mask>0

    return goes_ts_new,mask,inds_start

class Labeler():
    def __init__(self,index_file:str = None,out_file:str = None,flare_catalog:str = None,
                 flare_windows:list=[24],goes_dir:str='Data/goes/',nworkers:int=4):
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
        self.goes_dir = goes_dir
        self.nworkers = nworkers

        # read in flare catalog
        self.flares = read_catalog(flare_catalog)

        # read in index file
        self.samples = read_catalog(index_file,na_values=' ')
        self.samples['year'] = self.samples['date'].astype(str).str[:4].astype(int)

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
        cols = [col.rstrip('_MDIHWOSPG512') for col in self.samples.columns if not col.rstrip('_MDIHWOSPG512') in ['filename','fits_file','date','year','timestamp','t_obs']]
        cols = list(dict.fromkeys(cols))  # filter only unique values
        header_row.extend(cols)
        header_row.extend(['flare_rate_y','flare_rate_m','flare_rate_w','max_flare_72h','max_flare_48h','max_flare_24h'])
        for window in self.flare_windows:
            header_row.append('xrsb_max_in_'+str(window)+'h')

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

        # label each year
        # for year in pd.unique(self.samples['year']):

        #         out_writer.writerow(file_data)

        t0 = time.time()
        years = pd.unique(self.samples['year'])

        with Pool(self.nworkers) as pool:
            for result in pool.map(self.label_year,years):
                out_writer.writerows(result) 

        out_file.close()
        t1 = time.time()
        print('Finished labeling',years[0],'-',years[-1],'in',(t1-t0)/60,'minutes')

    def label_year(self,year):
        """
        Label a year of samples
        
        Parameters:
            year (int)       to be labeled
        
        Returns:
            labels (list)    label data for all samples in year
        """
        year_samples = self.samples[self.samples['year']==year]
        print('Generating labels for',len(year_samples),'samples in',year)

        goes_ts = self.retrieve_goes_year_data(year)
        no_year_data = False
        if len(goes_ts) == 0:
            no_year_data = True
            print('No year data for',year)
        labels = []
        for i in year_samples.index:
            sample = self.samples.iloc[i]
            if no_year_data:   # error obtaining timeseries for whole year so just find local timeseries
                goes_ts = self.retrieve_goes_data(datetime.strptime(str(sample['date']),'%Y%m%d'))
            file_data = self.generate_file_data(sample,goes_ts)
            labels.append(file_data)
        return labels


    def generate_file_data(self,sample,goes_ts):
        """
        For a given data sample, generates list of information for labels file

        Parameters:
            sample:     pandas series with filenames and times for this days sample
            goes_ts:    pandas dataframe containing GOES timeseries data
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
        # goes_ts = self.retrieve_goes_data(sample_time)
        if len(goes_ts) == 0:
            goes_ts_sample = []
        else:
            goes_ts_sample = goes_ts[(goes_ts.index>sample_time)&(goes_ts.index<=sample_time+timedelta(hours=max(self.flare_windows)))]

        for window in self.flare_windows:
            file_data.append(self.add_regression_data(goes_ts_sample,sample_time,window))

        return file_data
    
    def retrieve_goes_year_data(self,year):
        """
        Retrieve and clean GOES data for year plus days (up to max flare window)
        
        Parameters:
            year (int)      desired year

        Returns:
            goes_ts (dataframe):        GOES sunpy timeseries 
        """

        goes_files = glob.glob(self.goes_dir+str(year)+'/**')
        for i in range(int(np.ceil(max(self.flare_windows)/24))):
            sample_time_next = datetime(year+1,1,i+1)
            goes_files.extend(glob.glob(self.goes_dir+str(year+1)+'/**'+datetime.strftime(sample_time_next,'%y%m%d')+'**'))
        if len(goes_files) == 0:
            return []
        try:
            goes_ts = ts.TimeSeries(goes_files,concatenate=True).to_dataframe()
        except:
            print('Error on ',year)
            return []
        
        # calibrate data before 2001
        goes_ts.loc[goes_ts.index<datetime(2001,3,1),'xrsb'] = goes_ts.loc[goes_ts.index<datetime(2001,3,1),'xrsb']/0.7

        # clean all GOES data
        goes_ts.loc[goes_ts['xrsb']<2e-9,'xrsb'] = np.nan
        goes_ts.fillna(method='ffill',inplace=True)
        goes_ts.fillna(method='bfill',inplace=True)
        goes_ts_new,_,inds_start = fix_artifacts(goes_ts)
        goes_ts['xrsb_clean'] = 10**goes_ts_new

        plt.figure(figsize=(12,3))
        plt.plot(goes_ts.index,goes_ts['xrsb'])
        plt.plot(goes_ts.index,10**goes_ts_new)
        plt.plot(goes_ts.index[inds_start],goes_ts['xrsb'][inds_start],'*')
        plt.yscale('log')
        plt.savefig(str(year)+'_goes.png')

        return goes_ts
    
    def retrieve_goes_data(self,sample_time):
        """
        Retrieve GOES data ahead of sample (up to max flare window)
        
        Parameters:
            sample_time (datetime)      desired time

        Returns:
            goes_ts (dataframe):        GOES sunpy timeseries 
        """
        goes_files = glob.glob(self.goes_dir+str(sample_time.year)+'/**'+datetime.strftime(sample_time,'%y%m%d')+'**')
        for i in range(int(np.ceil(max(self.flare_windows)/24))):
            sample_time_next = sample_time+timedelta(days=i+1)
            goes_files.extend(glob.glob(self.goes_dir+str(sample_time_next.year)+'/**'+datetime.strftime(sample_time_next,'%y%m%d')+'**'))
        if len(goes_files) == 0:
            return []
        try:
            goes_ts = ts.TimeSeries(goes_files,concatenate=True).to_dataframe()
            goes_ts = goes_ts[(goes_ts.index <= sample_time+timedelta(hours=max(self.flare_windows))) & (goes_ts.index>sample_time)]
        except:
            print('Error on ',sample_time)
            return []
        
        # clean all GOES data
        if len(goes_ts) == 0:
            return []
        
        goes_ts.loc[goes_ts['xrsb']<2e-9,'xrsb'] = np.nan
        goes_ts.fillna(method='ffill',inplace=True)
        goes_ts.fillna(method='bfill',inplace=True)
        goes_ts_new,_,_ = fix_artifacts(goes_ts)
        goes_ts['xrsb_clean'] = 10**goes_ts_new

        return goes_ts
    

    def add_regression_data(self,goes_ts,sample_time,window):
        """
        Returns maximum of xrsb in timeseries over window. If no GOES data found, returns nan
        
        Parameters:
            goes_ts (dataframe)     GOES timeseries dataframe
            sample_time (datetime)  time of sample
            window (int)            in hours

        Returns:
            max value of GOES data over window
        """
        if len(goes_ts) == 0:
            return np.nan
        
        goes_max = goes_ts['xrsb_clean'][goes_ts.index<=sample_time+timedelta(hours=window)].max()
        return np.float32(goes_max)
        

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
    parser.add_argument('goesdir',
                        type=str,
                        help='path to GOES data',
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
    labeler = Labeler(parser.index_file, parser.out_file,
                      flare_catalog,parser.flare_windows,
                      goes_dir=parser.goesdir)
    labeler.write_header()
    labeler.label_data()

if __name__ == '__main__':
    main()