import sys,os
sys.path.append(os.getcwd())

from astropy.io import fits
import pandas as pd
import argparse
from datetime import datetime,timedelta
import csv
from src.data_preprocessing.helper import read_catalog

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

def write_header(flare_windows,out_writer,cols=[]):
    """
    Writes header columns to labels file

    Parameters:
        flare_windows (list):   forecasting windows in hours
        out_writer:             csv writer object to write header to
        cols:                   additional columns for header
    """
    # header columns
    header_row = ['filename','sample_time','dataset']
    cols = [col.rstrip('_MDIHWOSPG512') for col in cols if not col.rstrip('_MDIHWOSPG512') in ['filename','fits_file','date','timestamp','t_obs']]
    cols = list(set(cols))  # filter only unique values
    header_row.extend(cols)
    for window in flare_windows:
        header_row.append('C_flare_in_'+str(window)+'h')
        header_row.append('M_flare_in_'+str(window)+'h')
        header_row.append('X_flare_in_'+str(window)+'h')
        header_row.append('flare_intensity_in_'+str(window)+'h')

    # write header to file
    print(header_row)
    out_writer.writerow(header_row)
    return 

def add_label_data(flare_data,file_data):
    """
    Adds flare labeling data to a list

    Parameters:
        flare_data (dataframe):     relevant catalog of flares
        file_data (list):           list of data to append labels to

    Returns:
        file_data (list):           with appended labels for C, M, X flares 
                                    and flare intensity
    """
    for flare_class in ['C','M','X']:
        if sum(flare_data['CMX']==flare_class) > 0:
            file_data.append(1)
        else:
            file_data.append(0)
    if len(flare_data)>0:
        file_data.append('{:0.1e}'.format(max(flare_data['intensity'])))
    else:
        file_data.append(0)
    return file_data

def generate_file_data(sample,flares,flare_windows):
    """
    For a given data sample, generates list of information for labels file

    Parameters:
        sample:     pandas series with filenames and times for this days sample
        flares:     dataframe containing the flare catalog
        flare_windows:  list of forecasting windows
    
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
    
    # add flare labels for each forecast window
    for window in flare_windows:
        flare_data = flares[(flares['peak_time']>=sample_time)&(flares['peak_time']<=sample_time+timedelta(hours=window))]
        file_data = add_label_data(flare_data,file_data)

    return file_data

def main():
    # parse command line arguments
    parser = parse_args()
    flare_windows = parser.flare_windows
    out_filename = parser.out_file

    # read in index file
    samples = read_catalog(parser.index_file,na_values=' ')

    # drop samples with the same date, keeping the first entry
    samples.drop_duplicates(subset='date',keep='first',ignore_index=True,inplace=True)

    # open flare catalog
    flares = read_catalog('Data/hek_flare_catalog.csv')

    # set start date for dataset as the max forecast window before the first flare in the catalog
    start_date = int(datetime.strftime(flares['start_time'][0]-timedelta(hours = max(flare_windows)),'%Y%m%d'))
    # discard samples earlier than the start date
    samples = samples[samples['date']>=start_date]
    samples.reset_index(drop=True,inplace=True)

    # open labels file and write header
    out_file = open(out_filename,'w')
    out_writer = csv.writer(out_file,delimiter=',')
    write_header(flare_windows,out_writer,samples.columns)

    # iterate through index and add flare label data
    for i in samples.index:
        sample = samples.iloc[i]
        file_data = generate_file_data(sample,flares,flare_windows)
        out_writer.writerow(file_data)

    out_file.close()

if __name__ == '__main__':
    main()