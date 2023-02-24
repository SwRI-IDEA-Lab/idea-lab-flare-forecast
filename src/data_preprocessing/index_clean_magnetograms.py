"""
Script to index and clean directories of magnetogram fits files that are 
organized by year. Works for MDI, HMI, SPMG, 512 and MWO files. Does not yet 
extract any metadata from the fits headers. 
"""

from astropy.io import fits
import os
import csv
import tarfile
from datetime import datetime,timedelta
from pathlib import Path
# Remove Warnings
import warnings
warnings.filterwarnings('ignore')

# set up
data = 'SPMG'
root_dir = Path('../../Data/')
filename = root_dir / 'index_spmg.csv'

# header data for csv file
header = ['filename','date','time','timestamp']

# open csv file writer
out_file = open(filename,'w')
out_writer = csv.writer(out_file,delimiter=',')
out_writer.writerow(header)

# iterate through files and add to index
for year in sorted(os.listdir(root_dir / data)):
    for file in sorted(os.listdir(root_dir / data / year)):

        # extract date and time from filename
        extra_day = 0   # placeholder for correcting MWO dates
        if data == 'MDI':        
            file_split = file.split('.')
            date = file_split[-3].split('_')[0]
            time = file_split[-3].split('_')[1]
        elif data == 'HMI':
            file_split = file.split('.')
            date = file_split[-4].split('_')[0]
            time = file_split[-4].split('_')[1] 
        elif data == 'SPMG' or data == '512':
            file_split = file.strip('.fits').split('_')
            date = file_split[-2]
            time = file_split[-1]
        elif data == 'MWO':
            # multiple naming conventions for Mt Wilson Observatory data
            file_split = file.strip('.fits').split('_')
            if year[0:2]=='YR':
                break
            if int(year) < 1995:
                date = '19'+file_split[-2][1:]
                time = file_split[-1]
            elif int(year) < 2000:
                date = '19'+file_split[-3].strip('m')
                time = file_split[-2]
            else:
                date = '20'+file_split[-3].strip('m')
                time = file_split[-2]    
            # correct errors in filenames from time rounding
            if int(time[2:])>59:
                time = str(int(time[:2])+1) + str(int(time[2:])-60).zfill(2)
            if int(time[:2])>23:
                time = str(int(time[:2])-24).zfill(2)+time[2:]
                extra_day = 1

        # convert to datetime object
        timestamp = datetime.strptime(date+time,'%Y%m%d%H%M%S') + timedelta(days=extra_day)
        
        # store filename, date, time and timestamp 
        index_data = [root_dir/data/year/file,date,time,timestamp]

        # open file
        data_fits = fits.open(root_dir/data/year/file,cache=False)
        data_fits.verify('fix')
        if data in ['HMI','MDI']:
            header = data_fits[1].header
        elif data in ['SPMG','512','MWO']:
            header = data_fits[0].header

        # clean (don't add file to index) by checking quality flag
        if data == 'MDI':
            # for MDI, good quality images have quality keyword 512 or 513
            if header['QUALITY'] != 512 and header['QUALITY'] != 513:
                continue
        elif data == 'HMI':
            # for HMI, good quality images have quality keyword 0
            if header['QUALITY'] != 0:
                continue            

        # write metadata to file
        out_writer.writerow(index_data)
    
    print(year)

out_file.close()





