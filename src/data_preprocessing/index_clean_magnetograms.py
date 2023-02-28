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
from helper import extract_date_time
# Remove Warnings
import warnings
warnings.filterwarnings('ignore')

# set up
data = 'HMI'

root_dir = Path('../../Data/')
filename = root_dir / ('index_'+data+'.csv')

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
        date,time,timestamp = extract_date_time(file,data,year)
        if date == None:
            continue
        
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





