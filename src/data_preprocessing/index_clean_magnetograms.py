"""
Script to index and clean magnetogram files. Works for MDI or HMI files. 
Does not yet extract any metadata from the fits headers.
"""

from astropy.io import fits
import os
import csv
import tarfile
from datetime import datetime,timedelta
from pathlib import Path

# set up
data = 'HMI'
root_dir = Path('../../Data/')
filename = root_dir / 'index_hmi.csv'

# header data for csv file
header = ['filename','date','time','timestamp']

# open csv file writer
out_file = open(filename,'w')
out_writer = csv.writer(out_file,delimiter=',')
out_writer.writerow(header)

# iterate through files and add to index
for year in sorted(os.listdir(root_dir / data)):
    for file in sorted(os.listdir(root_dir / data / year)):
        file_split = file.split('.')
        # extract date and time from filename
        if data == 'MDI':
            date = file_split[-3].split('_')[0]
            time = file_split[-3].split('_')[1]
        elif data == 'HMI':
            date = file_split[-4].split('_')[0]
            time = file_split[-4].split('_')[1] 

        # convert to datetime object
        timestamp = datetime.strptime(date+time,'%Y%m%d%H%M%S')
        # store filename, date, time and timestamp 
        index_data = [root_dir/data/year/file,date,time,timestamp]

        # open file
        data_fits = fits.open(root_dir/data/year/file,cache=False)
        header = data_fits[1].header

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





