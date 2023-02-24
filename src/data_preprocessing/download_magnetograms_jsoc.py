"""
Script to download MDI and HMI magnetograms from JSOC as fits files
"""

import os
import drms
import time
from datetime import datetime
import csv
from pathlib import Path

# set up download parameters
email = 'kiera.vds@gmail.com'
export_protocol = 'fits'    # 'fits' will download header data as well
series = 'hmi.M_720s'       # 'hmi.M_720s' or 'mdi.fd_M_96m_lev182'
data = 'HMI'                # 'MDI'
root = Path('../../Data/'+data)   # where to save data
cadence = '1d'              # desired cadence (much less than 1 day may require 
                            # breaking requests down to chunks smaller than a year)
segments = 'magnetogram'    

# could alternatively specify desired start and end times
if data == 'HMI':
    yr_start = 2010
    yr_end = 2023
elif data == 'MDI':
    yr_start = 1996
    yr_end = 2011

if not os.path.exists(root):
    os.mkdir(root)

for year in range(yr_start,yr_end):
    t_start = str(year)+'.01.01_00:00:00_TAI'
    t_end = str(year)+'.12.31_23:59:00_TAI'

    print('Working on',data,'from', t_start, 'to',t_end)

    save_dir = root / str(year)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    qstr = series+'[' + t_start + '-' + t_end + '@' + cadence + ']'

    client = drms.Client(verbose=False)

    # export data
    t0 = time.time()
    print("     Running the export command...")
    result = client.export(qstr, method='url', protocol=export_protocol, email=email) 

    # loop to wait for query to go through
    while True:
        try: 
            if result.status == 2 and result.id != None:
                result.wait(timeout=20*60)
                break
            time.sleep(10)
        except:
            time.sleep(10)
        if time.time()-t0 > 20*60:
            print('Failed to export query after 20 min...')
            break

    status = result.status
    print("     client.export.status = ",status)
    if status != 0:
        print("*********** DRMS error for",data, "year", year)
        break
    print("         Export request took ", time.time() - t0, ' seconds')

    # Download the files
    t1 = time.time()

    print('\nRequest URL: %s' % result.request_url)
    if '%s' % result.request_url != 'None':
        print('%d file(s) available for download.\n' % len(result.urls))
    print("     Running the download command...")
    result.download(save_dir, verbose=False)
    print("          Download took: ", time.time() - t1, "seconds")

    print("     ",data,"year", year, "took ", time.time() - t0, " seconds")
    print("\n\n")


