"""
Helper functions for data pre-processing
"""

from datetime import datetime,timedelta

def extract_date_time(file,data,year):
    """
    Extracts date and time from a magnetogram fits filename. 

    Parameters:
        file (str):         fits filename
        data (str):         can be 'MDI','HMI','SPMG','512',or 'MWO'
        year (str):         year or year month indicator

    Returns:
        date (str):         in %Y%m%d format
        time (str):         in %H%M%S format
        timestamp(datetime):datetime object 
    """
    if data == 'MDI':        
        file_split = file.split('.')
        date = file_split[-3].split('_')[0]
        time = file_split[-3].split('_')[1]
        timestamp = datetime.strptime(date+time,'%Y%m%d%H%M%S') 
    elif data == 'HMI':
        file_split = file.split('.')
        date = file_split[-4].split('_')[0]
        time = file_split[-4].split('_')[1] 
        timestamp = datetime.strptime(date+time,'%Y%m%d%H%M%S') 
    elif data == 'SPMG' or data == '512':
        file_split = file.strip('.fits').split('_')
        date = file_split[-2]
        time = file_split[-1]
        time = time.ljust(6,'0')
        timestamp = datetime.strptime(date+time,'%Y%m%d%H%M%S') 
    elif data == 'MWO':
        extra_day = 0   # placeholder for correcting time data
        # multiple naming conventions for Mt Wilson Observatory data
        file_split = file.strip('.fits').split('_')
        if year[0:2]=='YR':
            return None,None,None
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
        time = time.ljust(6,'0')
        timestamp = datetime.strptime(date+time,'%Y%m%d%H%M%S') + timedelta(days=extra_day)
        date = datetime.strftime(timestamp,'%Y%m%d')
    else:
        raise ValueError('Invalid data type: ',data,'\n Must be MDI,HMI,SPMG,512 or MWO')


    return date,time,timestamp