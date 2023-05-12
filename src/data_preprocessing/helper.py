"""
Helper functions for data pre-processing
"""

from datetime import datetime,timedelta
import numpy as np
from src.utils.utils import mapPixelArea, makeBMask
import pandas as pd

def read_catalog(filename,**kwargs):
    """
    Reads a csv file into a pandas dataframe and converts any columns
    to datetime that can be successfully converted

    Parameters:
        filename (str):     location of csv file
        kwargs:             optional keyword arguments to pass to pd.read_csv
    
    Returns:
        df:                 converted pandas dataframe

    """
    df = pd.read_csv(filename,**kwargs)
    for col in df.columns[df.dtypes=='object']:
        try:
            df[col] = pd.to_datetime(df[col])
        except (pd.errors.ParserError,ValueError):
            # if unable to convert to datetime, then leave unconverted
            pass
    return df

def add_label_data(flare_data):
    """
    Adds flare labeling data to a list

    Parameters:
        flare_data (dataframe):     relevant catalog of flares

    Returns:
        file_data (list):           list with labels for C, M, X flares 
                                    and flare intensity
    """
    file_data = []
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

def calculate_flaring_rate(flare_data,window):
    """
    Calculate the rate of flaring over the number of days covered by flare_data

    Parameters: 
        flare_data (dataframe):    catalog of relevant flares 
        window (int):              number of days to calculate flaring rate over
    
    Returns: 
        flare_rate (float):        rate of flaring over window
    """
    # filter only M flares
    flare_data = flare_data[flare_data['intensity']>=1e-5]
    # filter unique days
    peak_days = flare_data['peak_time'].dt.date
    unique_days = pd.unique(peak_days)
    # calculate flaring rate
    flare_rate = len(unique_days)/window
    return flare_rate

def extract_date_time(file,data,year):
    """
    Extracts date and time from a magnetogram fits filename. 

    Parameters:
        file (str):         fits filename
        data (str):         can be 'MDI','HMI','SPMG','512',or 'MWO'
        year (str):         year or year month indicator

    Returns:
        date (str):         in %Y%m%d format
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


    return date,timestamp

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
        if header['MISSVALS'] > 2000 or header['QUALITY'] == 262656:
            return False
    elif data == 'HMI':
        # for HMI, good quality images have quality keyword 0
        if header['QUALITY'] != 0:
            return False
    return True

def compute_tot_flux(map, Blim=30, area_threshold=64):
    """
    Calculate total signed and unsigned flux for magnetogram by multiplying with 
    pixel area and masking only larger regions of flux

    Parameters:
        map (sunpy map):    LOS magnetogram
        Blim (int):         min threshold for masking 
        area_threshold (int): min size of region to mask (pixels)
    
    Returns:
        tot_flux (float):   total signed magnetic flux
        tot_us_flux (float): total unsigned magnetic flux
    """
    area_map = mapPixelArea(map)
    Bdata = map.data*area_map.data
    Bmask = makeBMask(Bdata,Blim=Blim,area_threshold=area_threshold,connectivity=2,dilationR=8)
    tot_flux = np.nansum(Bdata*Bmask)
    tot_us_flux = np.nansum(np.abs(Bdata*Bmask))
    return tot_flux, tot_us_flux

def extract_fits(data_fits,dataset):
    """
    Extract image and header from fits data according to magnetogram dataset

    Parameters:
        data_fits:  opened fits file
        dataset (str):  which type of magnetogram (HMI,MDI,SPMG,512,MWO)
    
    Returns:
        img:        observations from fits file
        header:     header dictionary from fits file
    """
    if dataset in ['HMI','MDI']:
        img = data_fits[1].data
        header = data_fits[1].header
    elif dataset in ['MWO']:
        img = data_fits[0].data
        header = data_fits[0].header
    elif dataset in ['SPMG']:
        img = data_fits[0].data[5,:,:]
        header = data_fits[0].header
    elif dataset in ['512']:
        img = data_fits[0].data[2,:,:]
        header = data_fits[0].header  
    return img,header   

def fix_header(header,data,timestamp):
    """
    Fill in missing header values for historical data

    Parameters:
        header (dict):  original fits header
        data (str):     which dataset ('MWO','512','SPMG')
        timestamp (datetime): sample time obtained from filename

    Returns:
        header (dict):  repaired fits header
    """
    if data == 'MWO':
        t_obs = timestamp.strftime('%Y.%m.%d_%H:%M:%S')+'_TAI'
        header['CUNIT1'] = 'arcsec'
        header['CUNIT2'] = 'arcsec'
        header['CTYPE1']  = 'HPLN-TAN'                                                            
        header['CTYPE2']  = 'HPLT-TAN'
        header['CDELT1'] = header['DXB_IMG']
        header['CDELT2'] = header['DYB_IMG']
        header['CRVAL1'] = 0.0
        header['CRVAL2'] = 0.0
        header['RSUN_OBS'] = (header['R0'])*header['DXB_IMG']
        header['CROTA2'] = 0.0
        header['CRPIX1'] = header['X0']
        header['CRPIX2'] = header['Y0']
        header['T_OBS']   = t_obs
        header['DATE-OBS']   = t_obs
        header['DATE_OBS']   = t_obs
        header['RSUN_REF']= 696000000
        header['crln_obs'] = header['cenlon']

    elif data == 'SPMG' or data == '512':
        header['cunit1'] = 'arcsec'
        header['cunit2'] = 'arcsec'
        header['RSUN_OBS'] = header['EPH_R0 ']
        header['PC2_1'] = 0
        header['PC1_2'] = 0
        header['t_obs'] = header['date_obs']

    return header