"""
Script to download GOES data from Fido
"""

import astropy.units as u
from astropy.table import QTable
from datetime import datetime,timedelta
import os
import pandas as pd
from sunpy import timeseries as ts
from sunpy.net import Fido
from sunpy.net import attrs as a
from sunpy.net.base_client import QueryResponseTable
import sys

root = sys.argv[1]

for year in range(1980,2023):
    if not os.path.exists(root+str(year)):
        os.mkdir(root+str(year))

    # create query
    tstart=datetime(year,1,1)
    tend=datetime(year,12,31)
    result = Fido.search(a.Time(datetime.strftime(tstart,'%Y-%m-%d %H:%M'),
                                datetime.strftime(tend,'%Y-%m-%d %H:%M')),
                                a.Instrument('XRS'))
    
    if len(result[0])==0:   # no data found
        continue
    
    # filter only latest GOES satellite data
    df_result = result[0].to_pandas()
    df_result.sort_values(['Start Time','SatelliteNumber'],inplace=True)
    df_result.drop_duplicates('Start Time',keep='last',inplace=True)
    # remake into query 
    result_table = QTable.from_pandas(df_result)
    result_new = QueryResponseTable(result_table)
    result_new.client = result[0].client

    # download data
    file_goes = Fido.fetch(result_new,path=root+str(year))

