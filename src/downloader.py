import datetime
import numpy as np
import os
import drms
import argparse

class Downloader:
    def __init__(self, email:str=None, sdate:str=None, edate:str=None, wavelength:int=None, instrument:str = None, cadence:str = None, format:str = None, path:str = None, downloadLimit:int = None):
        """
        Initialize a downloader class with paramaters to interface with jsoc http://jsoc.stanford.edu/

        Parameters:
            email: (str)
                JSOC registered email to enable dowloading of fits and jpg images
            sdate: (str)
                Starting date in ISO format (YYYY-MM-DD hh:mm:ss) to define the period of observations to download.
                Has to be after May 25th, 2010
            edate: (str)
                End date in ISO format (YYYY-MM-DD hh:mm:ss) to define the period of observations to download
            instrument: (str)
                Instrument to download, currently only HMI and AIA
            wavelength: (int)
                AIA wavelength of images to download, it is ignored if instrument is HMI
            cadence: (str)
                Frequency of the images within the download interval has to be a number and a string character.
                "s" for seconds, "m" for minutes, "h" for hours, and "d" for days.
            format: (str)
                Specify the file type to download, either fits or jpg
            path: (str)
                Path to download the files to (default is current directory)
            downloadLimit: (int)
                Limit the number of files to download, if None, all files will be downloaded
        """
        self.email = email
        self.sdate = datetime.date.fromisoformat(sdate)
        self.edate = datetime.date.fromisoformat(edate)
        self.instrument = instrument.lower()
        self.validinstruments = ["aia", "hmi"]
        self.wavelength = wavelength
        self.validwavelengths = [1700, 4500, 1600, 304, 171, 193, 211, 335, 94, 131]
        self.cadence = cadence
        self.validcadence = ['s', 'm', 'h', 'd']       
        self.format = format
        self.validformats = ['fits', 'jpg']
        self.path = path
        self.largeFileLimit = False
        self.downloadLimit = downloadLimit

        self.client = drms.Client(email = self.email, verbose = True)

        if not os.path.exists(self.path):
            os.mkdir(self.path)


    def assembleJsocString(self, sdate, edate):
        '''
        Given all the parameters, create the jsoc string to query the data

        Parameters:
            None

        Returns:
            jsocString
        '''
        jsocString = f"[{sdate.isoformat()}-{edate.isoformat()}@{self.cadence}]"

        if(self.instrument == 'aia'):
            if(self.wavelength in [94, 131, 171, 193, 211, 304, 335]):
                jsocString = 'aia.lev1_euv_12s' + self.jsocString + f"[{self.wavelength}]"
            elif(self.wavelength in [1600, 1700]):
                jsocString = 'aia.lev1_uv_24s' + self.jsocString + f"[{self.wavelength}]"
            elif(self.wavelength == 4500):
                jsocString = 'aia.lev1_vis_1h' + self.jsocString + f"[{self.wavelength}]"

        if(self.instrument == 'hmi'):
            jsocString = 'hmi.M_720s' + jsocString 
        
        return jsocString


    def createQueryRequest(self):
        '''
        Create a query request to get the number of files to download

        Parameters:
            None

        Returns:
            query: (panda.df)
                Dataframe with the number of files to download
                - Dates of the files
                - Number of files
        '''
        jsocString = self.assembleJsocString(self.sdate,self.edate)
        query = self.client.query(jsocString, key = 't_rec')
        return query


    def downloadData(self):
        '''
        Takes the jsoc string and downloads the data

        Parameters:
            None

        Returns:
            export_request: (panda.df)
                Dataframe with the number of files to download
        '''
        for year in range(self.sdate.year,self.edate.year+1):
            start = np.max([self.sdate,datetime.date(year,1,1)])
            end = np.min([self.edate,datetime.date(year+1,1,1)])
            jsocString = self.assembleJsocString(start,end)
            export_request = self.client.export(jsocString, protocol = self.format, method='url')

            out_path = os.path.join(self.path,str(year))
            if not os.path.exists(out_path):
                os.mkdir(out_path)
            self.export = export_request.download(out_path, 0)

        return self.export



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
    parser.add_argument('--email',
                        type=str,
                        help='JSOC registered email to enable dowloading of fits and jpg images'
                        )
    parser.add_argument('-sd','--sdate',
                        type=str,
                        help='Starting date in ISO format (YYYY-MM-DD hh:mm:ss) to define the period of observations to download. Has to be after May 25th, 2010'
                        )
    parser.add_argument('-ed','--edate',
                        type=str,
                        help='End date in ISO format (YYYY-MM-DD hh:mm:ss) to define the period of observations to download'
                        )    
    parser.add_argument('-i','--instrument',
                        type=str,
                        help='Instrument to download, currently only HMI and AIA'
                        )    
    parser.add_argument('-wl','--wavelength',
                        type=int,
                        help='AIA wavelength of images to download, it is ignored if instrument is HMI'
                        )    
    parser.add_argument('-c','--cadence',
                        type=str,
                        help='Frequency of the images within the download interval has to be a number and a string character. "s" for seconds, "m" for minutes, "h" for hours, and "d" for days.'
                        )  
    parser.add_argument('-f','--format',
                        type=str,
                        help='Specify the file type to download, either fits or jpg'
                        )  
    parser.add_argument('-p','--path',
                        type=str,
                        help='Path to download the files to (default is current directory)'
                        )  
    parser.add_argument('-dlim','--downloadLimit',
                        type=int,
                        default = 1000,
                        help='Limit the number of files to download, defaults to 1000'
                        )  

    return parser.parse_args(args)


if __name__=="__main__":

    parser = parse_args()
    downloader = Downloader(parser.email, 
                            parser.sdate, 
                            parser.edate, 
                            parser.wavelength, 
                            parser.instrument, 
                            parser.cadence, 
                            parser.format, 
                            parser.path, 
                            parser.downloadLimit)
    
    downloader.assembleJsocString(downloader.sdate, downloader.edate)
    downloader.downloadData()

