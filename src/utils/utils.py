from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np
from skimage.morphology import dilation, area_opening, disk
from scipy.interpolate import griddata
from sunpy.coordinates.frames import HeliographicStonyhurst
from sunpy.map import Map, make_fitswcs_header

def latLonRemap(inputmap, refMap=None, dlat=None, dlon=None, method='linear'):
    """Function to reproject a sunpy map into a regular latitude-longitude grid

    Parameters
    ----------
    inputmap : sunpy map
        Map to reproject
    refMap : sunpy map
        Map that will be used to specify the latitudinal
        and longitudinal extend of the reprojection
    dlat : u.deg
        Latitudinal grid size in u.deg (degrees)
    dlon : u.deg
        Longitudinal grid size in u.deg (degrees)
    method : string
        Method to use in the griddata interpolation: 'linear', 'cubic'

    Returns
    -------
    lat: numpy array in u.deg
        Grid of latitudes associated with the reprojected data
    lon: numpy array in u.deg
        Grid of longitudes associated with the reprojected data
    lotLanData: numpy array
        Interpolated data
    """  

    y, x = np.meshgrid(*[np.arange(v) for v in inputmap.data.shape])* u.pixel
    latlon = inputmap.pixel_to_world(x, y).transform_to(HeliographicStonyhurst)

    # turn latitude into colatitude
    colat = (90*u.deg - latlon.data.lat.to(u.deg))

    # Modify longitude to be continuous in visible hemisphere
    lon_shift = latlon.data.lon.to(u.deg).value
    lon_shift[lon_shift>180] = lon_shift[lon_shift>180]-360
    lon_shift = lon_shift*u.deg

    # Find latitude and longitude limits
    if refMap is None:

        colat1 = 90*u.deg - np.nanmax(latlon.lat.to(u.deg))
        colat2 = 90*u.deg - np.nanmin(latlon.lat.to(u.deg))

        colatLims = np.array([colat1.to(u.deg).value, colat2.to(u.deg).value])
        colatLims = np.array([np.min(colatLims), np.max(colatLims)])*u.deg

        lon1 = np.nanmax(latlon.lon.to(u.deg)).value
        if lon1>180:
            lon1 -= 360

        lon2 = np.nanmin(latlon.lon.to(u.deg)).value
        if lon2>180:
            lon2 -= 360

        lonLims = np.array([lon1, lon2])
        lonLims = np.array([np.min(lonLims), np.max(lonLims)])*u.deg

    else:
        bottom_left = refMap.bottom_left_coord.transform_to(HeliographicStonyhurst)
        top_right = refMap.top_right_coord.transform_to(HeliographicStonyhurst)        

        colat1 = 90*u.deg - top_right.data.lat.to(u.deg)
        colat2 = 90*u.deg - bottom_left.data.lat.to(u.deg)

        colatLims = np.array([colat1.to(u.deg).value, colat2.to(u.deg).value])
        colatLims = np.array([np.min(colatLims), np.max(colatLims)])*u.deg

        lon1 = bottom_left.data.lon.to(u.deg).value
        if lon1>180:
            lon1 -= 360

        lon2 = top_right.data.lon.to(u.deg).value
        if lon2>180:
            lon2 -= 360

        lonLims = np.array([lon1, lon2])
        lonLims = np.array([np.min(lonLims), np.max(lonLims)])*u.deg


    # Calculate median latitude and longitude differentials
    mask = colat>=colatLims[0]
    mask = np.logical_and(mask, colat<=colatLims[1])
    mask = np.logical_and(mask, lon_shift>=lonLims[0])
    mask = np.logical_and(mask, lon_shift<=lonLims[1])

    # Caclulate median differentials if not provided
    if dlat is None:
        axis = 1
        tmp_diff = np.abs(np.diff(latlon.data.lat, axis=axis))
        dlat = np.nanmedian(tmp_diff[mask[:,0:-1]].reshape(-1).to(u.deg))

    if dlon is None:
        axis = 0
        tmp_diff = np.abs(np.diff(lon_shift, axis=axis))    
        dlon = np.nanmedian(tmp_diff[mask[0:-1,:]].reshape(-1).to(u.deg))


    # Create uniform grids
    n_lat = int(np.round(np.abs(colatLims[1]-colatLims[0])/dlat))
    n_lon = int(np.round(np.abs(lonLims[1]-lonLims[0])/dlon))

    colat_uniform = np.linspace(colatLims[0],colatLims[1],n_lat)
    lon_uniform = np.linspace(lonLims[0],lonLims[1],n_lon)

    colatGrid, lonGrid = np.meshgrid(colat_uniform, lon_uniform)

    # Reshape and clean input data
    colat = colat.reshape(-1)
    lon_shift = lon_shift.reshape(-1)
    zBfield = inputmap.data.T.reshape(-1)

    lon_shift = lon_shift[np.isfinite(colat)]
    zBfield = zBfield[np.isfinite(colat)]
    colat = colat[np.isfinite(colat)]

    colat = colat[np.isfinite(lon_shift)]
    zBfield = zBfield[np.isfinite(lon_shift)]
    lon_shift = lon_shift[np.isfinite(lon_shift)]

    colat = colat[np.isfinite(zBfield)]
    lon_shift = lon_shift[np.isfinite(zBfield)]
    zBfield = zBfield[np.isfinite(zBfield)]

    lotLanData = griddata((lon_shift.value, colat.value), zBfield, (lonGrid.value, colatGrid.value), method=method)

    return lonGrid.T, colatGrid.T, lotLanData.T



def mapCrop(inputmap, crop_map, pixelPadding=0):
    """Wrapper function to fix a problem with Sunpy's submap routine which
       crashes when making a submap based in corner coordinates coming from another
       map. 

    Parameters
    ----------
    inputmap : sunpy map
        Map with the larger field of view (FOV)
    crop_map : sunpy map
        Map with the smaller FOV that will be used to crop inputmap
    pixelPadding : int
        Extra padding to add to the FOV to enable lat-lon interpolations with maximum
        coverage

    Returns
    -------
    cropped_map: Sunpy map
        Map containing the cropped region of the inputmap to the FOV of the crop_map.
    """    
    
    bottom_left = inputmap.world_to_pixel(crop_map.bottom_left_coord)
    bottom_left = [np.nanmax([bottom_left.x.value-pixelPadding, 0]), np.nanmax([bottom_left.y.value-pixelPadding, 0])]*u.pixel
    top_right = inputmap.world_to_pixel(crop_map.top_right_coord)
    top_right = [np.nanmin([top_right.x.value+pixelPadding, inputmap.data.shape[0]-1]), np.nanmin([top_right.y.value+pixelPadding, inputmap.data.shape[1]-1])]*u.pixel
    
    return inputmap.submap(bottom_left, top_right=top_right)



def mapPixelArea(map):
    """Function to calculate each pixel's photospheric area for a given map. 

    Parameters
    ----------
    map : sunpy map
        Map containing observations

    Returns
    -------
    area : sunpy map
        Map containing the area of each pixel.
    """
    
    x, y = np.meshgrid(*[np.arange(v) for v in map.data.shape])* u.pixel

    # Calculate position of three of the pixel's corners
    a = map.pixel_to_world(x+0.5*u.pixel, y+0.5*u.pixel).transform_to(HeliographicStonyhurst)
    b = map.pixel_to_world(x+0.5*u.pixel, y-0.5*u.pixel).transform_to(HeliographicStonyhurst)
    c = map.pixel_to_world(x-0.5*u.pixel, y-0.5*u.pixel).transform_to(HeliographicStonyhurst)
    d = map.pixel_to_world(x-0.5*u.pixel, y+0.5*u.pixel).transform_to(HeliographicStonyhurst)

    ## Stacking latitudes and longitudes

    lats = np.concatenate((a.data.lat[:,:,None], b.data.lat[:,:,None], c.data.lat[:,:,None], d.data.lat[:,:,None], a.data.lat[:,:,None]), axis=2)
    lons = np.concatenate((a.data.lon[:,:,None], b.data.lon[:,:,None], c.data.lon[:,:,None], d.data.lon[:,:,None], a.data.lon[:,:,None]), axis=2)


    # Get colatitude (a measure of surface distance as an angle)
    an = np.sin(lats/2)**2 + np.cos(lats)* np.sin(lons/2)**2
    colat = 2*np.arctan2( np.sqrt(an), np.sqrt(1-an) )

    #azimuth of each point in segment from the arbitrary origin
    az = np.arctan2(np.cos(lats) * np.sin(lons), np.sin(lats)) % (2*np.pi*u.rad)

    # Calculate step sizes
    daz = np.diff(az, axis=2)
    daz = (daz + np.pi*u.rad) % (2 * np.pi*u.rad) - np.pi*u.rad

    # Determine average surface distance for each step
    deltas=np.diff(colat, axis=2)/2
    colat=colat[:,:,0:-1]+deltas

    # Integral over azimuth is 1-cos(colatitudes)
    integrands = (1-np.cos(colat)) * daz

    # Integrate and save the answer as a fraction of the unit sphere.
    # Note that the sum of the integrands will include a factor of 4pi.
    area = np.abs(np.nansum(integrands, axis=2))/(4*np.pi*u.rad) # Could be area of inside or outside
    area = np.concatenate((area[:,:,None], 1-area[:,:,None]), axis=2)
    area = np.min(area, axis=2)

    # Convert fractional area to real area
    # # Solar Radius
    rsun = (map.meta['RSUN_REF']*u.m).to(u.km)

    area = 4*np.pi*(area*rsun*rsun).to(u.Mm*u.Mm)
    return Map(area, map.meta)



def makeBMask(data, 
            Blim=30, 
            area_threshold=128,
            connectivity=2,
            dilationR=8):
    """Function to greate mask surrounding strong fields.

    Parameters
    ----------
    data : numpy array
        data used to create the mask
    Blim : float
        Magnetic field theshold used to determine the mask kernels
    area_threshold : int
        area_threshold passed to area_opening operation
    connectivity : int
        (1) for using only immediate neighbors (vertical and horizontal).
        (2) for using also diagonals
    dilationR : 8
        Radius of dilation disk

    Returns
    -------
    BMask : sunpy map
        Map containing the Bmask.
    """

    fieldMask = np.abs(data)>Blim           
    step1 = area_opening(fieldMask, area_threshold=area_threshold, connectivity=connectivity)
    footprint = disk(dilationR)
    mask = dilation(step1, footprint).astype(np.float)

    return mask



def sphericalGrad(colat, lon, data, rsun=(695700*u.km).to(u.Mm)):
    """Function to calculate the spherical gradient of one of the colat-lon regrids
       https://en.wikipedia.org/wiki/Del_in_cylindrical_and_spherical_coordinates

    Parameters
    ----------
    colat : numpy array
        Uniform colatitude grid in a meshgrid. 
    lon : numpy array
        Uniform longitude grid in a meshgrid. 
    data : numpy array
        Magnetic field reprojected into a uniform colat-lon grid
    rsun : astropy.u
        Solar radius using astropy units 

    Returns
    -------
    dBdT : numpy array
        Latitudinal gradient.
    dBdP : numpy array
        Longitudinal gradient.
    gradMag : numpy array
        Magnitude of the gradient.
    """


    gradient = np.gradient(data, colat[:,0], lon[0,:])

    dBdT = gradient[0]/rsun*u.deg
    dBdP = gradient[1]/np.sin(colat.to(u.rad))/rsun*u.deg
    gradMag = np.sqrt(dBdT**2 + dBdP**2)

    return dBdT, dBdP, gradMag  

def reprojectToVirtualInstrument(map,
                                 dim=None,
                                 radius=1*u.au,
                                 scale=u.Quantity([0.6,0.6],u.arcsec/u.pixel)):
    """
    Reproject map to an instrument at specified radius and plate scale
    
    Parameters
    ----------
    map : sunpy map
        Map containing observations
    dim : int
        Desired resolution of reprojected map
    radius: u.au
        Distance to sun of desired virtual observation
    scale: [,] u.arcsec/u.pixel
        Virtual observation pixel scale 
        
    Returns
    -------
    out_map : sunpy map
        Reprojected map
    """
    if dim == None:
        # keep at the original resolution
        dim = map.meta['naxis1']

    map_observer = map.reference_coordinate.frame.observer
    sc = SkyCoord(
        0*u.arcsec,0*u.arcsec,
        frame='helioprojective',
        rsun= map.reference_coordinate.rsun,
        obstime=map_observer.obstime,
        observer = HeliographicStonyhurst(map_observer.lon,
                                          map_observer.lat,
                                          radius,
                                          obstime=map_observer.obstime,
                                          rsun=map_observer.rsun)
    )
    out_header = make_fitswcs_header((dim,dim),sc,scale=scale)
    out_map = map.reproject_to(out_header)   
    return out_map