import sys,os
sys.path.append(os.getcwd())

import unittest
import astropy.units as u
from astropy.io import fits
import numpy as np
import sunpy
from src.utils.utils import reprojectToVirtualInstrument, mapPixelArea, zeroLimbs
from src.data_preprocessing.helper import compute_tot_flux


class DataCleaningTest(unittest.TestCase):

    def setUp(self):
        root = 'Data/MDI/2011'
        files = os.listdir(root)
        i = 0
        with fits.open(root+os.sep+files[i],cache=False) as fits_file:
            fits_file.verify('fix')
            self.map = sunpy.map.Map(fits_file[1].data,fits_file[1].header)
        self.radius = 1*u.au
        self.scale = 4*u.Quantity([0.55,0.55],u.arcsec/u.pixel)
        self.dim = 1024
        self.zeroradius = 0.95

    def test_reprojectIsMap(self):
        out_map = reprojectToVirtualInstrument(self.map)
        self.assertIsInstance(out_map,sunpy.map.GenericMap)
    
    def test_reprojectRadiusScale(self):
        out_map = reprojectToVirtualInstrument(self.map,radius=self.radius,scale=self.scale)
        self.assertAlmostEqual(out_map.meta['dsun_obs'],self.radius.to(u.m).value,places=7)
        self.assertAlmostEqual(out_map.meta['cdelt1'],self.scale[0].to(u.deg/u.pixel).value,places=7)

    def test_reprojectDim(self):
        out_map = reprojectToVirtualInstrument(self.map,dim=self.dim,radius=self.radius,scale=self.scale)
        self.assertEqual(out_map.data.shape[0],self.dim)
        self.assertEqual(out_map.data.shape[1],self.dim)

    def test_computeFlux(self):
        tot_usflux = compute_tot_flux(self.map,signed=False)
        self.assertIsInstance(tot_usflux,float)
        print('Total unsigned flux:',tot_usflux)
        self.assertGreater(tot_usflux,np.max(self.map.data[~np.isnan(self.map.data)]))
        self.assertLess(tot_usflux,np.nansum(np.abs(self.map.data))*np.pi*(self.map.meta['rsun_ref']/1e6*2/(0.75*self.map.meta['naxis1']))**2)

    def test_computeFluxReproject(self):
        zeroed_map = zeroLimbs(self.map,radius=self.zeroradius)
        tot_flux = compute_tot_flux(zeroed_map,signed=True)
        out_map = reprojectToVirtualInstrument(self.map,dim=self.dim,radius=self.radius,scale=self.scale)
        zeroed_out_map = zeroLimbs(out_map,radius=self.zeroradius)
        out_tot_flux = compute_tot_flux(zeroed_out_map,signed=True)
        print('Total signed flux:',tot_flux)
        print('After reprojection:',out_tot_flux)
        self.assertAlmostEqual(tot_flux,out_tot_flux,delta=0.1*np.abs(tot_flux))

    def test_computeUSFluxReproject(self):
        zeroed_map = zeroLimbs(self.map,radius=self.zeroradius)
        tot_usflux = compute_tot_flux(zeroed_map,signed=False)
        out_map = reprojectToVirtualInstrument(self.map,dim=self.dim,radius=self.radius,scale=self.scale)
        zeroed_out_map = zeroLimbs(out_map,radius=self.zeroradius)
        out_tot_usflux = compute_tot_flux(zeroed_out_map,signed=False)
        print('Total unsigned flux:',tot_usflux)
        print('After reprojection:',out_tot_usflux)
        self.assertAlmostEqual(tot_usflux,out_tot_usflux,delta=0.3*np.abs(tot_usflux))

    def test_mapPixelArea(self):
        area = mapPixelArea(self.map)
        sum_area = np.nansum(area.data)
        out_map = reprojectToVirtualInstrument(self.map,dim=self.dim,radius=self.radius,scale=self.scale)
        out_area = mapPixelArea(out_map)
        sum_out_area = np.nansum(out_area.data)
        print(sum_area,sum_out_area)
        self.assertAlmostEqual(sum_area,sum_out_area,delta=0.1*sum_area)

    def test_zeroLimbExists(self):
        map = zeroLimbs(self.map)
        self.assertIsInstance(map,sunpy.map.GenericMap)

    def test_zeroLimb(self):
        map = zeroLimbs(self.map,radius=0,fill_value=0)
        self.assertEqual(np.nansum(map.data),0)

if __name__ == "__main__":
    unittest.main()
