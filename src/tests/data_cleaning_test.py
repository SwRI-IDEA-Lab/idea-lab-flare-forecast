import sys,os
sys.path.append(os.getcwd())

import unittest
import astropy.units as u
from astropy.io import fits
import sunpy
from src.utils.utils import reprojectToVirtualInstrument


class DataCleaningTest(unittest.TestCase):

    def setUp(self):
        root = 'Data/MDI/2010'
        files = os.listdir('Data/MDI/2010')
        i = 0
        with fits.open(root+os.sep+files[i],cache=False) as fits_file:
            fits_file.verify('fix')
            self.map = sunpy.map.Map(fits_file[1].data,fits_file[1].header)
        self.radius = 1*u.au
        self.scale = u.Quantity([0.6,0.6],u.arcsec/u.pixel)
        self.dim = 512

    def test_reprojectIsMap(self):
        out_map = reprojectToVirtualInstrument(self.map)
        self.assertIsInstance(out_map,sunpy.map.GenericMap)
    
    def test_reprojectRadius(self):
        out_map = reprojectToVirtualInstrument(self.map,radius=self.radius,scale=self.scale)
        self.assertAlmostEqual(out_map.meta['dsun_obs'],self.radius.to(u.m).value,places=7)

    def test_reprojectScale(self):
        out_map = reprojectToVirtualInstrument(self.map,radius=self.radius,scale=self.scale)
        self.assertAlmostEqual(out_map.meta['cdelt1'],self.scale[0].to(u.deg/u.pixel).value,places=7)

    def test_reprojectDim(self):
        out_map = reprojectToVirtualInstrument(self.map,dim=self.dim,radius=self.radius,scale=self.scale)
        self.assertEqual(out_map.data.shape[0],self.dim)
        self.assertEqual(out_map.data.shape[1],self.dim)


if __name__ == "__main__":
    unittest.main()
