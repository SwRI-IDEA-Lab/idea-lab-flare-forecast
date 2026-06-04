# %%
import astropy.units as u

from sunpy.net import Fido
from sunpy.net import attrs as a

from astropy.io import fits

import matplotlib.pyplot as plt

# %%
data_dir = '/d0/jkobayashi/data/gong/manual/magnetograms'
hdu_list = fits.open(data_dir+'/zqa/bbzqa260420t0024.fits.gz')
hdu_list.info()

# %%
image_data = hdu_list[0].data
print(image_data.shape)


hdu_list.close()

# %%
plt.imshow(image_data, cmap='gray')

# %%
# plt.figure(figsize=(20,20))
plt.imshow(image_data[::-1,::], cmap='gray')
# %%
