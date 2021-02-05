import numpy as np

def circle_crop(fits_image, rad_factor = 1.0):
    rad = fits_image.imdata.shape[0]/2
    rad_sq = rad**2

    for ix,iy in np.ndindex(fits_image.imdata.shape):
        if (ix - rad)**2 + (iy - rad)**2 > rad_sq:
            fits1.imdata[ix, iy] = np.nan
