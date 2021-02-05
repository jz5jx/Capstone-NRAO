import numpy as np

def circle_crop(fits_image, rad_factor = 1.0):
'''Function to crop square images to a circle

Inputs
------
fits_image: an object of class FitsImage

Params
------
rad_factor: float multiple allowing change to size of circle_crop
    default is 1
    value equal to 0.7 crops to a circle with radius that is 70% as large as the max image radius
    values < 0 not allowed
    values >= sqrt(2) will return original image

Returns
-------
new_imdata: np array of same size as image data array, but with values outside radius set to nan
'''

    if rad_factor < 0:
        raise ValueError('rad_factor must be >= 0')

    rad = fits_image.imdata.shape[0]/2
    rad_sq = (rad*rad_factor)**2

    new_imdata = fits_image.imdata.copy()

    for ix,iy in np.ndindex(new_imdata.shape):
        if (ix - rad)**2 + (iy - rad)**2 > rad_sq:
            new_imdata[ix, iy] = np.nan

    return new_imdata
