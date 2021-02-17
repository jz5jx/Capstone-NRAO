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


def pb_multiply(im_array, pb_path):
    '''Function to multiply a FITS image by a .pb file to deemphasize edges

    Inputs
    ------
    im_array: 2d array representing a FITS image data
    pb_path: str indicating file location of corresponding .pb file

    Returns
    -------
    new_imdata: np array of same size as image data array
        consisting of elementwise multiplication of image and pb file
    '''

    pb_fits = fits.open(pb_path)
    pb = pb_fits[0].data.squeeze()

    new_imdata = np.multiply(im_array, pb)

    return new_imdata

def get_noise_level(fits_image, nchunks = 3, rms_quantile = 0.5):
    '''Calculates estimated noise level in image intensity
    Stores value in FitsImage object noise attribute

    Arguments
    ---------
    nchunks: int number of chunks to use in grid, must be odd
    rms_quantile: float in range [0, 1] indicating quantile of chunk RMS to use for noise level (0 = min RMS, 0.5 = median, etc)

    Returns
    -------
    noise: float estimated noise in image intensity values
    '''

    id1 = np.argwhere(np.isnan(fits_image.imdata))[:,0]
    id2 = np.argwhere(np.isnan(fits_image.imdata))[:,1]

    imdata = fits_image.imdata.copy()

    imdata = np.delete(imdata, id1, axis=0)
    imdata = np.delete(imdata, id2, axis=1)

    #now break the image into chunks and do the same analysis;
    # one of the chunks should have no signal in and give you an estimate of the noise (= rms).# number of chunks in each direction:
    # an odd value is used so that the centre of the image does not correspond to the edge of chunks;
    # when you ask for observations with ALMA, you usually specify that the object of interest be in the
    # center of your image.
    size = [i//nchunks for i in imdata.shape]
    remain = [i % nchunks for i in imdata.shape]
    chunks = dict()
    k = 0
    for j,i in itertools.product(range(nchunks),range(nchunks)):
        chunks[k] = size.copy()
        k += 1# next, account for when the image dimensions are not evenly divisible by `nchunks`.
    row_remain, column_remain = 0, 0
    for k in chunks:
        if k % nchunks < remain[0]:
            row_remain = 1
        if k // nchunks < remain[1]:
            column_remain = 1
        if row_remain > 0:
            chunks[k][0] += 1
            row_remain -= 1
        if column_remain > 0:
            chunks[k][1] += 1
            column_remain -= 1# with that in hand, calculate the lower left corner indices of each chunk
    indices = dict()
    for k in chunks:
        indices[k] = chunks[k].copy()
        if k % nchunks == 0:
            indices[k][0] = 0
        elif k % nchunks != 0:
            indices[k][0] = indices[k-1][0] + chunks[k][0]
        if k >= nchunks:
            indices[k][1] = indices[k-nchunks][1] + chunks[k][1]
        else:
            indices[k][1] = 0
    stddev_chunk = dict()
    rms_chunk = dict()
    for k in chunks:
        i,j = indices[k]
        di,dj = chunks[k]
        x = imdata[i:i+di,j:j+dj]
        stddev_this = np.nanstd(x)
        rms_this = np.sqrt(np.nanmean(x**2))
        stddev_chunk[k] = stddev_this
        rms_chunk[k] = rms_this

    noise = np.quantile(list(noise_all.values()), q = rms_quantile)
    fits_image.noise = noise
    return(noise)
