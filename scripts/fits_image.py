"""Class for working with FITS images
"""

class fits_image:

    def __init__(self, file_path):
        self.fits = fits.open(file_path)
        self.image = self.fits[0]
        self.imdata = self.image.data.squeeze()

    def plot_image(self, colorbar = True):
        plt.imshow(self.imdata, origin='lower')
        if colorbar:
            plt.colorbar()

    def fit_bivariate_gaussian(self, noise_thresh):
        X = range(len(self.imdata[0]))
        Y = range(len(self.imdata))
        Z = self.imdata
        Z = np.nan_to_num(Z,  0)

        Z[Z<(noise_thresh)] = 0  # remove noise in figure
        X, Y = np.meshgrid(X, Y)
        pos = np.dstack((X, Y))
        x1_bar = np.average(X, weights=Z)      #Weighted X mean
        x2_bar = np.average(Y, weights=Z)      #Weighted Y mean
        x1_var = np.average((X-x1_bar)**2, weights=Z)  #Weighted X variance
        x2_var = np.average((Y-x2_bar)**2, weights=Z)  #Weighted Y variance
        x_cov = np.average(X*Y, weights=Z)-x1_bar*x2_bar #Weighted Covariance

        cov_mat = np.array([[x1_var, x_cov],[x_cov, x2_var]]) #Covariance matrix

        self.gaussian_mean = [x1_bar, x2_bar]
        self.gaussian_cov = cov_mat

        return([x1_bar, x2_bar, cov_mat])

        rv = stats.multivariate_normal([x1_bar, x2_bar], cov_mat) #Fitting a new Gaussian Normal with our parameters.

#TODO: finish plot fit function
    # def plot_gaussian_fit(self):


    def get_noise_level(self, nchunks = 3, rms_quantile = 0):
        '''
        nchunks = number of chunks for grid, must be odd
        rms_quantile = quantile of chunk RMS to use for noise level (0 = min RMS, 0.5 = median, etc)
        '''

        id1 = np.argwhere(np.isnan(self.imdata))[:,0]
        id2 = np.argwhere(np.isnan(self.imdata))[:,1]

        imdata = self.imdata.copy()

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
        self.noise = noise
        return(noise)
