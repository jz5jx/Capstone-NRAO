"""Class for working with FITS images
Methods for plotting and fitting bivariate Gaussians to images
"""

from astropy.io import fits
import astropy.io
from astropy.table import Table

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools

class FitsImage:

    def __init__(self, file_path):
        '''Reads in FITS image from file and stores intensity values in object

        Arguments
        ---------
        file_path: str indicating location of FITS file'''

        self.fits = fits.open(file_path)
        self.image = self.fits[0]
        self.imdata = self.image.data.squeeze()

    def plot_image(self, colorbar = True):
        '''Plots FITS image intensity levels

        Arguments
        ---------
        colorbar: boolean indicating whether or not to include a colorbar scale'''

        plt.imshow(self.imdata, origin='lower')
        if colorbar:
            plt.colorbar()

    def fit_bivariate_gaussian(self, noise_thresh):
        '''Fits a single bivariate Gaussian to a FITS image

        Arguments
        ---------
        noise_thresh: float value of estimated noise intensity; all values below this level will be truncated

        Returns
        -------
        [x1_bar, x2_bar, cov_mat]: array containing the means and covariance matrix of the fitted Gaussian
        '''

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
