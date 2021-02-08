import unittest
from astropy.io import fits
import astropy.io
import numpy as np
import matplotlib
from astropy.table import Table
import astropy.io
from scipy import stats
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from astroML.stats import fit_bivariate_normal
from astroML.stats.random import bivariate_normal
import numpy as np
import pandas as pd

class TestParameters(unittest.TestCase):
    pathway = input('what is your pathway to fits files?') #/Users/johnzhang/Desktop/MSDS_MASTERS/Spring/Capstone-NRAO/work/
    filename = input('what is your file name?') #test_4.fits
    fits1 = fits.open(pathway + filename)
    image = fits1[0]
    
    def test_weighted_X(self):
        X = range(len(self.image.data[0,0,:,:][0])) 
        self.assertIsNotNone(X) #Makes sure X is not Null
        
    def test_weighted_Y(self):
        Y = range(len(self.image.data[0,0,:,:]))
        self.assertIsNotNone(Y) #Makes sure Y is not Null
    
    def test_weighted_Z(self):
        Z = self.image.data[0,0,:,:]
        Z = np.nan_to_num(Z,  0) 
        np.testing.assert_equal(Z,Z) #Tests to see if NP array is equal to itself
        self.assertIsNotNone(Z)
        
    def test_xbar(self): #weighted X
        X = range(len(self.image.data[0,0,:,:][0]))
        Y = range(len(self.image.data[0,0,:,:]))
        X, Y = np.meshgrid(X, Y)
        Z = self.image.data[0,0,:,:]
        Z = np.nan_to_num(Z,  0) 
        x1_bar = np.average(X, weights=Z) 
        self.assertIsNotNone(x1_bar)
        
    def test_ybar(self): #weighted X
        X = range(len(self.image.data[0,0,:,:][0]))
        Y = range(len(self.image.data[0,0,:,:]))
        X, Y = np.meshgrid(X, Y)
        Z = self.image.data[0,0,:,:]
        Z = np.nan_to_num(Z,  0) 
        x2_bar = np.average(Y, weights=Z)  
        self.assertIsNotNone(x2_bar)
        
    def test_plot(self):
        plt.imshow(self.image.data[0,0,:,:], origin='lower')
        plt.colorbar()
        self.assertIsNotNone(plt.imshow(self.image.data[0,0,:,:], origin='lower'))