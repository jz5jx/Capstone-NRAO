

import scipy.optimize as opt
import numpy as np
import pylab as plt
import astropy.io
from astropy.table import Table

#define model function and pass independant variables x and y as a list
def twoD_Gaussian(xdata_tuple, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    (x, y) = xdata_tuple                                                        
    xo = float(xo)                                                              
    yo = float(yo)                                                              
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)   
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)    
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)   
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo)         
                        + c*((y-yo)**2)))                                   
    return g.ravel()


fits1 = astropy.io.fits.open('test_2.fits')
image1 = fits1[0]
tab1 = image1.data[0,0,:,:]
data = tab1

ypix, xpix = tab1.shape
x = np.linspace(0, xpix, xpix)
y = np.linspace(0, ypix, ypix)
x, y = np.meshgrid(x, y)

# plot twoD_Gaussian data generated above
plt.figure()
plt.imshow(data)
plt.colorbar()

# start fitting, create initial guess
initial_guess = (np.max(tab1), xpix//2, ypix//2, 20, 40, 8, 0)
# add noise to data
#data_noisy = data + 0.2*np.random.normal(size=data.shape)
# get parameters with opt.curve_fit
popt, pcov = opt.curve_fit(twoD_Gaussian, (x, y), data.ravel(), p0=initial_guess)

data_fitted = twoD_Gaussian((x, y), *popt)

fig, ax = plt.subplots(1, 1)
ax.imshow(data.reshape(ypix, xpix), cmap=plt.cm.jet, origin='lower',
    extent=(x.min(), x.max(), y.min(), y.max()))
ax.contour(x, y, data_fitted.reshape(ypix, xpix), 2, colors='red', linewidths = 1)
plt.show()

rmse = np.sqrt(((data_fitted - tab1.ravel())**2).sum()/len(tab1.ravel()))

plt.imshow(data_fitted.reshape(ypix, xpix))














