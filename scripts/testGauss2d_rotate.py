## Script for fitting a rotated 2D gaussian model
## Then applied to a real ALMA data example
## Script by A. Plunkett (NRAO), aplunket@nrao.edu
## Updated Sept 2020


import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pandas as pd
from astropy.modeling import models, fitting
from spectral_cube import SpectralCube 

## https://github.com/radio-astro-tools/gaussfit_catalog/blob/master/gaussfit_catalog/core.py#L309

def gaussian(height, center_x, center_y, width_x, width_y):
    """Returns a gaussian function with the given parameters"""
    width_x = float(width_x)
    width_y = float(width_y)
    return lambda x,y: height*np.exp(
                -(((center_x-x)/width_x)**2+((center_y-y)/width_y)**2)/2)

#########
## (A) Fitting a rotated 2d gaussian model
#########


## Set up your Gaussian distribution, using the "Gaussian" function above
## Input data grid
Xin, Yin = np.mgrid[0:201, 0:201]

## The Gaussian parameters for the data are: height=3; center_x=100; center_y=100; width_x=20; width_y=40
## The second part adds some noise (if you leave this out, then the fit will be perfect)
data = gaussian(3, 100, 100, 20, 40)(Xin, Yin) + np.random.random(Xin.shape)
## You could try different parameters (i.e. center, width, height, etc.)

## Here is the model data you will use.
## I think you have to guess these...
## but even if you guess wrong, the fit might still be ok.
amp = 5
xcent = 100
ycent = 100
xwid = 20
ywid = 60
mymodel = models.Gaussian2D(amp,xcent,ycent,xwid,ywid)
print('##Inputs: amplitude = {0}, x_cent = {1}, y_cent = {2}, xwid = {3}, ywid = {4} '.format(amp,xcent,ycent,xwid,ywid))

## Choose a "fitter", in our case, called "Levenberg-Marquardt Least Squares"
## (https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm), 
## and then fit the data with your chosen model 
fitter = fitting.LevMarLSQFitter()
fitted = fitter(mymodel, Xin, Yin, data)
fitim = fitted(Xin,Yin)
print('##Model fit: amplitude = {0}, x_cent = {1}, y_cent = {2}, xwid = {3}, ywid = {4}, angle = {5} '.format(
    fitted.amplitude.value,fitted.x_mean.value,fitted.y_mean.value,fitted.x_stddev.value,fitted.y_stddev.value,fitted.theta.value))

## Calculate the "residual image" (to show how well the fit was, and what's left over)
residual = data-fitim

## Plot the input data
fig = plt.figure()

plt.clf()
ax1 = fig.add_subplot(2,2,1)
im = ax1.imshow(data, cmap='viridis', origin='lower',
                interpolation='nearest')

## Plot the Gaussian fit image

vmin, vmax = im.get_clim() ## This allows to scale to the same x, and y axes
ax2 = fig.add_subplot(2,2,2)
ax2.imshow(fitim, cmap='viridis', origin='lower',
           interpolation='nearest', vmin=vmin, vmax=vmax)

## Plot the residual 

ax3 = fig.add_subplot(2,2,3)
ax3.imshow(residual, cmap='viridis', origin='lower',
           interpolation='nearest', vmin=vmin, vmax=vmax)

## Plot the data, with the Gaussian fit as contours

ax4 = fig.add_subplot(2,2,4)
ax4.imshow(data, cmap='viridis', origin='lower',
                interpolation='nearest')

## Show model contours on top of input data
ax4.contour(fitim)
fig.show()

#########
## (B) Fit ACTUAL data.
#########

## EXAMPLE: 2015.1.00283.S
fitsimage = '/Users/aplunket/ResearchNRAO/QR/SDS/member.uid___A001_X2f6_X258.Serpens_South_sci.spw25_27_29_31_33_35.cont.I.pbcor.fits'
cube = SpectralCube.read(fitsimage)
print(cube)

## choose to use only the central region of the image
## But depending on the source, you may have to adjust this
## Note that you probably should not include any "NAN" pixels (for example, around the edge of the map.)
boxsize = 40 ## number of pixels (on a side) for the box you choose
xcent_im = int(cube.shape[2]/2) ## central pixel of the image
ycent_im = int(cube.shape[1]/2)
#obsdata = cube[0,ycent_im-int(boxsize/2):ycent_im+int(boxsize/2),xcent_im-int(boxsize/2):xcent_im+int(boxsize/2)] 
obsdata = cube[0][ycent_im-int(boxsize/2):ycent_im+int(boxsize/2),xcent_im-int(boxsize/2):xcent_im+int(boxsize/2)]
yy, xx = np.mgrid[:obsdata.shape[0], :obsdata.shape[1]]

## Here is the model data you will use.
## I think you have to guess these...
## but even if you guess wrong, the fit might still be ok.
amp = np.nanmax(obsdata) ## guess that the amplitude will be the max value
xcent = int(obsdata.shape[1]/2) ## not sure which order x,y should go in
ycent = int(obsdata.shape[0]/2)
xwid = 20 ## just a guess
ywid = 20 ## just a guess
mymodel = models.Gaussian2D(amp.value,xcent,ycent,xwid,ywid)
print('##Image cutout size: {}'.format(obsdata.shape))
print('##Inputs: amplitude = {0}, x_cent = {1}, y_cent = {2}, xwid = {3}, ywid = {4} '.format(amp,xcent,ycent,xwid,ywid))

## This part may take a while (30 seconds?)

## Choose a "fitter", in our case, called "Levenberg-Marquardt Least Squares"
## (https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm), 
## and then fit the data with your chosen model 
fitter = fitting.LevMarLSQFitter()
fitted = fitter(mymodel, xx, yy, obsdata.value)
fitim = fitted(xx,yy)
print('##Model fit: amplitude = {0}, x_cent = {1}, y_cent = {2}, xwid = {3}, ywid = {4}, angle = {5} deg '.format(
    fitted.amplitude.value,fitted.x_mean.value,fitted.y_mean.value,fitted.x_stddev.value,fitted.y_stddev.value,
    (fitted.theta.value)*180./np.pi))

## Calculate the "residual image" (to show how well the fit was, and what's left over)
residual = obsdata.value-fitim

## Plot the input data
fig2 = plt.figure()

plt.clf()
ax1 = fig2.add_subplot(2,2,1)
im = ax1.imshow(obsdata.value, cmap='viridis', origin='lower',
                interpolation='nearest')
ax1.set_title('Observational')

## Plot the Gaussian fit image
vmin, vmax = im.get_clim() ## This allows to scale to the same x, and y axes
ax2 = fig2.add_subplot(2,2,2)
ax2.imshow(fitim, cmap='viridis', origin='lower',
           interpolation='nearest', vmin=vmin, vmax=vmax)
ax2.set_title('Model fit')

## Plot the residual 

ax3 = fig2.add_subplot(2,2,3)
ax3.imshow(residual, cmap='viridis', origin='lower',
           interpolation='nearest', vmin=vmin, vmax=vmax)
ax3.set_title('Residual')

## Plot the data, with the Gaussian fit as contours

ax4 = fig2.add_subplot(2,2,4)
ax4.imshow(obsdata.value, cmap='viridis', origin='lower',
                interpolation='nearest')
ax4.set_title('Obs with model contours')

## Show model contours on top of input data
ax4.contour(fitim,cmap='plasma')

fig2.show()