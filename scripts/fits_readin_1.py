# -*- coding: utf-8 -*-
"""
Basic fits file readin
Created on Tue Sep 15 01:38:41 2020

@author: Tiger
"""

import astropy.io
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from astropy.table import Table
import astropy.io

fits1 = astropy.io.fits.open('member.uid___A001_X133d_X603.MACSJ0717-11_sci.spw5_7_9_11.cont.I.pbcor.fits')

fits1.info()

image = fits1[0]

#plt.imshow(image.data[0,0,:,:], origin='lower')

plt.imshow(image.data[0,0,:,:], cmap = 'gray')
plt.imshow(image.data[0,0,:,:])
plt.colorbar()

evt_data = Table(fits1[0].data)

print(evt_data)

print('Min:', np.nanmin(image.data[0,0,:,:]))
print('Max:', np.nanmax(image.data[0,0,:,:]))
print('Mean:', np.nanmean(image.data[0,0,:,:]))
print('Stdev:', np.nanstd(image.data[0,0,:,:]))


xsize = image.header['NAXIS1']
xcentral = xsize//2
xcentral

'''
NAXIS - dimensions and axis, focus on the central pixal first

BMAJ - major axis of the beam 'major axis of the beam' | usually in arcsec = degree/3600 | degree for this case
BMIN - min ---
BPA - position angle of the beam
resolution elements 
always elipse - don't trust anything smaller than bean size'
pixal 3-5 size smaller than beam size 

OBJECT - name of the target, named by individual astromers, no necessarily same with other

BUNIT - Jy - janskys? | intensity depends on the beam size | K is more universal

RADESYS - RA and dec , right asscesion, dec 

CTYPE1 - coordinate type of axis 1
CRVAL1 - value of degree for RA | refer to exact location on the astronomical map
CDELT1 - size of the pixal | minus - right to left
CRPIX1 - which pixal is assigned

RESTFRQ - rest freq that was set at 

SPECMODE - always look at 'cont' data

pbcor - primary beam corrected, center more sensitive, bring up signals at the edge
'''

