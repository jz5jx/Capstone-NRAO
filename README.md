NRAO Capstone


  Data Column Headings:
  
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
