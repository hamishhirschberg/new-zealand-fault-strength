#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Hamish Hirschberg
"""

def strainStyleMag(model, style, stylese, soln=0, minmag=0., comp=0,
                   obrange=None):
    """Adjust strain rate observations based on style.
    
    This function adjusts strain rate observations and their standard
    errors to match the observed strain style based on a given strain
    rate magnitude. The magnitude is taken from the magnitude of strain
    rate in the previous iteration.
    
    model : permdefmap model
        Model where the observations are being adjusted.
    style : array of float
        Strain rate style in elements.
    stylese : array of float
        Standard error of strain rate style.
    soln : 0 (default) or 1
        Determines whether the a priori (0) or a posteriori (1)
        solution is used for strain rate magnitude.
    minmag : float, default=0.
        Elements with a strain rate magnitude smaller than minmag will
        use minmag instead when calculating standard errors for their
        observations.
    comp : int, default=0
        Index of the observation component that constrains strain rate
        style.
    obrange : iterable
        Indices of observations to be adjusted. Default is all
        observations.
    """
    import numpy as np
    
    if obrange is None:
        obrange = range(model.nstrainob)
    
    if str(soln) == '0' or soln == 'apri' or soln == 'apriori':
        # Magnitude of strain rate from previous iteration
        mag = np.sqrt(model.strainxx0[:model.nel]**2 
                      + model.strainyy0[:model.nel]**2
                      + 2 * model.strainxy0[:model.nel]**2)
    elif str(soln) == '1' or soln == 'apost' or soln == 'aposteriori':
        mag = np.sqrt(model.strainxx1[:model.nel]**2 
                      + model.strainyy1[:model.nel]**2
                      + 2 * model.strainxy1[:model.nel]**2)
    else:
        print('Solution not recognised.')
        return
    
    for i in obrange:
        # Element of observation
        e = model.elofstrainob[i]
        
        model.strainobvalue[comp,i] = mag[e] * style[e]
        if mag[e] < minmag:
            # Use minmag to calculate std error if magnitude is too small
            model.strainobse[comp,i] = minmag * stylese[e]
        else:
            model.strainobse[comp,i] = mag[e] * stylese[e]
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    