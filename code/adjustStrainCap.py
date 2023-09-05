#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Hamish Hirschberg
"""

import numpy as np

def uniformStress(model, postProp=0., aveRange=None, newStress=None):
    """Adjusts strain rate capacity to produce uniform stress magnitudes.
    
    This function adjusts strain rate capacity to produce approximately
    uniform a priori stress magnitudes in all elements. Strain-rate
    capacities in elements with large apriori stress magnitudes are increased
    to make the elements weaker. Strain-rate capacities in elements with
    small a priori stress magnitudes are reduced to make the elements
    stronger.
    
    Parameters
    ----------
    model : permdefmap model
        Model where the strain-rate capacities are being adjusted. The
        a priori and a posteriori results need to have been read into
        the model but this function does not check for that.
    postProp : float or array of float, default=0.
        Perform adjustment based on a proportion of the a posteriori stress
        relative to a priori stress. A proportion of 0 uses just the
        a priori stress. A proportion of 1 uses the a posteriori stress.
        A proportion of 0.5 uses the average of the a priori and
        a posteriori stress. This allows the adjustment to be applied with
        another adjustment, e.g. to force potentials, with the proportion
        being the same for both adjustments.
    aveRange : range or array of int or array of bool, default=None
        Average stress over this range of elements. Any variable that
        is accepted by a numpy array as an index can be used. Default
        averages over all elements.
    newStress : float, default=None
        New stress magnitude that elements are adjusted towards. Magnitude
        is sqrt(0.5*s_ij*s_ij). Ignored if aveRange is set. Default
        calculates new stress magnitude using average from previous
        iteration.
    """
    # Set range to calculate stress average over
    if aveRange is None:
        aveRange = np.arange(model.nel)
    else:
        newStress = None
    
    if newStress is None:
        # A priori stress magnitude
        mag0 = np.sqrt(model.stressxx0**2 + model.stressyy0**2 
                      + 2*model.stressxy0**2)
        # Total area of all elements
        atot = np.sum(model.elarea[aveRange])
        # Sum of stress magnitudes, weighted by area
        magtot = np.sum(model.elarea[aveRange] * mag0[aveRange])
        # Mean stress magnitude, weightedy by area
        magave = magtot / atot
    else:
        # Use given stress magnitude, converted to convention.
        magave = newStress * np.sqrt(2)
    
    # Calculate stress based on proportion of a posteriori used
    stressxx = (model.stressxx0[:model.nel]
                + model.stressxxd[:model.nel]*postProp)
    stressyy = (model.stressyy0[:model.nel]
                + model.stressyyd[:model.nel]*postProp)
    stressxy = (model.stressxy0[:model.nel]
                + model.stressxyd[:model.nel]*postProp)
    
    # New stress magnitude
    mag1 = np.sqrt(stressxx**2 + stressyy**2 + 2*stressxy**2)
    
    scale = np.divide(mag1, magave)
    # Scale strain-rate capacity
    model.straincapc[:model.nel] *= scale
    model.straincapcc[:model.nel] *= scale
    model.straincapcs[:model.nel] *= scale
    model.straincaps[:model.nel] *= scale
    model.straincapsc[:model.nel] *= scale
    model.straincapss[:model.nel] *= scale
    



    
    
    
    
    
    
    
    
    
    
    
    
    
    
    