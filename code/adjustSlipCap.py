#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Hamish Hirschberg
"""
import numpy as np

def anisotropicallyDip(model, dip, dipunc, prop=1., slipmin=1e-12, capmin=1e-6):
    """Adjusts slip rate capacity anisotropically.
    
    This function adjusts slip rate capacity anisotropically on a
    semgent-by-segment basis based on the relative values of the
    a posteriori and a priori slip rates. The adjustment is approximately
    the a posteriori slip rate divided by the a priori slip rate. The
    c-component of slip rate capacity is based on the normal component
    of slip rate. The s-component is based on the transverse component
    of slip rate and the c-component of capacity.
    
    Parameters
    ----------
    model : permdefmap model
        Model where the slip-rate capacities are being adjusted. The
        a priori and a posteriori results need to have been read into
        the model but this function does not check for that.
    prop : float, default=1.
        A value, typically between 0 and 1, indicating the proportion
        of the standard adjustment should be applied.
    slipmin : float, default=1e-12
        Minimum relative slip rate to avoid numerical problems
        associated with dividing by a small number.
    capmin : float, default=1e-6
        Minimum s-component of slip rate capacity relative to c-component
        of slip rate capacity.
    """
    
    # Set minimum relative slip rate
    magtot = np.sum(model.slipmag0 * model.seglength)
    lsegtot = np.sum(model.seglength)
    magmin = slipmin * magtot / lsegtot
    
    # Set combined (c+s)-component of slip rate capacity
    # A priori with min slip rate
    slip0 = np.abs(model.slipt0)
    slip0[slip0<magmin] = magmin
    # A posteriori with min slip rate
    slip1 = model.slipt1 * np.sign(model.slipt0)
    # Account for a posteriori in opposite direction to a priori
    slip1[slip1<-slip0] = -slip0[slip1<-slip0]
    slip1 = np.abs(slip1)
    slip1[slip1<magmin] = magmin
    # Caculate scale for (c+s)-component
    scale = np.divide(slip1, slip0, where=slip0!=0) ** prop
    slipcap = model.slipcapc + model.slipcaps
    slipcapnew = slipcap * scale
    
    # Set c-component of slip rate capacity
    # A priori with min slip rate
    slip0 = np.abs(model.slipn0)
    slip0[slip0<magmin] = magmin
    # A posteriori with min slip rate
    slip1 = model.slipn1 * np.sign(model.slipn0)
    # Account for a posteriori in opposite direction to a priori
    slip1[slip1<-slip0] = -slip0[slip1<-slip0]
    slip1 = np.abs(slip1)
    slip1[slip1<magmin] = magmin
    # Caculate scale for c-component
    scale = np.divide(slip1, slip0, where=slip0!=0) ** prop
    capcnew = model.slipcapc * scale
    
    # Check resulting dip is within uncertainty
    # Calculate cos**2 of effective dip
    ceff = capcnew / slipcapnew
    # Max dip and min cos**2
    dipmax = np.min([np.ones_like(dip)*(np.pi/2-capmin),dip+dipunc],axis=0)
    cmin = np.cos(dipmax) ** 2
    # Min dip and max cos**2
    dipmin = np.max([np.ones_like(dip)*capmin,dip-dipunc],axis=0)
    cmax = np.cos(dipmin) ** 2
    
    # Identify dips outside constraints
    dipover = ceff > cmax
    dipunder = ceff < cmin
    dipbad = np.logical_or(dipover, dipunder)
    # Get new effective cos**2 for dips outside constraints
    ceff[dipover] = cmax[dipover]
    ceff[dipunder] = cmin[dipunder]
    
    # Apply weighted combination of capacity adjustments for bad dips
    mag0 = (model.tract0[dipbad]*model.slipt0[dipbad] + model.tracn0[dipbad]*model.slipn0[dipbad]) * ceff[dipbad]
    mag1 = (model.tract0[dipbad] * model.slipt0[dipbad] * slipcapnew[dipbad] * ceff[dipbad]
            + model.tracn0[dipbad] * model.slipn0[dipbad] * capcnew[dipbad])
    slipcapnew[dipbad] = mag1 / mag0
    capcnew[dipbad] = slipcapnew[dipbad] * ceff[dipbad]
    
    # Store new capacities
    hasseg = model.sideonfault >= 0
    model.slipcapc[hasseg] = capcnew[hasseg]
    model.slipcaps[hasseg] = slipcapnew[hasseg] - model.slipcapc[hasseg]











