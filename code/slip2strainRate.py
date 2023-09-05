#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Hamish Hirschberg
"""

import numpy as np

def someObs(model, obs):
    """Convert specified slip rate observation to strain rate.
    
    This function converts a single slip rate observation into a strain
    rate observation in neighbouring elements. The slip rate is split
    between the two points at each end of the observation segment and
    is then distributed as strain rate equally between the elements
    with that point as a vertex.
    
    Parameters
    ----------
    model : permdefmap model
        Convert slip rate observation in this model.
    obs : array of float
        Index of slip rate observation to be converted.
    """
    
    from addStrainrateObs import addNew
    from addSliprateObs import removeObs
    
    # Load geometry
    model.loadAllGeometry()
    
    # Set up array of strain rate in each element...
    exxe = np.zeros((model.nel))
    eyye = np.zeros((model.nel))
    exye = np.zeros((model.nel))
    # ... and standard error.
    seexxe = np.zeros((model.nel))
    seeyye = np.zeros((model.nel))
    seexye = np.zeros((model.nel))
    
    for i in obs:
        # Fault segment geometry
        side = model.sideofslipob[i]
        gp1 = model.gp1onside[side]
        gp2 = model.gp2onside[side]
        fault = model.faultofslipob[i]
        seg = model.segonside[side]
        length = model.seglength[seg,fault]
        # Tangents and normals
        tx = model.segtx[seg,fault]
        ty = model.segty[seg,fault]
        nx = model.segnx[seg,fault]
        ny = model.segny[seg,fault]
        # Rotate observations from tn to xy coordinates.
        # Note cannot assume A is orthonormal.
        A = np.array([[model.slipobcoefft[0, i], model.slipobcoeffn[0, i]],
                      [model.slipobcoefft[1, i], model.slipobcoeffn[1, i]]])
        b = np.array([model.slipobvalue[0, i], model.slipobvalue[1, i]])
        [slipt, slipn] = np.linalg.solve(A, b)
        # Convert slip rate observations to strain rates * area / 2 at faults
        exxf = 0.5 * (slipn*nx + slipt*tx) * nx * length
        eyyf = 0.5 * (slipn*ny + slipt*ty) * ny * length
        exyf = 0.5 * (slipn*nx*ny + 0.5*slipt*(nx*ty+ny*tx)) * length
        # Rotate slip rate std err and convert to strain rate std err
        b = np.array([model.slipobse[0, i], model.slipobse[1, i]])
        [seslipt, seslipn] = np.linalg.solve(A, b)
        seexxf = 0.5 * (np.abs(seslipn*nx*nx) + np.abs(seslipt*tx*nx)) * length
        seeyyf = 0.5 * (np.abs(seslipn*ny*ny) + np.abs(seslipt*ty*ny)) *length
        seexyf = 0.5 * (np.abs(seslipn * nx * ny)
                        + np.abs(0.5 * seslipt * (nx*ty+ny*tx))) * length
        
        # Apply half of observation to elements surrounding each end point
        for gp in [gp1, gp2]:
            # Identify elements surrounding point
            els = np.logical_or(model.gp1ofel==gp, model.gp2ofel==gp)
            els = np.logical_or(els, model.gp3ofel==gp)
            area = np.sum(model.elarea[els])
            # Calculate average strain rate and standard error
            exx = exxf / area
            eyy = eyyf / area
            exy = exyf / area
            seexx = seexxf / area
            seeyy = seeyyf / area
            seexy = seexyf / area
            # Add observation to element total
            for el in els:
                exxe[el] += exx
                eyye[el] += eyy
                exye[el] += exy
                seexxe[el] += seexx
                seeyye[el] += seeyy
                seexye[el] += seexy
    
    # Trim strain rate observation arrays to only those with observations
    hasob = seexxe > 0
    els = np.arange(model.nel)[hasob]
    se = np.array([seexxe[hasob], seeyye[hasob], seexye[hasob]])
    values = np.array([exxe[hasob], eyye[hasob], exye[hasob]])
    # Store strain rate observations
    addNew(model, els, se, values)
    # Remove slip rate observations
    removeObs(model, obs)

def rectTaper(dist):
    """Rectangular taper.
    
    A rectangular 'taper' for use determining the proportion of the
    strain rate being assigned at a given distance relative to the
    width over which strain rates are being assigned. The returned
    value is constant for all input distances.
    
    Parameters
    ----------
    dist : float
        Distance from the centre of the taper, as a function of the
        width of the taper. Centre of taper is 0. and end of taper is 1.
    
    Returns
    -------
    out : float
        Value of the taper at specified distance.
    """
    return 1.

def triTaper(dist):
    """Triangular taper.
    
    A triangular taper for use determining the proportion of the
    strain rate being assigned at a given distance relative to the
    width over which strain rates are being assigned. The returned
    value is 2 for the centre of the taper, decreasing linearly to 0
    at the end of the taper.
    
    Parameters
    ----------
    dist : float
        Distance from the centre of the taper, as a function of the
        width of the taper. Centre of taper is 0. and end of taper is 1.
    
    Returns
    -------
    out : float
        Value of the taper at specified distance.
    """
    return 2 * (1-dist)

def cosTaper(dist):
    """Cosine taper.
    
    A cosine taper for use determining the proportion of the
    strain rate being assigned at a given distance relative to the
    width over which strain rates are being assigned. The returned
    value is 2 for the centre of the taper, decreasing according to a
    sinusoid to 0 for the end of the taper, with an inflection point
    halfway between with a value of 1.
    
    Parameters
    ----------
    dist : float
        Distance from the centre of the taper, as a function of the
        width of the taper. Centre of taper is 0. and end of taper is 1.
    
    Returns
    -------
    out : float
        Value of the taper at specified distance.
    """
    return 1 + np.cos(dist*np.pi*0.5)

def slip2strainTaper(model, width, taper='rect', clear=False,
                     fill=True, unc=1e-12):
    """Converts slip rates to strain rates spread over elements.
    
    This function takes slip rate observations and converts them to
    strain rate observations spread over elements within a specified
    distance from the slip rate observation's segment. The
    contribution of a given slip rate observation at a given point is
    that slip rate times a distance factor determined by the taper.
    The strain rate at a point is the sum of contributions from all
    the slip rate observations. The output strain rate observation in
    an element is the mean of the strain rates at its vertices.
    
    Parameters
    ----------
    model : permdefmap model
        Model where the slip rate observations are being converted to
        strain rate observations.
    width : float
        Width of the taper. This is half the distance over which the
        strain rate is spread. Points more than one width from the
        fault do not get a contribution from that fault.
    taper : string, default='rect'
        Taper used to spread strain rate among elements. The taper can
        be specified as:
            'rectangular' or 'rect' or 'r'
            'triangular' or 'tri' or 't'
            'cosine' or 'cos' or 'c'
        See the individual taper functions for more information about
        each.
    clear : bool, default=False
        True will clear all pre-existing strain rate observations.
        False will add the observations generated here to the existing
        observations.
    fill : bool, default=True
        Fill elements that would not otherwise get a strain rate
        observation because they are too far from any slip rate
        observations. These elements are given strain rate observations
        of zero for all three components.
    unc : float, default=1e-12
        Default uncertainty to apply to observations of zero strain
        rate for elements that are filled using fill=True. Not used
        when fill=False.
    """
    
    # Choose taper
    if taper=='r' or taper=='rect' or taper=='rectangular':
        taperfun = rectTaper
    elif taper=='t' or taper=='tri' or taper=='triangular':
        taperfun = triTaper
    elif taper=='c' or taper=='cos' or taper=='cosine':
        taperfun = cosTaper
    
    # Load points if not already done so
    if model.gp1ofel[0] == 0:
        print('Element points not loaded. Loading now.')
        model.getElementPoints()
    
    if clear:
        # Clear previous observations if desired
        model.nstrainob = 0
    
    # Width expressed as degrees
    degw = width / model.earthradius * 180 / np.pi
    
    # Summed strain rates at grid points
    exxgp = np.zeros((model.ngp))
    eyygp = np.zeros((model.ngp))
    exygp = np.zeros((model.ngp))
    # Summed strain rate variances at grid points
    varexxgp = np.zeros((model.ngp))
    vareyygp = np.zeros((model.ngp))
    varexygp = np.zeros((model.ngp))
    
    # Loop through slip rate observations
    for i in range(model.nslipob):
        # Fault segment geometry
        gp1 = model.gp1onside[model.sideofslipob[i]]
        gp2 = model.gp2onside[model.sideofslipob[i]]
        x1 = model.gplong[gp1]
        x2 = model.gplong[gp2]
        y1 = model.gplat[gp1]
        y2 = model.gplat[gp2]
        dx = (x2 - x1) * np.cos(np.radians((y1+y2) / 2))
        dy = y2 - y1
        dl = np.hypot(dx, dy)
        # Tangents and normals
        tx = dx / dl
        ty = dy / dl
        nx = -dy / dl
        ny = dx / dl
        # Rotate observations from tn to xy coordinates.
        # Note cannot assume A is orthonormal.
        A = np.array([[model.slipobcoefft[0, i], model.slipobcoeffn[0, i]],
                      [model.slipobcoefft[1, i], model.slipobcoeffn[1, i]]])
        b = np.array([model.slipobvalue[0, i], model.slipobvalue[1, i]])
        [slipt, slipn] = np.linalg.solve(A, b)
        # Convert slip rate observations to strain rates at faults
        exxf = 0.5 * (slipn*nx + slipt*tx) * nx / width
        eyyf = 0.5 * (slipn*ny + slipt*ty) * ny / width
        exyf = 0.5 * (slipn*nx*ny + 0.5*slipt*(nx*ty+ny*tx)) / width
        # Rotate slip rate uncertainties and convert to strain rate uncertainties
        b = np.array([model.slipobse[0, i], model.slipobse[1, i]])
        [seslipt, seslipn] = np.linalg.solve(A, b)
        seexxf = 0.5 * (np.abs(seslipn*nx*nx) + np.abs(seslipt*tx*nx)) / width
        seeyyf = 0.5 * (np.abs(seslipn*ny*ny) + np.abs(seslipt*ty*ny)) / width
        seexyf = 0.5 * (np.abs(seslipn * nx * ny)
                        + np.abs(0.5 * seslipt * (nx*ty+ny*tx))) / width
        
        # Find points within width of fault
        degwdl = degw + dl      # Width + segment length in degrees
        for p in range(model.ngp):
            # Start with quick and dirty calculation to remove far points
            # Compare point to start point of segment
            if (np.abs(x1 - model.gplong[p]) <= degwdl
                and np.abs(y1 - model.gplat[p]) <= degwdl):
                # Now perform proper calculation
                # Distances from start point in degrees of latitude
                dx1 = ((model.gplong[p] - x1)
                       * np.cos(np.radians((model.gplat[p]+y1) / 2)))
                dy1 = model.gplat[p] - y1
                # Check if within width of fault segment using dot product with
                # normal to fault.
                if np.abs(dx1*nx + dy1*ny) <= degw:
                    # Check if transverse distance along fault is within
                    # segment length by using dot product with tangent to fault.
                    distt = dx1*tx + dy1*ty
                    if distt >= 0 and distt <= dl:
                        # Now we know we want this point.
                        distn = np.abs(dx1*nx + dy1*ny)
                        # Apply taper based on distance from fault.
                        taperfac = taperfun(distn/degw)
                        if distt==0 or distt==dl:
                            # Account for point where segments meet
                            taperfac*=0.5
                        # Add observation contribution to point sum
                        exxgp[p] += taperfac * exxf
                        eyygp[p] += taperfac * eyyf
                        exygp[p] += taperfac * exyf
                        varexxgp[p] += (taperfac*seexxf) ** 2
                        vareyygp[p] += (taperfac*seeyyf) ** 2
                        varexygp[p] += (taperfac*seexyf) ** 2
    
    # Convert strain at points to strain rate in elements
    for e in range(model.nel):
        gpe = np.array([model.gp1ofel[e], model.gp2ofel[e], model.gp3ofel[e]])
        exxe = np.mean(exxgp[gpe])
        eyye = np.mean(eyygp[gpe])
        exye = np.mean(exygp[gpe])
        if exxe != 0. or eyye != 0. or exye != 0.:
            # If strain rate observation is nonzero, then store it
            model.strainobvalue[0, model.nstrainob] = exxe
            model.strainobvalue[1, model.nstrainob] = eyye
            model.strainobvalue[2, model.nstrainob] = exye
            # Standard error
            model.strainobse[0, model.nstrainob] = np.mean(np.sqrt(varexxgp[gpe]))
            model.strainobse[1, model.nstrainob] = np.mean(np.sqrt(vareyygp[gpe]))
            model.strainobse[2, model.nstrainob] = np.mean(np.sqrt(varexygp[gpe]))
            # Observation location, direction etc.
            model.nstrainobcomp[model.nstrainob] = 3
            model.elofstrainob[model.nstrainob] = e
            model.strainobcoeffxx[0, model.nstrainob] = 1.
            model.strainobcoeffyy[1, model.nstrainob] = 1.
            model.strainobcoeffxy[2, model.nstrainob] = 1.
            model.nstrainob += 1
        elif fill:
            # If strain rate observation is zero but still setting observation
            model.strainobvalue[:, model.nstrainob] = 0.
            model.strainobse[:, model.nstrainob] = unc
            model.nstrainobcomp[model.nstrainob] = 3
            model.elofstrainob[model.nstrainob] = e
            model.strainobcoeffxx[0, model.nstrainob] = 1.
            model.strainobcoeffyy[1, model.nstrainob] = 1.
            model.strainobcoeffxy[2, model.nstrainob] = 1.
            model.nstrainob += 1
    
def slip2strainTransferTaper(mold, mnew, width, taper='rect', clear=False,
                             fill=True, unc=1e-12, maxlen=0.):
    """Converts slip rates in one model to strain rates spread over
    elements in another model.
    
    This function takes slip rate observations from one model and
    converts them to strain rate observations spread over elements
    in another model within a specified distance from the slip rate
    observation's segment. The contribution of a given slip rate
    observation at a given point is that slip rate times a distance
    factor determined by the taper. The strain rate at a point is the
    sum of contributions from all the slip rate observations. The
    output strain rate observation in an element is the mean of the
    strain rates at its vertices.
    
    Parameters
    ----------
    mold : permdefmap model
        Model that is the source of the slip rate observations.
    mnew : permdefmap model
        Model where the strain rate observations are saved to.
    width : float
        Width of the taper. This is half the distance over which the
        strain rate is spread. Points more than one width from the
        fault do not get a contribution from that fault.
    taper : string, default='rect'
        Taper used to spread strain rate among elements. The taper can
        be specified as:
            'rectangular' or 'rect' or 'r'
            'triangular' or 'tri' or 't'
            'cosine' or 'cos' or 'c'
        See the individual taper functions for more information about
        each.
    clear : bool, default=False
        True will clear all pre-existing strain rate observations.
        False will add the observations generated here to the existing
        observations.
    fill : bool, default=True
        Fill elements that would not otherwise get a strain rate
        observation because they are too far from any slip rate
        observations. These elements are given strain rate observations
        of zero for all three components.
    unc : float, default=1e-12
        Default uncertainty to apply to observations of zero strain
        rate for elements that are filled using fill=True. Not used
        when fill=False.
    maxlen : float, default=0.
        Maximum length of faults that are converted to strain rates.
        Any observations on faults longer than this length are not
        converted to strain rates. The default value of 0 means that
        observations on all faults are converted to strain rates.
    """
    
    # Choose taper
    if taper=='r' or taper=='rect' or taper=='rectangular':
        taperfun = rectTaper
    elif taper=='t' or taper=='tri' or taper=='triangular':
        taperfun = triTaper
    elif taper=='c' or taper=='cos' or taper=='cosine':
        taperfun = cosTaper
    
    # Load points if not already done so
    if mnew.gp1ofel[0] == 0:
        print('Element points not loaded. Loading now.')
        mnew.getElementPoints()
    
    if clear:
        # Clear previous observations if desired
        mnew.nstrainob = 0
    
    if maxlen:
        # Load segment lengths if needed
        if mold.seglength[0,0] == 0.:
            print('Fault segment lengths not loaded. Calculating now.')
            mold.faultSegmentLength()
    else:
        maxlen = np.inf
    
    # Width expressed as degrees
    degw = width / mnew.earthradius * 180 / np.pi
    
    # Summed strain rates at grid points
    exxgp = np.zeros((mnew.ngp))
    eyygp = np.zeros((mnew.ngp))
    exygp = np.zeros((mnew.ngp))
    # Summed strain rate variances at grid points
    varexxgp = np.zeros((mnew.ngp))
    vareyygp = np.zeros((mnew.ngp))
    varexygp = np.zeros((mnew.ngp))
    
    # Loop through slip rate observations
    for i in range(mold.nslipob):
        if np.sum(mold.seglength[:, mold.faultofslipob[i]]) > maxlen:
            # Ignore observations on fault longer than maxlen
            continue
        # Fault segment geometry
        gp1 = mold.gp1onside[mold.sideofslipob[i]]
        gp2 = mold.gp2onside[mold.sideofslipob[i]]
        x1 = mold.gplong[gp1]
        x2 = mold.gplong[gp2]
        y1 = mold.gplat[gp1]
        y2 = mold.gplat[gp2]
        dx = (x2 - x1) * np.cos(np.radians((y1+y2) / 2))
        dy = y2 - y1
        dl = np.hypot(dx, dy)
        # Tangents and normals
        tx = dx / dl
        ty = dy / dl
        nx = -dy / dl
        ny = dx / dl
        # Rotate observations from tn to xy coordinates.
        # Note cannot assume A is orthonormal.
        A = np.array([[mold.slipobcoefft[0, i], mold.slipobcoeffn[0, i]],
                      [mold.slipobcoefft[1, i], mold.slipobcoeffn[1, i]]])
        b = np.array([mold.slipobvalue[0, i], mold.slipobvalue[1, i]])
        [slipt, slipn] = np.linalg.solve(A, b)
        # Convert slip rate observations to strain rates at faults
        exxf = 0.5 * (slipn*nx + slipt*tx) * nx / width
        eyyf = 0.5 * (slipn*ny + slipt*ty) * ny / width
        exyf = 0.5 * (slipn*nx*ny + 0.5*slipt*(nx*ty+ny*tx)) / width
        # Rotate slip rate uncertainties and convert to strain rate uncertainties
        b = np.array([mold.slipobse[0, i], mold.slipobse[1, i]])
        [seslipt, seslipn] = np.linalg.solve(A, b)
        seexxf = 0.5 * (np.abs(seslipn*nx*nx) + np.abs(seslipt*tx*nx)) / width
        seeyyf = 0.5 * (np.abs(seslipn*ny*ny) + np.abs(seslipt*ty*ny)) / width
        seexyf = 0.5 * (np.abs(seslipn * nx * ny)
                        + np.abs(0.5 * seslipt * (nx*ty+ny*tx))) / width
        
        # Find points within width of fault
        degwdl = degw + dl      # Width + segment length in degrees
        for p in range(mnew.ngp):
            # Start with quick and dirty calculation to remove far points
            # Compare point to start point of segment
            if (np.abs(x1 - mnew.gplong[p]) <= degwdl
                and np.abs(y1 - mnew.gplat[p]) <= degwdl):
                # Now perform proper calculation
                # Distances from start point in degrees of latitude
                dx1 = ((mnew.gplong[p] - x1)
                       * np.cos(np.radians((mnew.gplat[p]+y1) / 2)))
                dy1 = mnew.gplat[p] - y1
                # Check if within width of fault segment using dot product with
                # normal to fault.
                if np.abs(dx1*nx + dy1*ny) <= degw:
                    # Check if transverse distance along fault is within
                    # segment length by using dot product with tangent to fault.
                    distt = dx1*tx + dy1*ty
                    if distt >= 0 and distt <= dl:
                        # Now we know we want this point.
                        distn = np.abs(dx1*nx + dy1*ny)
                        # Apply taper based on distance from fault.
                        taperfac=taperfun(distn/degw)
                        if distt==0 or distt==dl:
                            # Account for point where segments meet
                            taperfac*=0.5
                        # Add observation contribution to point sum
                        exxgp[p] += taperfac * exxf
                        eyygp[p] += taperfac * eyyf
                        exygp[p] += taperfac * exyf
                        varexxgp[p] += (taperfac*seexxf) ** 2
                        vareyygp[p] += (taperfac*seeyyf) ** 2
                        varexygp[p] += (taperfac*seexyf) ** 2
    
    # Convert strain at points to strain rate in elements
    for e in range(mnew.nel):
        gpe = np.array([mnew.gp1ofel[e], mnew.gp2ofel[e], mnew.gp3ofel[e]])
        exxe = np.mean(exxgp[gpe])
        eyye = np.mean(eyygp[gpe])
        exye = np.mean(exygp[gpe])
        if exxe != 0. or eyye != 0. or exye != 0.:
            # If strain rate observation is nonzero, then store it
            mnew.strainobvalue[0, mnew.nstrainob] = exxe
            mnew.strainobvalue[1, mnew.nstrainob] = eyye
            mnew.strainobvalue[2, mnew.nstrainob] = exye
            # Standard error
            mnew.strainobse[0, mnew.nstrainob] = np.mean(np.sqrt(varexxgp[gpe]))
            mnew.strainobse[1, mnew.nstrainob] = np.mean(np.sqrt(vareyygp[gpe]))
            mnew.strainobse[2, mnew.nstrainob] = np.mean(np.sqrt(varexygp[gpe]))
            # Observation location, direction etc.
            mnew.nstrainobcomp[mnew.nstrainob] = 3
            mnew.elofstrainob[mnew.nstrainob] = e
            mnew.strainobcoeffxx[0, mnew.nstrainob] = 1.
            mnew.strainobcoeffyy[1, mnew.nstrainob] = 1.
            mnew.strainobcoeffxy[2, mnew.nstrainob] = 1.
            mnew.nstrainob += 1
        elif fill:
            # If strain rate observation is zero but still setting observation
            mnew.strainobvalue[:, mnew.nstrainob] = 0.
            mnew.strainobse[:, mnew.nstrainob] = unc
            mnew.nstrainobcomp[mnew.nstrainob] = 3
            mnew.elofstrainob[mnew.nstrainob] = e
            mnew.strainobcoeffxx[0, mnew.nstrainob] = 1.
            mnew.strainobcoeffyy[1, mnew.nstrainob] = 1.
            mnew.strainobcoeffxy[2, mnew.nstrainob] = 1.
            mnew.nstrainob += 1
            
def slip2strainTransferElements(mold, mnew, clear=False, fill=True, unc=0.,
                                minstrain=0., prec=1e-12):
    """Converts slip rates in one model to strain rates in elements in
    another model.
    
    This function takes slip rate observations on fault segments from
    one model and converts them to strain rate in elements that the
    segment passes through in another model. Strain rate contributions
    from multiple faults are summed in each element into strain rate
    observations in that element.
    
    Parameters
    ----------
    mold : permdefmap model
        Model that is the source of the slip rate observations.
    mnew : permdefmap model
        Model where the strain rate observations are saved to.
    clear : bool, default=False
        True will clear all pre-existing strain rate observations.
        False will add the observations generated here to the existing
        observations.
    fill : bool, default=True
        Fill elements that would not otherwise get a strain rate
        observation because they are too far from any slip rate
        observations. These elements are given strain rate observations
        of zero for all three components.
    unc : float, default=1e-12
        Default uncertainty to apply to observations of zero strain
        rate for elements that are filled using fill=True. Not used
        when fill=False.
    minstrain : float, default=0.
        Elements with total strain rates less than minstrain are
        treated as having strain rates of zero.
    prec : float, default=1e-12
        Precision for accounting for numerical errors when determing
        intersections of lines.
    """
    
    # Load points if not already done so
    if mnew.gp1ofel[0] == 0:
        print('Element points not loaded. Loading now.')
        mnew.getElementPoints()
    
    if clear:
        # Clear previous observations if desired
        mnew.nstrainob = 0
     
    # Summed strain rates in elements
    exx = np.zeros((mnew.nel))
    eyy = np.zeros((mnew.nel))
    exy = np.zeros((mnew.nel))
    # Summed strain rate variances in elements
    varexx = np.zeros((mnew.nel))
    vareyy = np.zeros((mnew.nel))
    varexy = np.zeros((mnew.nel))
    
    # Initial element for search
    el = 0
    # Loop through slip rate observations
    for i in range(mold.nslipob):
        # Fault segment geometry
        s = mold.segonside[mold.sideofslipob[i]]
        f = mold.faultofslipob[i]
        gpf = mold.gponfault[s:s+2,f]
        lo0,lo1 = mold.gplong[gpf]
        la0,la1 = mold.gplat[gpf]
        dx = (lo1 - lo0) * np.cos(np.radians((la0+la1) / 2))
        dy = la1 - la0
        dl = np.hypot(dx, dy)
        # Tangents and normals
        tx = dx / dl
        ty = dy / dl
        nx = -dy / dl
        ny = dx / dl
        # Rotate observations from tn to xy coordinates.
        # Note cannot assume A is orthonormal.
        A = np.array([[mold.slipobcoefft[0, i], mold.slipobcoeffn[0, i]],
                      [mold.slipobcoefft[1, i], mold.slipobcoeffn[1, i]]])
        b = np.array([mold.slipobvalue[0, i], mold.slipobvalue[1, i]])
        [slipt, slipn] = np.linalg.solve(A, b)
        # Convert slip rate observations to strain rates at faults
        exxf = 0.5 * (slipn*nx + slipt*tx) * nx
        eyyf = 0.5 * (slipn*ny + slipt*ty) * ny
        exyf = 0.5 * (slipn*nx*ny + 0.5*slipt*(nx*ty+ny*tx))
        # Rotate slip rate uncertainties and convert to strain rate uncertainties
        b = np.array([mold.slipobse[0, i], mold.slipobse[1, i]])
        [seslipt, seslipn] = np.linalg.solve(A, b)
        seexxf = 0.5 * (np.abs(seslipn*nx*nx) + np.abs(seslipt*tx*nx))
        seeyyf = 0.5 * (np.abs(seslipn*ny*ny) + np.abs(seslipt*ty*ny))
        seexyf = 0.5 * (np.abs(seslipn * nx * ny)
                        + np.abs(0.5 * seslipt * (nx*ty+ny*tx)))
        
        # Find element just after first point (to avoid coincident grid points)
        lo = lo0*(1-1e-3) + lo1*1e-3
        la = la0*(1-1e-3) + la1*1e-3
        out = mnew.findElement(lo, la, weights=True, flag=True, el=el)
        if out[4] == 2:
            print('Unable to find element for start of fault '+str(f))
            return
        el = out[0]
        
        # Length of segment accounted for so far
        prevdist = 0.
        
        # Loop through elements on this segment (limited to number of elements)
        for e in range(mnew.nel):
            # Start with element with start of segment
            # Find intersect of the line with the sides of element
            lenfrac = [0, 0, 0]
            # Grid points of element (start point also end point)
            gps = np.array([mnew.gp1ofel[el], mnew.gp2ofel[el],
                            mnew.gp3ofel[el], mnew.gp1ofel[el]])
            for side in range(3):
                # Find relative intersect of side with segment
                x0, x1 = mnew.gplong[gps[side:side+2]]
                y0, y1 = mnew.gplat[gps[side:side+2]]
                det = (lo1-lo0) * (y1-y0) - (la1-la0) * (x1-x0)
                dist = (x1-lo0) * (y1-y0) - (y1-la0) * (x1-x0)
                # Distance along side where segment intersects
                # Distances are fractions of length of segment
                lenfrac[side] = dist / det
            # Identify order in which segment intersects sides
            # Mid is middle side and max is last side intersected
            midf = np.median(lenfrac)
            maxf = np.max(lenfrac)
            midi = lenfrac.index(midf)
            maxi = lenfrac.index(maxf)
            if midf <= prevdist + prec:
                # Here segment enters element by intersecting mid
                # Portion of segment in element starts where previous one ended
                start = prevdist + 0
                # Portion ends when segment exits element or segment ends
                end = min(maxf, 1)
                # Side through which segment exits element
                nextside = maxi + 0
            else:
                # Here segment exits element by intersecting mid
                start = prevdist + 0
                end = min(midf, 1)
                nextside = midi + 0
            
            # Length of segment in this element
            length = (end-start) * mold.seglength[s,f]
            # Calculate the contribution to the element's strain rate
            exx[el] += exxf * length
            eyy[el] += eyyf * length
            exy[el] += exyf * length
            varexx[el] += (seexxf*length) ** 2
            vareyy[el] += (seeyyf*length) ** 2
            varexy[el] += (seexyf*length) ** 2
            
            # Set where on segment this element ended
            prevdist = end + 0
            if end >= 1 - prec:
                # End of fault segment
                break
            
            if nextside == 0:
                # Next side is side 3
                side = mnew.side3ofel[el]
            elif nextside == 1:
                # Next side is side 1
                side = mnew.side1ofel[el]
            else:
                # Next side is side 2
                side = mnew.side2ofel[el]
            # Find next element
            if mnew.el1onside[side] == el:
                el = mnew.el2onside[side] + 0
            else:
                el = mnew.el1onside[side] + 0
            if el == 0:
                # Boundary reached
                break
        else:
            print('Maximum number of elements reached on observation '+str(i))
            return
    
    # Store strain rate observations in new model
    for e in range(mnew.nel):
        if (np.abs(exx[e]) > minstrain or np.abs(eyy[e]) > minstrain
                or np.abs(exy[e]) > minstrain):
            # If strain rate obs is greater than min value,  then store it
            mnew.strainobvalue[0, mnew.nstrainob] = exx[e] / mnew.elarea[e]
            mnew.strainobvalue[1, mnew.nstrainob] = eyy[e] / mnew.elarea[e]
            mnew.strainobvalue[2, mnew.nstrainob] = exy[e] / mnew.elarea[e]
            # Standard error
            mnew.strainobse[0, mnew.nstrainob] = np.sqrt(varexx[e]) / mnew.elarea[e]
            mnew.strainobse[1, mnew.nstrainob] = np.sqrt(vareyy[e]) / mnew.elarea[e]
            mnew.strainobse[2, mnew.nstrainob] = np.sqrt(varexy[e]) / mnew.elarea[e]
            mnew.nstrainobcomp[mnew.nstrainob] = 3
            mnew.elofstrainob[mnew.nstrainob] = e
            mnew.strainobcoeffxx[0, mnew.nstrainob] = 1.
            mnew.strainobcoeffyy[1, mnew.nstrainob] = 1.
            mnew.strainobcoeffxy[2, mnew.nstrainob] = 1.
            mnew.nstrainob += 1
        elif fill:
            # If strain rate obs zero but still setting observation
            mnew.strainobvalue[:, mnew.nstrainob] = 0.
            mnew.strainobse[:, mnew.nstrainob] = unc
            mnew.nstrainobcomp[mnew.nstrainob] = 3
            mnew.elofstrainob[mnew.nstrainob] = e
            mnew.strainobcoeffxx[0, mnew.nstrainob] = 1.
            mnew.strainobcoeffyy[1, mnew.nstrainob] = 1.
            mnew.strainobcoeffxy[2, mnew.nstrainob] = 1.
            mnew.nstrainob += 1
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    