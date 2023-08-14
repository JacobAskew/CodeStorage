#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 17:43:20 2023

@author: jacobaskew
"""

import numpy as np


def scint_velocity_alternate(params, dnu, tau, freq, dnuerr, tauerr):
    """
    Calculate scintillation velocity from ACF frequency and time scales
    """
    D = params['d']
    s = params['s']
    D_err = params['derr']
    kappa = params['k']

    freqGHz = freq / 1e3  # convert to GHz
    # Assuming thin screen Kolmogorov, from Cordes & Rickett 1998
    Aiss = kappa * 2.78e4 * np.sqrt((2*(1-s))/(s))
    viss = Aiss * (np.sqrt(D*dnu))/(freqGHz*tau)
    viss_err = viss * np.sqrt((D_err/(2*D))**2+(dnuerr/(2*dnu))**2 +
                              (-tauerr/tau)**2)
    return viss, viss_err
