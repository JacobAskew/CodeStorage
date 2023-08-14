#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 15:20:35 2023

@author: jacobaskew
"""
# from scintools.scint_sim import Simulation
from scintools.scint_utils import write_results
from scintools.dynspec import Dynspec
from copy import deepcopy as cp
import os
# import numpy as np

# 2022-12-30_Time0_Freq814_Dnu00591_Tau460097acf2d_approxdynspec_2Dfit
# Inputs
observation_date2 = '2022-11-21'
outfile = '/Users/jacobaskew/Desktop/test.txt'
if os.path.exists(outfile):
    os.remove(outfile)

#

wd0 = '/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/'
wd = wd0+'New/'
datadir = wd + 'Dynspec/'
outdir = wd + 'DataFiles/'
dynspecfile2 = \
    outdir+'DynspecPlotFiles/'+observation_date2 + \
    'Zeroed_CompleteDynspec.dynspec'

dyn = Dynspec(filename=dynspecfile2, process=False)

min_time = 170
cfreq = 844

dyn_cropped = cp(dyn)
dyn_cropped.refill(method='mean')
dyn_cropped.crop_dyn(tmin=min_time, tmax=min_time+10, fmin=cfreq-15,
                     fmax=cfreq+15)
# dyn_cropped.crop_dyn(tmin=min_time, tmax=min_time+10, fmin=818, fmax=821)
dyn_cropped.plot_dyn()
weights_2dacf = True
# dyn_cropped.get_acf_tilt(method='acf2d_approx', plot=True)
# dyn_cropped.plot_acf(input_acf=dyn_cropped.acf)
dyn_cropped.get_scint_params(method='acf2d_approx', display=True,
                             plot=True, nscale=4,
                             dnuscale_ceil=0.4,
                             tauscale_ceil=600,
                             weights_2dacf=weights_2dacf,
                             redchisqr=weights_2dacf,
                             phasewrapper=True,
                             cutoff=True, alpha=5/3)
# dyn_cropped.get_scint_params(method='acf2d_approx', display=True,
#                               plot=True, nscale=4,
#                               dnuscale_ceil=dyn_cropped.dnu*3,
#                               tauscale_ceil=dyn_cropped.tau*3,
#                               weights_2dacf=weights_2dacf,
#                               redchisqr=weights_2dacf,
#                               phasewrapper=False,
#                               cutoff=True, alpha=5/3)
print("Scint timescale:", dyn_cropped.tau, "+/-", dyn_cropped.tauerr, "s")
print("Scint bandwidth:", dyn_cropped.dnu, "+/-", dyn_cropped.dnuerr, "MHz")
# print("tilt in the acf:", dyn_cropped.acf_tilt, "+/-",
#       dyn_cropped.acf_tilt_err, "mins/MHz")
print("phasegrad in the acf:", dyn_cropped.phasegrad, "+/-",
      dyn_cropped.phasegraderr, "mins/MHz")
write_results(outfile, dyn=dyn_cropped)
# print("Error Message:", dyn_cropped.error_message)
