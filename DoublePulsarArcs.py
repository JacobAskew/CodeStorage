#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 16:30:32 2022

@author: jacobaskew
"""

# Making secondary spectrum arcs with the double pulsar!


##############################################################################
from scintools.dynspec import Dynspec
# from scintools.scint_utils import write_results, read_results, read_par, \
#         float_array_from_dict, get_ssb_delay, get_earth_velocity, \
#         get_true_anomaly
# import matplotlib.pyplot as plt
# import matplotlib
import numpy as np
import glob
# from copy import deepcopy as cp
# from lmfit import Parameters, minimize
# import os
from scintools.scint_sim import Simulation
##############################################################################


# def powerlaw_fitter(xdata, ydata, weights, reffreq, amp_init=1, amp_min=0,
#                     amp_max=np.inf, alpha_init=4, alpha_min=-5, alpha_max=5,
#                     steps=10000, burn=1000):
#     parameters = Parameters()
#     parameters.add('amp', value=amp_init, vary=True, min=amp_min,
# max=amp_max)
#     parameters.add('alpha', value=alpha_init, vary=True, min=alpha_min,
#                    max=alpha_max)
#     results = minimize(powerlaw, parameters,
#                        args=(xdata, ydata, weights, reffreq, amp_init),
#                        method='emcee', steps=steps, burn=burn)
#     Slope = results.params['alpha'].value
#     Slopeerr = results.params['alpha'].stderr

#     return Slope, Slopeerr


##############################################################################


# def powerlaw(params, xdata, ydata, weights, reffreq, amp):

#     if weights is None:
#         weights = np.ones(np.shape(xdata))

#     if ydata is None:
#         ydata = np.zeros(np.shape(xdata))

#     parvals = params.valuesdict()
#     if amp is None:
#         amp = parvals['amp']
#     alpha = parvals['alpha']

#     func = amp*(xdata/reffreq)**(alpha)

#     return func


##############################################################################


# def SearchEclipse(start_mjd, tobs):
#     end_mjd = start_mjd + tobs/86400
#     Eclipse_mjd = np.genfromtxt(eclipsefile, delimiter=',',
#                                 encoding=None, dtype=float)
#     Eclipse_events = np.array(np.where((Eclipse_mjd > start_mjd) *
#                               (Eclipse_mjd < end_mjd)))
#     if Eclipse_events.size == 0:
#         Eclipse_index = None
#         print("No Eclispe in dynspec")
#     else:
#         Eclipse_events_mjd = Eclipse_mjd[Eclipse_events]
#         mjds = start_mjd + dyn.times/86400
#         Eclipse_index = np.argmin(abs(mjds - Eclipse_events_mjd))
#     return Eclipse_index


# ##############################################################################
# psrname = 'J0737-3039A'
# pulsar = '0737-3039A'

# outdir = wd + 'DataFiles/'
# outfile = str(outdir) + str(psrname) + '_ScintillationResults_UHF.txt'
par_dir = '/Users/jacobaskew/Desktop/Swinburne_Daniel/ParFiles/'

# zap = True
# linear = False

# dyn1 = Dynspec(filename=dynspecs[0], process=False)
# dyn2 = Dynspec(filename=dynspecs[1], process=False)
# dyn3 = Dynspec(filename=dynspecs[2], process=False)
# dyn = dyn1 + dyn2 + dyn3
# dyn.trim_edges()  # trim zeroed edges
# dyn.crop_dyn(tmin=50, fmin=np.min(dyn.freqs)+48, fmax=np.max(dyn.freqs)-48)
# dyn.refill()  # biharmonic inpainting
# # dyn.plot_dyn()  # plot dynamic spectrum
# # dyn.plot_sspec(lamsteps=True, maxfdop=15) # plot secondary spectrum

# # Rescale the dynamic spectrum in time, using a velocity model
# dyn.scale_dyn(scale='lambda+velocity', s=0.7, d=0.9, inc=88.7, Omega=65,
#               parfile=par_dir+'J0737-3039A.par')

# # Plot the velocity-rescaled dynamic spectrum
# dyn.plot_dyn(velocity=True)
# # Fit the arc using the new spectrum, and plot
# dyn.fit_arc(velocity=True, cutmid=0, plot=True, weighted=False,
#             lamsteps=True, subtract_artefacts=True, log_parabola=True,
#             constraint=[100, 1000])
# dyn.plot_sspec(lamsteps=True, velocity=True, maxfdop=15, plotarc=True)
##############################################################################
wd = '/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/New/'
datadir = wd + 'Dynspec/'
dynspecs = sorted(glob.glob(datadir + '/*.dynspec'))
observations = []
for i in range(0, len(dynspecs)):
    observations.append(dynspecs[i].split(datadir)[1].split('-')[0]+'-' +
                        dynspecs[i].split(datadir)[1].split('-')[1]+'-' +
                        dynspecs[i].split(datadir)[1].split('-')[2])
observations = np.unique(np.asarray(observations))
for i in range(0, len(observations)):
    observation_date2 = observations[i]
    wd1 = "/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/New/"
    wd0 = wd1 + "ScintillationArcs/"

    dynspecfile = wd1+'DataFiles/DynspecPlotFiles/'+str(observation_date2) + \
        'Zap_CompleteDynspec.dynspec'

    sim = Simulation()
    dyn = Dynspec(dyn=sim, process=False)
    dyn.load_file(filename=dynspecfile)

    dyn.plot_dyn(filename=wd0+str(observation_date2)+'_dynspec.png', dpi=400)

    # Fit the arc using the new spectrum, and plot
    dyn.fit_arc(velocity=False, cutmid=0, plot=True, weighted=False,
                lamsteps=True, subtract_artefacts=True, log_parabola=True,
                constraint=[100, 1000], filename=wd0+str(observation_date2) +
                '_fitarc.png')
    dyn.lamsspec[:, 2042:2053] = 0
    dyn.plot_sspec(lamsteps=True, velocity=False, maxfdop=15, plotarc=False,
                   filename=wd0+str(observation_date2)+'_sspec.png')
    dyn.plot_sspec(lamsteps=True, velocity=False, maxfdop=15, plotarc=True,
                   filename=wd0+str(observation_date2)+'_fitsspec.png')

    # Rescale the dynamic spectrum in time, using a velocity model
    dyn.scale_dyn(scale='lambda+velocity', s=0.7, d=0.9, inc=88.7, Omega=65,
                  parfile=par_dir+'J0737-3039A.par')

    # Plot the velocity-rescaled dynamic spectrum
    dyn.plot_dyn(velocity=True,
                 filename=wd0+str(observation_date2)+'_dynspec_velocity.png')
    # Fit the arc using the new spectrum, and plot
    dyn.fit_arc(velocity=True, cutmid=0, plot=True, weighted=False,
                lamsteps=True, subtract_artefacts=True, log_parabola=True,
                constraint=[100, 1000])
    dyn.vlamsspec[:, 2042:2053] = 0
    dyn.plot_sspec(lamsteps=True, velocity=True, maxfdop=15, plotarc=False,
                   filename=wd0+str(observation_date2)+'_sspec_velocity.png')
    dyn.plot_sspec(lamsteps=True, velocity=True, maxfdop=15, plotarc=True,
                   filename=wd0+str(observation_date2) +
                   '_fitsspec_velocity.png')
##############################################################################
