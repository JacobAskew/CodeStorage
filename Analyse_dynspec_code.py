#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 18:38:56 2021

@author: jaskew
"""

# This code aims to test the new method sspec for model ACF fitting!

###############################################################################
from scintools.dynspec import Dynspec
import numpy as np
# from scintools.scint_utils import write_results, read_results, read_par, \
#          float_array_from_dict, get_ssb_delay, get_earth_velocity, \
#          get_true_anomaly, pars_to_params, scint_velocity
from scintools.scint_models import tau_acf_model, dnu_acf_model,\
    tau_sspec_model, dnu_sspec_model
# ,
# veff_thin_screen, effective_velocity_annual
# import glob
# from copy import deepcopy as cp
# import corner
from lmfit import Parameters, Minimizer
import matplotlib.pyplot as plt
import matplotlib
# import bilby
# import pdb
# import itertools as it
###############################################################################
wd = '/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/'
datadir = wd+'TestData/'
psrname = 'J0737-3039A'
pulsar = '0737-3039A'
spectradir = wd+'Spectra/'
eclipsefile = wd+'Datafiles/Eclipse_mjd.txt'
outdir = wd+'TestData/Datafiles/'
outfile = str(outdir) + 'J' + str(pulsar) + '_ScintillationResults.txt'

dynspec = wd+'TestData/J0737-3039A_2020-02-21-20:13:12_ch5.0_sub5.0.ar.dynspec'

uhf = False
model = True
###############################################################################


def SearchEclipse(start_mjd, tobs):
    end_mjd = start_mjd + tobs/86400
    Eclipse_mjd = np.genfromtxt(eclipsefile, delimiter=',',
                                encoding=None, dtype=float)
    Eclipse_events = np.array(np.where((Eclipse_mjd > start_mjd) *
                              (Eclipse_mjd < end_mjd)))
    if Eclipse_events.size == 0:
        Eclipse_index = None
        print("No Eclispe in dynspec")
    else:
        Eclipse_events_mjd = Eclipse_mjd[Eclipse_events]
        mjds = start_mjd + dyn.times/86400
        Eclipse_index = np.argmin(abs(mjds - Eclipse_events_mjd))
    return Eclipse_index


###############################################################################
# This part is just getting a bit of data ready as a Dynspec object:
File1 = dynspec.split(str(datadir))[1]
Filename = str(File1.split('.')[0])

dyn = Dynspec(filename=dynspec, process=False)
dyn.trim_edges()
start_mjd = dyn.mjd
tobs = dyn.tobs
Eclipse_index = SearchEclipse(start_mjd, tobs)
if Eclipse_index is not None:
    dyn.dyn[:, Eclipse_index-3:Eclipse_index+3] = 0

dyn.refill(linear=False)
dyn.plot_dyn(filename=str(outdir) + str(Filename) + '_Spectra.png',
             display=True)

# This part is for testing the new method:
nitr = 1
chisqr = np.inf

dyn.calc_acf()
dyn.calc_sspec(prewhite=False, window='hamming')

# ydata_f = np.real(np.fft.fft(dyn.acf[int(dyn.nchan):, int(dyn.nsub)]))
ydata_f = dyn.acf[int(dyn.nchan):, int(dyn.nsub)]
xdata_f = dyn.df * np.linspace(0, len(ydata_f), len(ydata_f))
# ydata_t = np.real(np.fft.fft(dyn.acf[int(dyn.nchan), int(dyn.nsub):]))
ydata_t = dyn.acf[int(dyn.nchan), int(dyn.nsub):]
xdata_t = dyn.dt * np.linspace(0, len(ydata_t), len(ydata_t))

nt = len(xdata_t)  # number of t-lag samples (along half of acf frame)
nf = len(xdata_f)
###############################################################################
# Here we are using lmft mcmc to generate the parameters for our model of 1DACF
for itr in range(nitr):
    params_tau = Parameters()
    params_tau.add('tau', value=np.random.uniform(low=0, high=2000),
                   vary=True, min=0, max=2000)
    params_tau.add('alpha', value=np.random.uniform(low=-10, high=10),
                   vary=False, min=-10, max=10)
    params_tau.add('amp', value=np.random.uniform(low=0, high=10),
                   vary=True, min=0, max=10)
    params_tau.add('wn', value=np.random.uniform(low=0, high=10), vary=True,
                   min=0, max=10)
    func_tau = Minimizer(tau_acf_model, params_tau, fcn_args=(xdata_t,
                                                              ydata_t,
                                                              None))
    results_tau = func_tau.minimize()
    func_tau = Minimizer(tau_acf_model, results_tau.params,
                         fcn_args=(xdata_t,
                                   ydata_t,
                                   None))
    mcmc_results = func_tau.emcee(steps=1000, burn=10, progress=True,
                                  is_weighted=False)
    results_tau = mcmc_results

    if results_tau.chisqr < chisqr:
        chisqr = results_tau.chisqr
        params_tau = results_tau.params
        tau_res = results_tau

for itr in range(nitr):
    params_dnu = Parameters()
    params_dnu.add('dnu', value=np.random.uniform(low=0, high=1000),
                   vary=True, min=0, max=1000)
    params_dnu.add('amp', value=np.random.uniform(low=0, high=10),
                   vary=True, min=0, max=10)
    params_dnu.add('wn', value=np.random.uniform(low=0, high=10), vary=True,
                   min=0, max=10)
    func_dnu = Minimizer(dnu_acf_model, params_dnu, fcn_args=(xdata_t,
                                                              ydata_t,
                                                              None))
    results_dnu = func_dnu.minimize()
    func_dnu = Minimizer(dnu_acf_model, results_dnu.params,
                         fcn_args=(xdata_f,
                                   ydata_f,
                                   None))
    mcmc_results = func_dnu.emcee(steps=1000, burn=10, progress=True,
                                  is_weighted=False)
    results_dnu = mcmc_results

    if results_dnu.chisqr < chisqr:
        chisqr = results_dnu.chisqr
        params_dnu = results_dnu.params
        dnu_res = results_dnu

tau_acf_model_result = \
    tau_acf_model(results_tau.params, xdata_t, ydata_t, None)  # - ydata_t
tau_acf_model_result_norm = (tau_acf_model_result -
                             np.min(tau_acf_model_result)) / \
    (np.max(tau_acf_model_result) - np.min(tau_acf_model_result))

dnu_acf_model_result = \
    dnu_acf_model(results_dnu.params, xdata_f, ydata_f, None)  # - ydata_f
dnu_acf_model_result_norm = (dnu_acf_model_result -
                             np.min(dnu_acf_model_result)) / \
    (np.max(dnu_acf_model_result) - np.min(dnu_acf_model_result))

# Plotting #
Font = 35
Size = 80*np.pi  # Determines the size of the datapoints used
font = {'size': 32}
matplotlib.rc('font', **font)

fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(1, 1, 1)
plt.plot(xdata_t, ydata_t, label='Data')
plt.plot(xdata_t, tau_acf_model_result_norm, label='Model')
ax.legend()
plt.xlabel('Time delay (s)')
plt.title('1D ACF Time Model')
plt.show()
plt.close()

fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(1, 1, 1)
plt.plot(xdata_f, ydata_f, label='Data')
plt.plot(xdata_f, dnu_acf_model_result_norm, label='Model')
ax.legend()
plt.xlabel('Frequency delay (MHz)')
plt.title('1D ACF Frequency Model')
plt.show()
plt.close()
###############################################################################
# Below is my attempt at creating a template for the sspec model fitting
# xdata_sspec_t = 1
# ydata_sspec_t = 1
# xdata_sspec_f = 1
# ydata_sspec_f = 1

# for itr in range(nitr):
#     params_tau = Parameters()
#     params_tau.add('tau', value=np.random.uniform(low=0, high=10000),
#                    vary=True, min=0, max=np.inf)
#     params_tau.add('alpha', value=np.random.uniform(low=-10, high=10),
#                    vary=False, min=-np.inf, max=np.inf)
#     params_tau.add('amp', value=np.random.uniform(low=0, high=10),
#                    vary=True, min=0, max=np.inf)
#     params_tau.add('wn', value=np.random.uniform(low=0, high=10), vary=True,
#                    min=0, max=np.inf)
#     func_tau = Minimizer(tau_sspec_model, params_tau, fcn_args=(xdata_sspec_t,
#                                                                 ydata_sspec_t))
#     results_tau = func_tau.minimize()
#     func_tau = Minimizer(tau_sspec_model, results_tau.params,
#                          fcn_args=(xdata_sspec_t,
#                                    ydata_sspec_t))
#     mcmc_results = func_tau.emcee(steps=1000, burn=10, progress=True,
#                                   is_weighted=False)
#     results_tau = mcmc_results

#     if results_tau.chisqr < chisqr:
#         chisqr = results_tau.chisqr
#         params_tau = results_tau.params
#         tau_res = results_tau

# for itr in range(nitr):
#     params_dnu = Parameters()
#     params_dnu.add('dnu', value=np.random.uniform(low=0, high=100),
#                    vary=True, min=0, max=np.inf)
#     params_dnu.add('amp', value=np.random.uniform(low=0, high=10),
#                    vary=True, min=0, max=np.inf)
#     params_dnu.add('wn', value=np.random.uniform(low=0, high=10), vary=True,
#                    min=0, max=np.inf)
#     func_dnu = Minimizer(dnu_sspec_model, params_dnu, fcn_args=(xdata_sspec_f,
#                                                                 ydata_sspec_f))
#     results_dnu = func_dnu.minimize()
#     func_dnu = Minimizer(dnu_sspec_model, results_dnu.params,
#                          fcn_args=(xdata_sspec_f,
#                                    ydata_sspec_f))
#     mcmc_results = func_dnu.emcee(steps=1000, burn=10, progress=True,
#                                   is_weighted=False)
#     results_dnu = mcmc_results

#     if results_dnu.chisqr < chisqr:
#         chisqr = results_dnu.chisqr
#         params_dnu = results_dnu.params
#         dnu_res = results_dnu

# ###############################################################################
