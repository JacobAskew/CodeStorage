#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 18 14:56:59 2023

@author: jacobaskew
"""
###############################################################################
# This script will test a given version of scintools in our .dev file
###############################################################################
# Imports
import sys
sys.path.insert(0, "/Users/jacobaskew/Desktop/Github/scintools_dev/scintools/")
from scintools.dynspec import Dynspec
from scintools.scint_utils import write_results, read_results, read_par, \
        float_array_from_dict, get_ssb_delay, get_earth_velocity, \
        get_true_anomaly, scint_velocity, pars_to_params, is_valid
from scintools.scint_sim import Simulation
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy as cp
###############################################################################
# A test dynamic spectra from simulations
wd = '/Users/jacobaskew/Desktop/Github/scintools_dev/scintools/scintools/examples/TestOutputs/'
sim = Simulation(seed=64)
dyn = Dynspec(dyn=sim, process=False)
dyn.plot_dyn(filename=wd+'dynspec.png')
dyn.calc_sspec()
dyn.plot_sspec(filename=wd+'sspec.png')
dyn.get_scint_params(filename=wd+'acf2d_approx.png', method='acf2d_approx', plot=True, display=True)
print("Scintbandwidth =", dyn.dnu, "+/-", dyn.dnuerr)
print("Scinttimescale =", dyn.tau, "+/-", dyn.tauerr)
dyn.get_scint_params(filename=wd+'acf1d.png', method='acf1d', plot=True, display=True)
print("Scintbandwidth =", dyn.dnu, "+/-", dyn.dnuerr)
print("Scinttimescale =", dyn.tau, "+/-", dyn.tauerr)
###############################################################################
# Testing two different models of weights

# nf, nt = np.shape(dyn.acf)
# weighted = True

# tticks = np.linspace(-dyn.tobs, dyn.tobs, nt)
# fticks = np.linspace(-dyn.bw, dyn.bw, nf)

# T, F = np.meshgrid(dyn.tobs - abs(tticks), dyn.bw - abs(fticks))
# # Create weights array
# N2d = dyn.nsub * dyn.nchan * (T/max(tticks)) * (F/max(fticks))
# errors_2d = 1/np.sqrt(N2d)
# errors_2d[~is_valid(errors_2d)] = np.inf

# weights_2d = np.ones(np.shape(dyn.acf))
# if weighted:
#     weights_2d = weights_2d / errors_2d

# plt.pcolormesh(weights_2d)
# plt.show()
# plt.close()

# #

# nf, nt = np.shape(dyn.acf)
# error_2dacf = np.ones((nf, nt)) * np.inf

# for i in range(0, nf):
#     for j in range(0, nt):
#         if ((nf/2 - abs(i - nf/2)) * (nt/2 - abs(j - nt/2))) != 0:
#             error_2dacf[i, j] = \
#                 1 / ((nf/2 - abs(i - nf/2)) * (nt/2 - abs(j - nt/2)))
# weights_2dacf = np.sqrt(1/error_2dacf)

# plt.pcolormesh(weights_2dacf)
# plt.show()
# plt.close()

# diff = weights_2dacf - weights_2d

# cb = plt.pcolormesh(diff)
# plt.colorbar(cb)
# plt.show()
# plt.close()
###############################################################################
# Testing the weighted reference frequency that takes into account RFI
sim = Simulation(seed=64, dlam=0.5)
dyn = Dynspec(dyn=sim, process=False)
dyn.plot_dyn(filename=wd+'dynspec.png')
dyn_crop = cp(dyn)
istart_t = 0
time_bin = 20
freq_bin = 100
istart_f = np.max(dyn_crop.freqs)
NUM = np.max(dyn.dyn)

# for istart_t in range(0, int(dyn_crop.tobs/60), int(time_bin)):
#     dyn_new = cp(dyn)
#     dyn_crop_time = dyn_new.crop_dyn(tmin=istart_t, tmax=istart_t+time_bin)
#     for istart_f in range(int(np.max(dyn_crop.freqs)), 0, -int(freq_bin)):
#         print(istart_f)
#         dyn_crop_freq = cp(dyn_crop_time)
#         dyn_crop_freq = dyn_crop_freq.crop_dyn(fmin=istart_f, fmax=istart_f+freq_bin)
#         dyn_crop_freq.plot_dyn()

dyn.dyn[180:220, :] = 0
dyn.dyn[10:22, :] = 0
dyn.dyn[60:120, :] = 0
dyn.plot_dyn()

index = []
indexs = np.arange(0, dyn.dyn.shape[0])
for i in range(0, dyn.dyn.shape[0]):
    if np.sum(dyn.dyn[i, :]) == 0:
        index.append(0)
    else:
        index.append(1)
index = np.asarray(index)
weighted_index = int(np.mean(indexs[np.argwhere(index > 0)]))

dyn.dyn[weighted_index, :] = NUM
dyn.plot_dyn()

###############################################################################
