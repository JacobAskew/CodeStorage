#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 22:52:35 2020

@author: jacobaskew
"""
from scintools.dynspec import Dynspec
from scintools.scint_sim import Simulation
from scintools.scint_utils import read_par, pars_to_params, get_true_anomaly, \
    centres_to_edges, is_valid
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy as cp
import matplotlib
from scipy.optimize import curve_fit 
import os
###############################################################################

desktopdir = '/Users/jacobaskew/Desktop/'
# datadir = desktopdir + 'DoublePulsar_Project/datasets/FullDataStorage/'
datadir = '/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/New/DataFiles/'
par_dir = '/Users/jacobaskew/Desktop/Swinburne_Daniel/ParFiles/'
psrname = 'J0737-3039A'
wd0 = "/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/New/"
outdir = wd0 + "Modelling"
pars = read_par(str(par_dir) + str(psrname) + '.par')
params = pars_to_params(pars)
###############################################################################
# For pavan

obs = "/Users/jacobaskew/Desktop/dynspecs/NewData/t181121_041930.rf.pcm.dynspec"
dyn = Dynspec(filename=obs, process=False)

dyn_crop = cp(dyn)
dyn_crop.crop_dyn(fmin=1280, fmax=1420)
dyn_crop.plot_dyn()
dyn_crop.get_scint_params(method="acf2d_approx", plot=True)
print(dyn_crop.dnu)
# dyn_crop.plot_dyn(filename="/Users/jacobaskew/Desktop/HiResLband1.png", dpi=400, lamsteps=False)



###
### Some data ###

## Low Res ##

# obs1 = "/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/New/Lband/LoRes/DynspecFiles/J0737-3039A_2023-06-23-07:38:56_zap.dynspec"
# obs2 = "/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/New/Lband/LoRes/DynspecFiles/J0737-3039A_2023-06-23-08:09:04_zap.dynspec"
# obs3 = "/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/New/Lband/LoRes/DynspecFiles/J0737-3039A_2023-06-23-10:09:12_zap.dynspec"
# dyn1 = Dynspec(filename=obs1, process=False)
# dyn2 = Dynspec(filename=obs2, process=False)
# dyn3 = Dynspec(filename=obs3, process=False)

# dyn = dyn1 + dyn2 + dyn3
# dyn.plot_dyn()

# dyn_crop = cp(dyn)
# dyn_crop.crop_dyn(fmin=1325, fmax=1500)
# # dyn_crop.plot_dyn()
# # dyn_crop.zap()

# bad_time_low = 15.75
# bad_time_high = 17
# bad_index_time_low = int((bad_time_low*60)/(dyn_crop.dt))
# bad_index_time_high = int((bad_time_high*60)/(dyn_crop.dt))
# dyn_crop.dyn[:, bad_index_time_low:bad_index_time_high] = 0
# #
# bad_time_low = 29.75
# bad_time_high = 30.5
# bad_index_time_low = int((bad_time_low*60)/(dyn_crop.dt))
# bad_index_time_high = int((bad_time_high*60)/(dyn_crop.dt))
# dyn_crop.dyn[:, bad_index_time_low:bad_index_time_high] = 0
# #
# bad_time_low = 138.5
# bad_time_high = 139.5
# bad_index_time_low = int((bad_time_low*60)/(dyn_crop.dt))
# bad_index_time_high = int((bad_time_high*60)/(dyn_crop.dt))
# dyn_crop.dyn[:, bad_index_time_low:bad_index_time_high] = 0
# #
# bad_time_low = 163.75
# bad_time_high = 164.75
# bad_index_time_low = int((bad_time_low*60)/(dyn_crop.dt))
# bad_index_time_high = int((bad_time_high*60)/(dyn_crop.dt))
# dyn_crop.dyn[:, bad_index_time_low:bad_index_time_high] = 0


# dyn_crop.plot_dyn(filename="/Users/jacobaskew/Desktop/LowResLband1.png", dpi=400, lamsteps=False)
# dyn_crop.plot_dyn(filename="/Users/jacobaskew/Desktop/LowResLband1.pdf", dpi=400, lamsteps=False)

# dyn_crop.refill()

# dyn_crop.plot_dyn(filename="/Users/jacobaskew/Desktop/LowResLband2.png", dpi=400, lamsteps=False)
# dyn_crop.plot_dyn(filename="/Users/jacobaskew/Desktop/LowResLband2.pdf", dpi=400, lamsteps=False)

## High Res L band ##

obs1 = "/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/New/Lband/HiRes/DynspecFiles/J0737-3039A_2023-06-23-07:38:56_zap.dynspec"
obs2 = "/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/New/Lband/HiRes/DynspecFiles/J0737-3039A_2023-06-23-08:08:56_zap.dynspec"
obs3 = "/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/New/Lband/HiRes/DynspecFiles/J0737-3039A_2023-06-23-10:09:12_zap.dynspec"

dyn1 = Dynspec(filename=obs1, process=False)
dyn2 = Dynspec(filename=obs2, process=False)
dyn3 = Dynspec(filename=obs3, process=False)

dyn = dyn1 + dyn2 + dyn3
dyn.plot_dyn()

dyn_crop = cp(dyn)
dyn_crop.crop_dyn(fmin=1325, fmax=1500)
dyn_crop.plot_dyn(filename="/Users/jacobaskew/Desktop/HiResLband1.png", dpi=400, lamsteps=False)
dyn_crop.plot_dyn(filename="/Users/jacobaskew/Desktop/HiResLband1.pdf", dpi=400, lamsteps=False)

bad_time_low = 15.75
bad_time_high = 17
bad_index_time_low = int((bad_time_low*60)/(dyn_crop.dt))
bad_index_time_high = int((bad_time_high*60)/(dyn_crop.dt))
dyn_crop.dyn[:, bad_index_time_low:bad_index_time_high] = 0
#
bad_time_low = 29.75
bad_time_high = 30.5
bad_index_time_low = int((bad_time_low*60)/(dyn_crop.dt))
bad_index_time_high = int((bad_time_high*60)/(dyn_crop.dt))
dyn_crop.dyn[:, bad_index_time_low:bad_index_time_high] = 0
#
bad_time_low = 138.5
bad_time_high = 139.5
bad_index_time_low = int((bad_time_low*60)/(dyn_crop.dt))
bad_index_time_high = int((bad_time_high*60)/(dyn_crop.dt))
dyn_crop.dyn[:, bad_index_time_low:bad_index_time_high] = 0
#
bad_time_low = 163.75
bad_time_high = 164.75
bad_index_time_low = int((bad_time_low*60)/(dyn_crop.dt))
bad_index_time_high = int((bad_time_high*60)/(dyn_crop.dt))
dyn_crop.dyn[:, bad_index_time_low:bad_index_time_high] = 0

dyn_crop.zap()
dyn_crop.refill()
dyn_crop.plot_dyn(filename="/Users/jacobaskew/Desktop/HiResLband2.png", dpi=400, lamsteps=False)
dyn_crop.plot_dyn(filename="/Users/jacobaskew/Desktop/HiResLband2.pdf", dpi=400, lamsteps=False)

Ldyn_crop = cp(dyn_crop)

Lband_start = dyn_crop.mjd
min_per_phasedeg = (pars['PB'] * 1440) / 360
###############################################################################
# Low res Sband

Sobs1 = "/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/New/Sband/LoRes/DynspecFiles/J0737-3039A_2023-05-16-11:38:48_zap.dynspec"
Sobs2 = "/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/New/Sband/LoRes/DynspecFiles/J0737-3039A_2023-05-16-12:08:56_zap.dynspec"
Sobs3 = "/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/New/Sband/LoRes/DynspecFiles/J0737-3039A_2023-05-16-14:09:04_zap.dynspec"

Sdyn1 = Dynspec(filename=Sobs1, process=False)
Sdyn2 = Dynspec(filename=Sobs2, process=False)
Sdyn3 = Dynspec(filename=Sobs3, process=False)

Sdyn = Sdyn1 + Sdyn2 + Sdyn3
Sdyn.plot_dyn()

Sdyn_crop = cp(Sdyn)
Sdyn_crop.crop_dyn(fmin=2275, fmax=2475)
Sdyn_crop.plot_dyn(filename="/Users/jacobaskew/Desktop/LowResSband1.png", dpi=400, lamsteps=False)
Sdyn_crop.plot_dyn(filename="/Users/jacobaskew/Desktop/LowResSband1.pdf", dpi=400, lamsteps=False)

bad_time_low = 15
bad_time_high = 15.75
bad_index_time_low = int((bad_time_low*60)/(Sdyn_crop.dt))
bad_index_time_high = int((bad_time_high*60)/(Sdyn_crop.dt))
Sdyn_crop.dyn[:, bad_index_time_low:bad_index_time_high] = 0
#
bad_time_low = 29.5
bad_time_high = 30.25
bad_index_time_low = int((bad_time_low*60)/(Sdyn_crop.dt))
bad_index_time_high = int((bad_time_high*60)/(Sdyn_crop.dt))
Sdyn_crop.dyn[:, bad_index_time_low:bad_index_time_high] = 0
#
bad_time_low = 149.75
bad_time_high = 150.25
bad_index_time_low = int((bad_time_low*60)/(Sdyn_crop.dt))
bad_index_time_high = int((bad_time_high*60)/(Sdyn_crop.dt))
Sdyn_crop.dyn[:, bad_index_time_low:bad_index_time_high] = 0
#
bad_time_low = 162.25
bad_time_high = 163.5
bad_index_time_low = int((bad_time_low*60)/(Sdyn_crop.dt))
bad_index_time_high = int((bad_time_high*60)/(Sdyn_crop.dt))
Sdyn_crop.dyn[:, bad_index_time_low:bad_index_time_high] = 0

Sdyn_crop.zap()
Sdyn_crop.refill()
Sdyn_crop.plot_dyn(filename="/Users/jacobaskew/Desktop/LowResSband2.png", dpi=400, lamsteps=False)
Sdyn_crop.plot_dyn(filename="/Users/jacobaskew/Desktop/LowResSband2.pdf", dpi=400, lamsteps=False)

Sband_start = Sdyn_crop.mjd


## Hi res Sband ##

# Sobs1 = "/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/New/Sband/HiRes/DynspecFiles/J0737-3039A_2023-05-16-11:38:48_zap.dynspec"
# Sobs2 = "/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/New/Sband/HiRes/DynspecFiles/J0737-3039A_2023-05-16-12:08:56_zap.dynspec"
# Sobs3 = "/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/New/Sband/HiRes/DynspecFiles/J0737-3039A_2023-05-16-14:09:04_zap.dynspec"

# Sdyn1 = Dynspec(filename=Sobs1, process=False)
# Sdyn2 = Dynspec(filename=Sobs2, process=False)
# Sdyn3 = Dynspec(filename=Sobs3, process=False)

# Sdyn = Sdyn1 + Sdyn2 + Sdyn3
# Sdyn.plot_dyn()

# Sdyn_crop = cp(Sdyn)
# Sdyn_crop.crop_dyn(fmin=2300, fmax=2500)
# Sdyn_crop.plot_dyn(filename="/Users/jacobaskew/Desktop/HiResSband1.png", dpi=400, lamsteps=False)
# Sdyn_crop.plot_dyn(filename="/Users/jacobaskew/Desktop/HiResSband1.pdf", dpi=400, lamsteps=False)

# bad_time_low = 15
# bad_time_high = 15.75
# bad_index_time_low = int((bad_time_low*60)/(Sdyn_crop.dt))
# bad_index_time_high = int((bad_time_high*60)/(Sdyn_crop.dt))
# Sdyn_crop.dyn[:, bad_index_time_low:bad_index_time_high] = 0
# #
# bad_time_low = 29.5
# bad_time_high = 30.25
# bad_index_time_low = int((bad_time_low*60)/(Sdyn_crop.dt))
# bad_index_time_high = int((bad_time_high*60)/(Sdyn_crop.dt))
# Sdyn_crop.dyn[:, bad_index_time_low:bad_index_time_high] = 0
# #
# bad_time_low = 149.75
# bad_time_high = 150.25
# bad_index_time_low = int((bad_time_low*60)/(Sdyn_crop.dt))
# bad_index_time_high = int((bad_time_high*60)/(Sdyn_crop.dt))
# Sdyn_crop.dyn[:, bad_index_time_low:bad_index_time_high] = 0
# #
# bad_time_low = 162.25
# bad_time_high = 163.5
# bad_index_time_low = int((bad_time_low*60)/(Sdyn_crop.dt))
# bad_index_time_high = int((bad_time_high*60)/(Sdyn_crop.dt))
# Sdyn_crop.dyn[:, bad_index_time_low:bad_index_time_high] = 0

# Sdyn_crop.zap()
# Sdyn_crop.refill()
# Sdyn_crop.plot_dyn(filename="/Users/jacobaskew/Desktop/HiResSband2.png", dpi=400, lamsteps=False)
# Sdyn_crop.plot_dyn(filename="/Users/jacobaskew/Desktop/HiResSband2.pdf", dpi=400, lamsteps=False)

# Sband_start = Sdyn_crop.mjd



###############################################################################

Uobs1 = "/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/New/Dynspec/J0737-3039A_2022-07-30-04:43:36_zap.dynspec"
Uobs2 = "/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/New/Dynspec/J0737-3039A_2022-07-30-05:13:44_zap.dynspec"
Uobs3 = "/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/New/Dynspec/J0737-3039A_2022-07-30-07:13:52_zap.dynspec"

Udyn1 = Dynspec(filename=Uobs1, process=False)
Udyn2 = Dynspec(filename=Uobs2, process=False)
Udyn3 = Dynspec(filename=Uobs3, process=False)

Udyn = Udyn1 + Udyn2 + Udyn3
Udyn.plot_dyn()

Udyn_crop = cp(Udyn)
Udyn_crop.crop_dyn(fmin=725, fmax=925)
Udyn_crop.plot_dyn(filename="/Users/jacobaskew/Desktop/HiResUHF1.png", dpi=400, lamsteps=False)
Udyn_crop.plot_dyn(filename="/Users/jacobaskew/Desktop/HiResUHF1.pdf", dpi=400, lamsteps=False)
#
bad_freq_low = 935
bad_freq_high = 949
bad_index_low = int((bad_freq_low - np.min(Udyn_crop.freqs)) /
                    (Udyn_crop.df))
bad_index_high = int((bad_freq_high - np.min(Udyn_crop.freqs)) /
                     (Udyn_crop.df))
Udyn_crop.dyn[bad_index_low:bad_index_high, :] = 0
#
bad_freq_low = 1028
bad_freq_high = 1031
bad_index_low = int((bad_freq_low - np.min(Udyn_crop.freqs)) /
                    (Udyn_crop.df))
bad_index_high = int((bad_freq_high - np.min(Udyn_crop.freqs)) /
                     (Udyn_crop.df))
Udyn_crop.dyn[bad_index_low:bad_index_high, :] = 0
#
bad_freq_low = 955
bad_freq_high = 960
bad_index_low = int((bad_freq_low - np.min(Udyn_crop.freqs)) /
                    (Udyn_crop.df))
bad_index_high = int((bad_freq_high - np.min(Udyn_crop.freqs)) /
                     (Udyn_crop.df))
Udyn_crop.dyn[bad_index_low:bad_index_high, :] = 0
#
bad_time_low = 17.5
bad_time_high = 18.5
bad_index_time_low = int((bad_time_low*60)/(Udyn_crop.dt))
bad_index_time_high = int((bad_time_high*60)/(Udyn_crop.dt))
Udyn_crop.dyn[:, bad_index_time_low:bad_index_time_high] = 0
#
bad_time_low = 30
bad_time_high = 30.5
bad_index_time_low = int((bad_time_low*60)/(Udyn_crop.dt))
bad_index_time_high = int((bad_time_high*60)/(Udyn_crop.dt))
Udyn_crop.dyn[:, bad_index_time_low:bad_index_time_high] = 0
#
bad_time_low = 150.25
bad_time_high = 151
bad_index_time_low = int((bad_time_low*60)/(Udyn_crop.dt))
bad_index_time_high = int((bad_time_high*60)/(Udyn_crop.dt))
Udyn_crop.dyn[:, bad_index_time_low:bad_index_time_high] = 0
#
bad_time_low = 165
bad_time_high = 166
bad_index_time_low = int((bad_time_low*60)/(Udyn_crop.dt))
bad_index_time_high = int((bad_time_high*60)/(Udyn_crop.dt))
Udyn_crop.dyn[:, bad_index_time_low:bad_index_time_high] = 0
#
Udyn_crop.zap()
Udyn_crop.refill()
Udyn_crop.plot_dyn(filename="/Users/jacobaskew/Desktop/HiResUHF2.png", dpi=400, lamsteps=False)
Udyn_crop.plot_dyn(filename="/Users/jacobaskew/Desktop/HiResUHF2.pdf", dpi=400, lamsteps=False)

Uband_start = Udyn_crop.mjd

###############################################################################
# L band

mjds_test = np.asarray([Lband_start, Lband_start+1])
U = get_true_anomaly(mjds_test, pars)
om = pars['OM'] + pars['OMDOT']*(mjds_test - pars['T0'])/365.2425
phase = U*180/np.pi + om
phase = phase % 360
Lband_phase = phase[0]

Ldyn_crop2 = cp(dyn_crop)

# Ldiff_min = ((dyn_crop.tobs / 60) / (params['PB'].value * 1440) - 1) * (params['PB'].value * 1440)

# S band

mjds_test = np.asarray([Sband_start, Sband_start+1])
U = get_true_anomaly(mjds_test, pars)
om = pars['OM'] + pars['OMDOT']*(mjds_test - pars['T0'])/365.2425
phase = U*180/np.pi + om
phase = phase % 360
Sband_phase = phase[0]

Sdiff_min = (Lband_phase - Sband_phase) * min_per_phasedeg

Sdyn_crop2 = cp(Sdyn_crop)


# UHF

# Udiff_min = ((Lband_start - Uband_start) - int((Lband_start - Uband_start))) * 1440
# Udiff_min = (((Lband_start - Uband_start) - int((Lband_start - Uband_start))) / params['PB'].value - int(((Lband_start - Uband_start) - int((Lband_start - Uband_start))) / params['PB'].value)) * params['PB'].value * 1440
mjds_test = np.asarray([Uband_start, Uband_start+1])
U = get_true_anomaly(mjds_test, pars)
om = pars['OM'] + pars['OMDOT']*(mjds_test - pars['T0'])/365.2425
phase = U*180/np.pi + om
phase = phase % 360
Uband_phase = phase[0]

Udiff_min = (Lband_phase - Uband_phase) * min_per_phasedeg
Udyn_crop2 = cp(Udyn_crop)

# Together calcs and plots

Sdyn_crop2.crop_dyn(tmin=Sdiff_min)
Udyn_crop2.crop_dyn(tmin=Udiff_min, tmax=Sdyn_crop2.tobs)
Ldyn_crop2.crop_dyn(tmax=Sdyn_crop2.tobs)

Sdyn_crop2.plot_dyn(filename="/Users/jacobaskew/Desktop/HiResSband3.png", figsize=(18, 9), dpi=400, lamsteps=False)
Sdyn_crop2.plot_dyn(filename="/Users/jacobaskew/Desktop/HiResSband3.pdf", figsize=(18, 9), dpi=400, lamsteps=False)

Ldyn_crop2.plot_dyn(filename="/Users/jacobaskew/Desktop/HiResLband3.png", figsize=(18, 9), dpi=400, lamsteps=False)
Ldyn_crop2.plot_dyn(filename="/Users/jacobaskew/Desktop/HiResLband3.pdf", figsize=(18, 9), dpi=400, lamsteps=False)

Udyn_crop2.plot_dyn(filename="/Users/jacobaskew/Desktop/HiResUHF3.png", figsize=(18, 9), dpi=400, lamsteps=False)
Udyn_crop2.plot_dyn(filename="/Users/jacobaskew/Desktop/HiResUHF3.pdf", figsize=(18, 9), dpi=400, lamsteps=False)

###############################################################################
#
# Smjds = Sdyn_crop.mjd + (Sdyn_crop.times / 86400)
Stimes_days = (Sdyn_crop.times / 86400)
Smjds = Stimes_days + Sdyn_crop.mjd
U = get_true_anomaly(Smjds, pars)
om = pars['OM'] + pars['OMDOT']*(Smjds - pars['T0'])/365.2425
Sband_phase = U*180/np.pi + om
Sband_phase = Sband_phase % 360

Stedges = centres_to_edges(Sband_phase)
Stedges = Stedges[np.argsort(Stedges)]
Sfedges = centres_to_edges(Sdyn_crop.freqs)
medval = np.median(Sdyn_crop.dyn[is_valid(Sdyn_crop.dyn)*np.array(np.abs(is_valid(Sdyn_crop.dyn)) > 0)])
minval = np.min(Sdyn_crop.dyn[is_valid(Sdyn_crop.dyn)*np.array(np.abs(is_valid(Sdyn_crop.dyn)) > 0)])
std = np.std(Sdyn_crop.dyn[is_valid(Sdyn_crop.dyn)*np.array(np.abs(is_valid(Sdyn_crop.dyn)) > 0)])
Svmin = minval + std
Svmax = medval + 4*std
#
Lmjds = Ldyn_crop.mjd + (Ldyn_crop.times / 86400)
LU = get_true_anomaly(Lmjds, pars)
Lom = pars['OM'] + pars['OMDOT']*(Lmjds - pars['T0'])/365.2425
Lband_phase = LU*180/np.pi + Lom
Lband_phase = Lband_phase % 360

Ltedges = centres_to_edges(Lband_phase)
Ltedges = Ltedges[np.argsort(Ltedges)]
Lfedges = centres_to_edges(Ldyn_crop.freqs)
medval = np.median(Ldyn_crop.dyn[is_valid(Ldyn_crop.dyn)*np.array(np.abs(is_valid(Ldyn_crop.dyn)) > 0)])
minval = np.min(Ldyn_crop.dyn[is_valid(Ldyn_crop.dyn)*np.array(np.abs(is_valid(Ldyn_crop.dyn)) > 0)])
std = np.std(Ldyn_crop.dyn[is_valid(Ldyn_crop.dyn)*np.array(np.abs(is_valid(Ldyn_crop.dyn)) > 0)])
Lvmin = minval + std
Lvmax = medval + 4*std

#
Umjds = Udyn_crop.mjd + (Udyn_crop.times / 86400)
U = get_true_anomaly(Umjds, pars)
om = pars['OM'] + pars['OMDOT']*(Umjds - pars['T0'])/365.2425
Uband_phase = U*180/np.pi + om
Uband_phase = Uband_phase % 360

Utedges = centres_to_edges(Uband_phase)
Utedges = Utedges[np.argsort(Utedges)]
Ufedges = centres_to_edges(Udyn_crop.freqs)
medval = np.median(Udyn_crop.dyn[is_valid(Udyn_crop.dyn)*np.array(np.abs(is_valid(Udyn_crop.dyn)) > 0)])
minval = np.min(Udyn_crop.dyn[is_valid(Udyn_crop.dyn)*np.array(np.abs(is_valid(Udyn_crop.dyn)) > 0)])
# standard deviation
std = np.std(Udyn_crop.dyn[is_valid(Udyn_crop.dyn)*np.array(np.abs(is_valid(Udyn_crop.dyn)) > 0)])
Uvmin = minval + std
Uvmax = medval + 4*std

##
# band_phase = np.concatenate(((np.concatenate((Sband_phase, Lband_phase))), Uband_phase))   
# freqs_total = np.concatenate(((np.concatenate((Sdyn_crop.freqs, Ldyn_crop.freqs))), Udyn_crop.freqs))   
# dyn_total = np.concatenate(((np.concatenate((Sdyn_crop.dyn, Ldyn_crop.dyn))), Udyn_crop.dyn))
# #
# tedges = centres_to_edges(band_phase)
# tedges = tedges[np.argsort(tedges)]
# fedges = centres_to_edges(freqs_total)
# medval = np.median(dyn_total[is_valid(dyn_total)*np.array(np.abs(is_valid(dyn_total)) > 0)])
# minval = np.min(dyn_total[is_valid(dyn_total)*np.array(np.abs(is_valid(dyn_total)) > 0)])
# # standard deviation
# std = np.std(dyn_total[is_valid(dyn_total)*np.array(np.abs(is_valid(dyn_total)) > 0)])
# vmin = minval + std
# vmax = medval + 4*std
##

Font = 35
Size = 80*np.pi  # Determines the size of the datapoints used
font = {'size': 32}
matplotlib.rc('font', **font)
matplotlib.rcParams['axes.linewidth'] = 3
##
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(18, 27))
#
ax1.pcolormesh(Stedges, Sfedges, Sdyn_crop.dyn, vmin=Svmin,
               vmax=Svmax, linewidth=0, rasterized=True,
               shading='auto')
#
ax2.pcolormesh(Ltedges, Lfedges, Ldyn_crop.dyn, vmin=Lvmin,
               vmax=Lvmax, linewidth=0, rasterized=True,
               shading='auto')
#
ax3.pcolormesh(Utedges, Ufedges, Udyn_crop.dyn, vmin=Uvmin,
               vmax=Uvmax, linewidth=0, rasterized=True,
               shading='auto')
#
fig.text(0.03, 0.5, 'Frequency (MHz)', va='center', rotation='vertical')
plt.xlabel('Orbital Phase (deg)')
plt.subplots_adjust(hspace=0.025)
plt.savefig("/Users/jacobaskew/Desktop/AllDynspec.pdf", dpi=400)
plt.savefig("/Users/jacobaskew/Desktop/AllDynspec.png", dpi=400)
plt.show()
plt.close()
##

##
# fig = plt.figure(figsize=(20, 10))
# fig.subplots_adjust(hspace=0.5, wspace=0.5)
# ax = fig.add_subplot(1, 1, 1)
# #
# ax.pcolormesh(tedges, fedges, dyn_total, vmin=vmin, vmax=vmax, linewidth=2,
#               rasterized=True, shading='auto')
# plt.xlabel('Orbital Phase (deg)')
# plt.ylabel('Observational Frequency (MHz)')
# plt.savefig("/Users/jacobaskew/Desktop/AllDynspecOneFig.pdf", dpi=400)
# plt.savefig("/Users/jacobaskew/Desktop/AllDynspecOneFig.png", dpi=400)
# plt.show()
# plt.close()
##

###############################################################################
def func(xdata, ydata, alpha):
    ref_ydata = np.median(ydata)
    ref_xdata = np.median(xdata)
    model = ref_ydata * ((xdata)/(ref_xdata))**alpha
    return model


dirname = '/Users/jacobaskew/Desktop/Anisotropy/'
ARs = np.asarray([1.0, 1.5, 2.0, 2.5, 3.0, 5.0])
psis = np.asarray([0, 30, 60, 90, 120, 150, 180])
Dnus = []
Dnuerrs = []
Freqs = []
Alphas = []
Alphaerrs = []
for i in range(0, 6):
    for ii in range(0, 7):
        if i == 0 and ii > 0:
            break
        dynspecloc = dirname+'dynspec_AR'+str(ARs[i])+'_Psi'+str(psis[ii])+'.dynspec'
        if os.path.exists(dynspecloc):
            sim = Simulation()
            dyn = Dynspec(dyn=sim, process=False)
            dyn.load_file(filename=dynspecloc)            
            fslice = (dyn.bw / 5)
            fmax = np.max(dyn.freqs)
            fmin = np.min(dyn.freqs)
            dnuA = np.ones((5))
            dnuerrA = np.ones((5))
            freqA = np.ones((5))
            Fmax1 = fmax - (fslice * 0)
            Fmin1 = fmax - (fslice * 1)
            Fmax2 = fmax - (fslice * 1)
            Fmin2 = fmax - (fslice * 2)
            Fmax3 = fmax - (fslice * 2)
            Fmin3 = fmax - (fslice * 3)
            Fmax4 = fmax - (fslice * 3)
            Fmin4 = fmax - (fslice * 4)
            Fmax5 = fmax - (fslice * 4)
            Fmin5 = fmax - (fslice * 5)
            dyn1 = cp(dyn)
            dyn1.crop_dyn(fmin=Fmin1, fmax=Fmax1)
            dyn1.get_scint_params(method='acf2d_approx', plot=False)
            dnuA[0] *= dyn1.dnu
            dnuerrA[0] *= dyn1.dnuerr
            freqA[0] *= dyn1.freq
            dyn2 = cp(dyn)
            dyn2.crop_dyn(fmin=Fmin2, fmax=Fmax2)
            dyn2.get_scint_params(method='acf2d_approx', plot=False)
            dnuA[1] *= dyn2.dnu
            dnuerrA[1] *= dyn2.dnuerr
            freqA[1] *= dyn2.freq
            dyn3 = cp(dyn)
            dyn3.crop_dyn(fmin=Fmin3, fmax=Fmax3)
            dyn3.get_scint_params(method='acf2d_approx', plot=False)
            dnuA[2] *= dyn3.dnu
            dnuerrA[2] *= dyn3.dnuerr
            freqA[2] *= dyn3.freq
            dyn4 = cp(dyn)
            dyn4.crop_dyn(fmin=Fmin4, fmax=Fmax4)
            dyn4.get_scint_params(method='acf2d_approx', plot=False)
            dnuA[3] *= dyn4.dnu
            dnuerrA[3] *= dyn4.dnuerr
            freqA[3] *= dyn4.freq
            dyn5 = cp(dyn)
            dyn5.crop_dyn(fmin=Fmin5, fmax=Fmax5)
            dyn5.get_scint_params(method='acf2d_approx', plot=False)
            dnuA[4] *= dyn5.dnu
            dnuerrA[4] *= dyn5.dnuerr
            freqA[4] *= dyn5.freq
            popt, pcov = curve_fit(func, freqA, dnuA)
            Alphas.append(popt[0])
            Alphaerrs.append(np.sqrt(np.diag(pcov))[0])
            Dnus.append(dnuA)
            Dnuerrs.append(dnuerrA)
            Freqs.append(freqA)
            break
        else:
            mb2 = 250
            dnu_diff = np.inf
            for x in range(0, 1000):
                sim = Simulation(mb2=mb2, rf=0.5, freq=815, dlam=0.6674846625766871,
                                 nf=1024, ar=ARs[i], psi=psis[ii], seed=64)
                dyn = Dynspec(dyn=sim, process=False)
                dyn2 = cp(dyn)
                dyn2.crop_dyn(fmin=1000)
                dyn2.get_scint_params(method='acf2d_approx', plot=False)
                dnu_Mdiff = dyn2.dnu - 5.4
                init_diff = abs(dnu_diff - dnu_Mdiff)
                if init_diff < 1:
                    sim = Simulation(mb2=mb2, rf=0.5, freq=815, dlam=0.6674846625766871,
                                     nf=16000, ar=ARs[i], psi=psis[ii], seed=64)
                    dyn = Dynspec(dyn=sim, process=False)
                    dyn.write_file(filename=dynspecloc)
                    fslice = (dyn.bw / 5)
                    fmax = np.max(dyn.freqs)
                    fmin = np.min(dyn.freqs)
                    dnuA = np.ones((5))
                    dnuerrA = np.ones((5))
                    freqA = np.ones((5))
                    Fmax1 = fmax - (fslice * 0)
                    Fmin1 = fmax - (fslice * 1)
                    Fmax2 = fmax - (fslice * 1)
                    Fmin2 = fmax - (fslice * 2)
                    Fmax3 = fmax - (fslice * 2)
                    Fmin3 = fmax - (fslice * 3)
                    Fmax4 = fmax - (fslice * 3)
                    Fmin4 = fmax - (fslice * 4)
                    Fmax5 = fmax - (fslice * 4)
                    Fmin5 = fmax - (fslice * 5)
                    dyn1 = cp(dyn)
                    dyn1.crop_dyn(fmin=Fmin1, fmax=Fmax1)
                    dyn1.get_scint_params(method='acf2d_approx', plot=False)
                    dnuA[0] *= dyn1.dnu
                    dnuerrA[0] *= dyn1.dnuerr
                    freqA[0] *= dyn1.freq
                    dyn2 = cp(dyn)
                    dyn2.crop_dyn(fmin=Fmin2, fmax=Fmax2)
                    dyn2.get_scint_params(method='acf2d_approx', plot=False)
                    dnuA[1] *= dyn2.dnu
                    dnuerrA[1] *= dyn2.dnuerr
                    freqA[1] *= dyn2.freq
                    dyn3 = cp(dyn)
                    dyn3.crop_dyn(fmin=Fmin3, fmax=Fmax3)
                    dyn3.get_scint_params(method='acf2d_approx', plot=False)
                    dnuA[2] *= dyn3.dnu
                    dnuerrA[2] *= dyn3.dnuerr
                    freqA[2] *= dyn3.freq
                    dyn4 = cp(dyn)
                    dyn4.crop_dyn(fmin=Fmin4, fmax=Fmax4)
                    dyn4.get_scint_params(method='acf2d_approx', plot=False)
                    dnuA[3] *= dyn4.dnu
                    dnuerrA[3] *= dyn4.dnuerr
                    freqA[3] *= dyn4.freq
                    dyn5 = cp(dyn)
                    dyn5.crop_dyn(fmin=Fmin5, fmax=Fmax5)
                    dyn5.get_scint_params(method='acf2d_approx', plot=False)
                    dnuA[4] *= dyn5.dnu
                    dnuerrA[4] *= dyn5.dnuerr
                    freqA[4] *= dyn5.freq
                    dyn.plot_dyn(filename=dirname+'dynspec_AR'+str(ARs[i])+'_Psi'+str(psis[ii])+'.png',
                                 dpi=400)
                    popt, pcov = curve_fit(func, freqA, dnuA)
                    Alphas.append(popt[0])
                    Alphaerrs.append(np.sqrt(np.diag(pcov))[0])
                    Dnus.append(dnuA)
                    Dnuerrs.append(dnuerrA)
                    Freqs.append(freqA)
                    break
                elif dnu_Mdiff < 0:
                    mb2 -= 5
                    dnu_diff = dyn2.dnu - 5.4
                elif dnu_Mdiff > 0:
                    mb2 += 5
                    dnu_diff = dyn2.dnu - 5.4


plt.scatter(Freqs, Dnus)
plt.errorbar(Freqs, Dnus, yerr=Dnuerrs, fmt=' ')
plt.show()
plt.close()


dyn.get_scint_params(method='acf2d_approx', plot=True)
dyn.get_scint_params(method='acf1d', plot=True)
# Plotting #
Font = 35
Size = 80*np.pi  # Determines the size of the datapoints used
font = {'size': 32}
matplotlib.rc('font', **font)

# nt = np.shape(dyn.acf)[1]
# nf = np.shape(dyn.acf)[0]

# t_delays \
#     = np.linspace(-dyn.tobs/60, dyn.tobs/60, nt)[int(nt/2):]*60
# f_shifts = np.linspace(-dyn.bw, dyn.bw, nf)[int(nf/2):]
# acf_error = np.std(dyn.acf)

# time_1Dslice = np.average(dyn.acf[int(nf/2), int(nt/2):],
#                           dyn.acf[int(nf/2), :int(nt/2)])
# freq_1Dslice = np.average(dyn.acf[int(nf/2):, int(nt/2)],
#                           dyn.acf[:int(nf/2), int(nt/2)])

# t_errors = 2/np.pi * np.arctan(time_1Dslice / dyn.tau) / \
#     np.sqrt(dyn.nsub)
# t_errors[t_errors == 0] = 1e-3
# f_errors = 2/np.pi * np.arctan(freq_1Dslice / dyn.dnu) / \
#     np.sqrt(dyn.nchan)
# f_errors[f_errors == 0] = 1e-3

# fig = plt.figure(figsize=(18, 9))
# fig.subplots_adjust(hspace=0.5, wspace=0.5)
# #
# ax1 = plt.subplot(2, 1, 1)
# ax1.scatter(t_delays, time_1Dslice, c='C0', alpha=0.3)
# ax1.fill_between(t_delays, time_1Dslice-t_errors, time_1Dslice+t_errors,
#                  color='C0', alpha=0.2)
# ax1.set_xlabel(r'$\tau_d$ (s)', ha='center')
# ax1.set_ylim([np.min(time_1Dslice), 1])
# ax1.set_xlim(t_delays[0], t_delays[255])
# #
# plt.ylabel('ACF                ')
# #
# ax2 = plt.subplot(2, 1, 2)
# ax2.scatter(f_shifts, freq_1Dslice, c='C0', alpha=0.3)
# ax2.fill_between(f_shifts, freq_1Dslice-f_errors, freq_1Dslice+f_errors,
#                  color='C0', alpha=0.2)
# ax2.set_xlabel(r'$\Delta\nu_d$ (MHz)', ha='center')
# ax2.set_ylim([np.min(freq_1Dslice), 1])
# ax2.set_xlim(f_shifts[0], f_shifts[255])
# #
# plt.ylabel('               Normalised')
