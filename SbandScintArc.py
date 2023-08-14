#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 22:52:35 2020

@author: jacobaskew
"""
from scintools.dynspec import Dynspec
from scintools.scint_sim import Simulation
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy as cp
import matplotlib

###############################################################################
# Plotting #
Font = 35
Size = 80*np.pi  # Determines the size of the datapoints used
font = {'size': 32}
matplotlib.rc('font', **font)
par_dir = '/Users/jacobaskew/Desktop/Swinburne_Daniel/ParFiles/'

obs1 = "/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/New/Sband/HiRes/DynspecFiles/J0737-3039A_2023-05-16-11:38:48_zap.dynspec"
obs2 = "/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/New/Sband/HiRes/DynspecFiles/J0737-3039A_2023-05-16-12:08:56_zap.dynspec"
obs3 = "/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/New/Sband/HiRes/DynspecFiles/J0737-3039A_2023-05-16-14:09:04_zap.dynspec"

dyn1 = Dynspec(filename=obs1, process=False)
dyn2 = Dynspec(filename=obs2, process=False)
dyn3 = Dynspec(filename=obs3, process=False)
dyn = dyn1 + dyn2 + dyn3
dyn.plot_dyn()
dyn.trim_edges()
Fmax = np.max(dyn.freqs) - 48
Fmin = np.min(dyn.freqs) + 48
dyn.crop_dyn(fmin=Fmin, fmax=Fmax)
dyn.calc_sspec()
dyn.plot_sspec()

# Show the dynspec before process
dyn.plot_dyn(dpi=400)
# Show the sspec before process
dyn.plot_sspec(lamsteps=True, delmax=2, maxfdop=6, dpi=400)
# Rescale the dynamic spectrum in time, using a velocity model
dyn.scale_dyn(scale='lambda+velocity', s=0.71, d=0.735, inc=89.35,
              Omega=90, parfile=par_dir+'J0737-3039A.par')
# Plot the velocity-rescaled dynamic spectrum
dyn.plot_dyn(velocity=True, dpi=400)
# plot new sspec
dyn.plot_sspec(lamsteps=True, delmax=2, velocity=True, maxfdop=6, prewhite=True,
                dpi=400)
# Fit the arc using the new spectrum
dyn.fit_arc(velocity=True, cutmid=5, plot=True, weighted=False,
            lamsteps=True, subtract_artefacts=True, log_parabola=True,
            constraint=[100, 1000], high_power_diff=-0.2,
            low_power_diff=-4, delmax=2,
            etamin=1, startbin=5)
# Plot the sspec with the fit
dyn.plot_sspec(lamsteps=True, delmax=2, velocity=True, maxfdop=6, plotarc=True,
                prewhite=True, dpi=400)


lowobs1 = "/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/New/Sband/LoRes/DynspecFiles/J0737-3039A_2023-05-16-11:38:48_zap.dynspec"
lowobs2 = "/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/New/Sband/LoRes/DynspecFiles/J0737-3039A_2023-05-16-12:08:56_zap.dynspec"
lowobs3 = "/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/New/Sband/LoRes/DynspecFiles/J0737-3039A_2023-05-16-14:09:04_zap.dynspec"

low_dyn1 = Dynspec(filename=lowobs1, process=False)
low_dyn2 = Dynspec(filename=lowobs2, process=False)
low_dyn3 = Dynspec(filename=lowobs3, process=False)
low_dyn = low_dyn1 + low_dyn2 + low_dyn3
low_dyn.plot_dyn()
low_dyn.trim_edges()
Fmax = np.max(low_dyn.freqs) - 48
Fmin = np.min(low_dyn.freqs) + 48
low_dyn.crop_dyn(fmin=Fmin, fmax=Fmax)
low_dyn.calc_sspec()
low_dyn.plot_sspec()

# Show the dynspec before process
low_dyn.plot_dyn(dpi=400)
# Show the sspec before process
low_dyn.plot_sspec(lamsteps=True, delmax=2, maxfdop=6, dpi=400)
# Rescale the dynamic spectrum in time, using a velocity model
low_dyn.scale_dyn(scale='lambda+velocity', s=0.71, d=0.350, inc=89.35,
              Omega=90, parfile=par_dir+'J0737-3039A.par')
# Plot the velocity-rescaled dynamic spectrum
low_dyn.plot_dyn(velocity=True, dpi=400)
# plot new sspec
low_dyn.plot_sspec(lamsteps=True, delmax=2, velocity=True, maxfdop=6, prewhite=True,
                dpi=400)
# Fit the arc using the new spectrum
low_dyn.fit_arc(velocity=True, cutmid=5, plot=True, weighted=False,
            lamsteps=True, subtract_artefacts=True, log_parabola=True,
            constraint=[100, 1000], high_power_diff=-2,
            low_power_diff=-4, delmax=2,
            etamin=1, startbin=5)
# Plot the sspec with the fit
low_dyn.plot_sspec(lamsteps=True, delmax=2, velocity=True, maxfdop=6, plotarc=True,
                prewhite=True, dpi=400)



dyn_test = cp(low_dyn)
#
bad_freq_low = 2002
bad_freq_high = 2007.5
bad_index_low = int((bad_freq_low - np.min(dyn_test.freqs)) /
                    (dyn_test.df))
bad_index_high = int((bad_freq_high - np.min(dyn_test.freqs)) /
                     (dyn_test.df))
dyn_test.dyn[bad_index_low:bad_index_high, :] = 0

bad_freq_low = 2201.5
bad_freq_high = 2207.5
bad_index_low = int((bad_freq_low - np.min(dyn_test.freqs)) /
                    (dyn_test.df))
bad_index_high = int((bad_freq_high - np.min(dyn_test.freqs)) /
                     (dyn_test.df))
dyn_test.dyn[bad_index_low:bad_index_high, :] = 0
#
bad_time_low = 15
bad_time_high = 15.75
bad_index_time_low = int((bad_time_low*60)/(dyn_test.dt))
bad_index_time_high = int((bad_time_high*60)/(dyn_test.dt))
dyn_test.dyn[:, bad_index_time_low:bad_index_time_high] = 0

bad_time_low = 30
bad_time_high = 30.5
bad_index_time_low = int((bad_time_low*60)/(dyn_test.dt))
bad_index_time_high = int((bad_time_high*60)/(dyn_test.dt))
dyn_test.dyn[:, bad_index_time_low:bad_index_time_high] = 0

bad_time_low = 150
bad_time_high = 150.5
bad_index_time_low = int((bad_time_low*60)/(dyn_test.dt))
bad_index_time_high = int((bad_time_high*60)/(dyn_test.dt))
dyn_test.dyn[:, bad_index_time_low:bad_index_time_high] = 0

bad_time_low = 162.25
bad_time_high = 163
bad_index_time_low = int((bad_time_low*60)/(dyn_test.dt))
bad_index_time_high = int((bad_time_high*60)/(dyn_test.dt))
dyn_test.dyn[:, bad_index_time_low:bad_index_time_high] = 0

dyn_test.plot_dyn(filename="/Users/jacobaskew/Desktop/Test.png")

# dyn_test.zap()

dyn_test.refill()

dyn_test.plot_dyn(filename="/Users/jacobaskew/Desktop/Test3.png")

# L band

Llow_obs1 = "/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/New/Lband/LoRes/2022-09-18-07:10:50.XPp.dynspec"
Llow_obs2 = "/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/New/Lband/LoRes/2022-09-18-07:40:56.XPp.dynspec"
Llow_obs3 = "/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/New/Lband/LoRes/2022-09-18-09:41:05.XPp.dynspec"

Llow_dyn1 = Dynspec(filename=Llow_obs1, process=False)
Llow_dyn2 = Dynspec(filename=Llow_obs2, process=False)
Llow_dyn3 = Dynspec(filename=Llow_obs3, process=False)
Llow_dyn = Llow_dyn1 + Llow_dyn2 + Llow_dyn3
Llow_dyn.plot_dyn()
Llow_dyn.trim_edges()
Fmax = np.max(Llow_dyn.freqs) - 48
Fmin = np.min(Llow_dyn.freqs) + 48
Llow_dyn.crop_dyn(fmin=Fmin, fmax=Fmax)

Llow_dyn.plot_dyn(filename="/Users/jacobaskew/Desktop/Test.png")

dyn_test = cp(Llow_dyn)

bad_time_low = 16.5
bad_time_high = 17.25
bad_index_time_low = int((bad_time_low*60)/(dyn_test.dt))
bad_index_time_high = int((bad_time_high*60)/(dyn_test.dt))
dyn_test.dyn[:, bad_index_time_low:bad_index_time_high] = 0

bad_time_low = 170.25
bad_time_high = 171.25
bad_index_time_low = int((bad_time_low*60)/(dyn_test.dt))
bad_index_time_high = int((bad_time_high*60)/(dyn_test.dt))
dyn_test.dyn[:, bad_index_time_low:bad_index_time_high] = 0

dyn_test.zap()
dyn_test.refill()

dyn_test.plot_dyn(filename="/Users/jacobaskew/Desktop/Test4.png")


# Llow_dyn.calc_sspec()
# Llow_dyn.plot_sspec()












