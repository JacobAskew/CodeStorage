#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 22:52:35 2020

@author: jacobaskew
"""
###############################################################################
from scintools.dynspec import Dynspec
from scintools.scint_sim import Simulation
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy as cp
###############################################################################


def autocorr_func(data):
    mean = np.ma.mean(data)
    std = np.ma.std(data)
    nr, nc = np.shape(data)
    autocorr = np.zeros((2*nr, 2*nc))
    for x in range(-nr+1, nr):
        for y in range(-nc+1, nc):
            segment1 = (data[max(0, x):min(x+nr, nr),
                             max(0, y):min(y+nc, nc)])
            segment2 = (data[max(0, -x):min(-x+nr, nr),
                             max(0, -y):min(-y+nc, nc)])
            numerator = np.ma.sum(np.ma.multiply(segment1 - mean,
                                                 segment2 - mean))
            denomenator = std ** 2
            autocorr[x+nr][y+nc] = numerator / denomenator
    autocorr /= np.nanmax(autocorr)
    return autocorr


###############################################################################
# dynspecfile2 \
#     = '/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/New/'
# dynspecfile2 \
#     = dynspecfile2 \
#     + 'DataFiles/DynspecPlotFiles/2022-12-30Zap_CompleteDynspec.dynspec'

sim = Simulation(seed=64, ns=16, nf=16, freq=1000)
dyn = Dynspec(dyn=sim, process=False)
# dyn.load_file(filename=dynspecfile2)
dyn_crop2 = cp(dyn)
# dyn_crop2.crop_dyn(tmin=95, tmax=105, fmin=1000, fmax=1030)

RFI_beg = 1000
RFI_end = 1050

r1 = np.argmin(abs(RFI_beg - dyn_crop2.freqs))
r2 = np.argmin(abs(RFI_end - dyn_crop2.freqs))
dyn_crop2.dyn[r1:r2, :] = 0
dyn_crop2.plot_dyn(filename="/Users/jacobaskew/Desktop/Spectra_Zeroed.png",
                   dpi=400)  # 5
###############################################################################

tspan = dyn_crop2.tobs
fspan = dyn_crop2.bw

dyn_crop2.calc_acf()
zeroed_acf = dyn_crop2.acf
t_delays = np.linspace(-tspan/60, tspan/60, np.shape(zeroed_acf)[1])
f_shifts = np.linspace(-fspan, fspan, np.shape(zeroed_acf)[0])

#

fig, ax1 = plt.subplots(figsize=(9, 9))
CS = ax1.contourf(t_delays, f_shifts, zeroed_acf)  # ,
#                  levels=np.linspace(-1, 1, 10))
fig.colorbar(CS)
plt.title("FFT_zeroed")
# plt.xlim(-1, 1)
# plt.ylim(-1, 1)
plt.savefig(
    "/Users/jacobaskew/Desktop/2DACF_FFT_zeroed.pdf",
    dpi=400)
plt.savefig(
    "/Users/jacobaskew/Desktop/2DACF_FFT_zeroed.png",
    dpi=400)
plt.show()
plt.close()  # 15

data = np.ma.masked_where(dyn_crop2.dyn == 0, dyn_crop2.dyn)
autocorr = autocorr_func(data)

tspan2 = dyn_crop2.tobs
fspan2 = dyn_crop2.bw
t_delays2 = np.linspace(-tspan2/60, tspan2/60, np.shape(autocorr)[1])
f_shifts2 = np.linspace(-fspan2, fspan2, np.shape(autocorr)[0])

fig, ax1 = plt.subplots(figsize=(9, 9))
CS = ax1.contourf(t_delays2, f_shifts2, autocorr)  # ,
#                  levels=np.linspace(-1, 1, 10))
fig.colorbar(CS)
plt.title("SlowACF_masked")
# plt.xlim(-1, 1)
# plt.ylim(-1, 1)
plt.savefig("/Users/jacobaskew/Desktop/2DACF_SlowACF_masked.pdf", dpi=400)
plt.savefig("/Users/jacobaskew/Desktop/2DACF_SlowACF_masked.png", dpi=400)
plt.show()
plt.close()  # 6
###############################################################################
