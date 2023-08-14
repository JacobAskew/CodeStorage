#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 14:23:36 2022
​
@author: dreardon
"""

from scintools.dynspec import Dynspec
from scintools.scint_sim import Simulation
import numpy as np
import time
import matplotlib.pyplot as plt
np.random.seed(1)


def autocorr(arr):
    mean = np.ma.mean(arr)
    std = np.ma.std(arr)
    nr, nc = np.shape(arr)
    autocorr = np.zeros((2*nr, 2*nc))
    for x in range(-nr, nr):
        for y in range(-nc, nc):
            segment = (arr[max(0, x):min(x+nr, nr),
                           max(0, y):min(y+nc, nc)] - mean) \
                    * (arr[max(0, -x):min(-x+nr, nr),
                           max(0, -y):min(-y+nc, nc)] - mean)
            numerator = np.ma.sum(segment)
            autocorr[x+nr][y+nc] = numerator / (std ** 2)
    autocorr /= np.nanmax(autocorr)
    return autocorr


# Generate some dummy data
nf = 128
ns = 128
sim = Simulation(ar=1, nf=nf, ns=ns, dlam=0.01)
dyn = Dynspec(dyn=sim)
# Compute the ACF as normal
start = time.time()
dyn.calc_acf()
end = time.time()

ds = dyn.dyn
plt.pcolormesh(ds)
plt.title('Original dynamic spectrum')
plt.show()

plt.pcolormesh(dyn.acf)
plt.title('Original ACF')
plt.clim([-0.5, 1])
plt.show()
print("time = {}".format(end-start))

start = time.time()
acf = autocorr(ds)
end = time.time()
plt.pcolormesh(acf)
plt.title('New ACF')
plt.clim([-0.5, 1])
plt.show()
print("time = {}".format(end-start))

truth = dyn.acf
plt.pcolormesh(truth - acf)
plt.title('ACF residuals')
plt.clim([-1e-2, 1e-2])
plt.colorbar()
plt.show()

print('#################')
print('NOW MASKING SOME OF THE DATA')
print('#################\n')

dyn.dyn[int(nf*0.2):int(nf*0.4), :] = 0
dyn.dyn[:, int(ns*0.3):int(ns*0.5)] = 0

# Compute the ACF as normal
start = time.time()
dyn.calc_acf()
end = time.time()

ds = np.ma.masked_where(dyn.dyn == 0, dyn.dyn)
plt.pcolormesh(ds)
plt.title('Masked dynamic spectrum')
plt.show()
#
# ds = np.ma.masked_where(dyn.dyn == 0, dyn.dyn)
# acf = autocorr(ds)

# fig = plt.subplots(figsize=(9, 9))
# plt.pcolormesh(acf)
# plt.clim([-0.5, 1])
# plt.colorbar()
# plt.title("Slow_Masked")
# plt.show()
# plt.close()
#

plt.pcolormesh(dyn.acf)
plt.title('Masked ACF')
plt.clim([-0.5, 1])
plt.show()
print("time = {}".format(end-start))

# Compute the ACF again
# dyn.refill()
# start = time.time()
# dyn.calc_acf()
# end = time.time()
#
dyn.refill()
# dyn.refill(method='median')
dyn.calc_acf()

# fig = plt.subplots(figsize=(9, 9))
# plt.pcolormesh(dyn.acf)
# plt.clim([-0.5, 1])
# plt.colorbar()
# plt.title("Median_FFT")
# plt.show()
# plt.close()
#
#
# fig = plt.subplots(figsize=(9, 9))
# plt.pcolormesh(acf-dyn.acf)
# # plt.clim([-1e-1, 1e-1])
# plt.colorbar()
# plt.title("Residual_FFTMedian_SlowMasked")
# plt.show()
# plt.close()
#

plt.pcolormesh(dyn.acf)
plt.title('Refilled ACF')
plt.clim([-0.5, 1])
plt.show()

start = time.time()
acf = autocorr(ds)
end = time.time()
plt.pcolormesh(acf)
plt.title('New ACF')
plt.clim([-0.5, 1])
plt.show()
print("time = {}".format(end-start))

plt.pcolormesh(truth - dyn.acf)
plt.title('Refilled ACF residuals')
plt.clim([-1e-1, 1e-1])
plt.colorbar()
plt.show()

plt.pcolormesh(truth - acf)
plt.title('Masked ACF residuals')
plt.clim([-1e-1, 1e-1])
plt.colorbar()
plt.show()

# sim = Simulation(ar=1, nf=nf, ns=ns, dlam=0.01, seed=64)
# dyn = Dynspec(dyn=sim)
# RFI_beg = 1400
# RFI_end = 1404
# r1 = np.argmin(abs(RFI_beg - dyn.freqs))
# r2 = np.argmin(abs(RFI_end - dyn.freqs))
# dyn.dyn[r1:r2, :] = 0

# dyn.plot_dyn()
# #
# ds = np.ma.masked_where(dyn.dyn == 0, dyn.dyn)
# acf = autocorr(ds)

# fig = plt.subplots(figsize=(9, 9))
# plt.pcolormesh(acf)
# plt.clim([-0.5, 1])
# plt.colorbar()
# plt.title("Slow_Masked")
# plt.show()
# plt.close()
# #
# #
# dyn.refill(method='median')
# dyn.calc_acf()

# fig = plt.subplots(figsize=(9, 9))
# plt.pcolormesh(dyn.acf)
# plt.clim([-0.5, 1])
# plt.colorbar()
# plt.title("Median_FFT")
# plt.show()
# plt.close()
# #
# #
# fig = plt.subplots(figsize=(9, 9))
# plt.pcolormesh(acf-dyn.acf)
# # plt.clim([-1e-1, 1e-1])
# plt.colorbar()
# plt.title("Residual_FFTMedian_SlowMasked")
# plt.show()
# plt.close()
#
# CS = ax1.contourf(t_delays, f_shifts, autocorr,
#                   levels=np.linspace(-0.5, 1, 10))
# fig.colorbar(CS)
# # plt.title("SlowACF_dynspec")
# plt.title("SlowACF_dynspec, " + str(SlowACF_dynspec_dnu) + " +/- " +
#           str(SlowACF_dynspec_dnuerr))
# plt.savefig("/Users/jacobaskew/Desktop/2DACF_Slow_dynspec.pdf", dpi=400)
# plt.savefig("/Users/jacobaskew/Desktop/2DACF_Slow_dynspec.png", dpi=400)
# plt.xlim(-1, 1)
# plt.ylim(-1, 1)
# plt.show()
# plt.close()  # 2

# fig, ax1 = plt.subplots(figsize=(9, 9))
# CS = ax1.contourf(t_delays, f_shifts, dyn_crop1.acf,
#                   levels=np.linspace(-0.5, 1, 10))
# fig.colorbar(CS)
# plt.title("FFT_refilled_median, " +
#           str(FFT_refilled_median_dnu) + " +/- " +
#           str(FFT_refilled_median_dnuerr))
# plt.xlim(-1, 1)
# plt.ylim(-1, 1)
# plt.savefig(
#     "/Users/jacobaskew/Desktop/2DACF_FFT_refilled_median.pdf",
#     dpi=400)
# plt.savefig(
#     "/Users/jacobaskew/Desktop/2DACF_FFT_refilled_median.png",
#     dpi=400)
# plt.show()
# plt.close()  # 15
