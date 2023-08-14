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

###############################################################################

# This section creates the simulated spectra that we can play with #

# mb2 = 0.773*(1100/1)**(5/6)
# resolution = 0.1
# sim = False

# filename0 = "/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/"
# filename1 = "Simulation/Dynspec/SimDynspec_"
# filename = filename0 + filename1

# # if not '/Users/jacobaskew/Desktop/SimSpectra_'+str(resolution)+'.png':
# sim = Simulation(mb2=mb2, rf=1, ds=0.01, alpha=5/3, ar=1, psi=0,
#                  inner=0.001, ns=128, nf=256,
#                  dlam=0.0375, lamsteps=False, seed=64, nx=None,
#                  ny=None, dx=None, dy=None, plot=False, verbose=False,
#                  freq=800, dt=8, mjd=50000, nsub=None, efield=False,
#                  noise=None)
# dyn_initial = Dynspec(dyn=sim, process=False)
# dyn_initial.trim_edges()
# dyn_initial.plot_dyn(filename='/Users/jacobaskew/Desktop/SimSpectra_' +
#                      str(resolution) + '.png', dpi=400)
# dyn_initial.write_file(filename=filename + str(resolution) + '.dynspec')
# else:
#     sim = Simulation()
#     dyn_initial = Dynspec(dyn=sim, process=False)
#     dyn_initial.load_file(filename=filename + str(resolution) + '.dynspec')
#     dyn_initial.trim_edges()
#     dyn_initial.plot_dyn(filename='/Users/jacobaskew/Desktop/SimSpectra_' +
#                          str(resolution) + '.png', dpi=400)

###############################################################################


def autocorr_func(data):
    mean = np.ma.mean(data)
    std = np.ma.std(data)
    nr, nc = np.shape(data)
    autocorr = np.zeros((2*nr, 2*nc))
    for x in range(-nr+1, nr):
        # if x == -nr:
        #     x = 0
        for y in range(-nc+1, nc):
            # if y == -nc:
            #     y = 0
            segment = (data[max(0, x):min(x+nr, nr),
                            max(0, y):min(y+nc, nc)] - mean) \
                * (data[max(0, -x):min(-x+nr, nr),
                        max(0, -y):min(-y+nc, nc)] - mean)
            numerator = np.ma.sum(segment)
            autocorr[x+nr][y+nc] = numerator / (std ** 2)
    autocorr /= np.nanmax(autocorr)
    return autocorr


# if np.isnan(autocorr[x+nr][y+nc]) and y != -75:
# if y == 75:
#     print("X", x)
#     print("Y", y)
#     print("segment", segment)
#     print("segmentA", (data[max(0, x):min(x+nr, nr),
#                             max(0, y):min(y+nc, nc)] - mean))
#     print("segmentB", (data[max(0, -x):min(-x+nr, nr),
#                             max(0, -y):min(-y+nc, nc)] - mean))
#     print("std", std)

###############################################################################
# Below is testing this against real data

dynspecfile2 \
    = '/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/New/'
dynspecfile2 \
    = dynspecfile2 \
    + 'DataFiles/DynspecPlotFiles/2022-12-30Zap_CompleteDynspec.dynspec'

sim = Simulation()
dyn = Dynspec(dyn=sim, process=False)
dyn.load_file(filename=dynspecfile2)

dyn_crop1 = cp(dyn)
dyn_crop1.crop_dyn(tmin=95, tmax=105, fmin=1000, fmax=1030)

data = dyn_crop1.dyn
dyn_crop1.plot_dyn(filename="/Users/jacobaskew/Desktop/Spectra.png",
                   dpi=400)  # 1

tspan = dyn_crop1.tobs
fspan = dyn_crop1.bw

autocorr = autocorr_func(data)
t_delays = np.linspace(-tspan/60, tspan/60, np.shape(autocorr)[1])
f_shifts = np.linspace(-fspan, fspan, np.shape(autocorr)[0])

dyn_crop1.acf = autocorr
dyn_crop1.get_scint_params(method="acf2d_approx")
SlowACF_dynspec_dnu = round(dyn_crop1.dnu, 3)
SlowACF_dynspec_dnuerr = round(dyn_crop1.dnuerr, 3)

fig, ax1 = plt.subplots(figsize=(9, 9))
CS = ax1.contourf(t_delays, f_shifts, autocorr,
                  levels=np.linspace(-0.5, 1, 10))
fig.colorbar(CS)
plt.title("SlowACF_dynspec, " + str(SlowACF_dynspec_dnu) + " +/- " +
          str(SlowACF_dynspec_dnuerr))
# plt.title("SlowACF_dynspec")
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.savefig("/Users/jacobaskew/Desktop/2DACF_Slow_dynspec.pdf", dpi=400)
plt.savefig("/Users/jacobaskew/Desktop/2DACF_Slow_dynspec.png", dpi=400)
plt.show()
plt.close()  # 2

dyn_crop1.calc_acf()
true_acf = dyn_crop1.acf
dyn_crop1.get_scint_params(method="acf2d_approx")
FFTACF_dynspec_dnu = round(dyn_crop1.dnu, 3)
FFTACF_dynspec_dnuerr = round(dyn_crop1.dnuerr, 3)

fig, ax1 = plt.subplots(figsize=(9, 9))
CS = ax1.contourf(t_delays, f_shifts, true_acf,
                  levels=np.linspace(-0.5, 1, 10))
fig.colorbar(CS)
plt.title("FFTACF_dynspec, " + str(FFTACF_dynspec_dnu) + " +/- " +
          str(FFTACF_dynspec_dnuerr))
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.savefig("/Users/jacobaskew/Desktop/2DACF_FFT_dynspec.pdf", dpi=400)
plt.savefig("/Users/jacobaskew/Desktop/2DACF_FFT_dynspec.png", dpi=400)
plt.show()
plt.close()  # 3

Residual1 = autocorr - dyn_crop1.acf

fig, ax1 = plt.subplots(figsize=(9, 9))
CS = ax1.contourf(t_delays, f_shifts, Residual1)  # ,
#                  levels=np.linspace(-0.5, 1, 10))
fig.colorbar(CS)
plt.title("Residual_Slow_FFT")
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.savefig("/Users/jacobaskew/Desktop/2DACF_Residual_Slow_FFT.pdf",
            dpi=400)
plt.savefig("/Users/jacobaskew/Desktop/2DACF_Residual_Slow_FFT.png",
            dpi=400)
plt.show()
plt.close()  # 4

RFI_beg = 1013
RFI_end = 1018
r1 = np.argmin(abs(RFI_beg - dyn_crop1.freqs))
r2 = np.argmin(abs(RFI_end - dyn_crop1.freqs))
dyn_crop1.dyn[r1:r2, :] = 0
dyn_crop1.plot_dyn(filename="/Users/jacobaskew/Desktop/Spectra_Zeroed.png",
                   dpi=400)  # 5

data = np.ma.masked_where(dyn_crop1.dyn == 0, dyn_crop1.dyn)
autocorr = autocorr_func(data)
dyn_crop1.acf = autocorr
dyn_crop1.get_scint_params(method="acf2d_approx")
SlowACF_masked_dnu = round(dyn_crop1.dnu, 3)
SlowACF_masked_dnuerr = round(dyn_crop1.dnuerr, 3)

t_delays_special = np.linspace(-tspan/60, tspan/60, np.shape(autocorr)[1])
f_shifts_special = np.linspace(-fspan, fspan, np.shape(autocorr)[0])

fig, ax1 = plt.subplots(figsize=(9, 9))
CS = ax1.contourf(t_delays_special, f_shifts_special, autocorr,
                  levels=np.linspace(-0.5, 1, 10))
fig.colorbar(CS)
plt.title("SlowACF_masked, " + str(SlowACF_masked_dnu) + " +/- " +
          str(SlowACF_masked_dnuerr))

plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.savefig("/Users/jacobaskew/Desktop/2DACF_SlowACF_masked.pdf", dpi=400)
plt.savefig("/Users/jacobaskew/Desktop/2DACF_SlowACF_masked.png", dpi=400)
plt.show()
plt.close()  # 6

dyn_crop1.calc_acf()
dyn_crop1.get_scint_params(method="acf2d_approx")
FFTACF_zeroed_dnu = round(dyn_crop1.dnu, 3)
FFTACF_zeroed_dnuerr = round(dyn_crop1.dnuerr, 3)

fig, ax1 = plt.subplots(figsize=(9, 9))
CS = ax1.contourf(t_delays, f_shifts, dyn_crop1.acf,
                  levels=np.linspace(-0.5, 1, 10))
fig.colorbar(CS)
plt.title("FFTACF_zeroed, " + str(FFTACF_zeroed_dnu) + " +/- " +
          str(FFTACF_zeroed_dnuerr))
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.savefig("/Users/jacobaskew/Desktop/2DACF_FFT_zeroed.pdf", dpi=400)
plt.savefig("/Users/jacobaskew/Desktop/2DACF_FFT_zeroed.png", dpi=400)
plt.show()
plt.close()  # 7
#############
fig, ax1 = plt.subplots(figsize=(9, 9))
CS = plt.pcolormesh(dyn_crop1.acf)
# CS = ax1.contourf(t_delays, f_shifts, autocorr)
plt.xlim(dyn_crop1.acf.shape[1]/2-10, dyn_crop1.acf.shape[1]/2+11)
plt.ylim(dyn_crop1.acf.shape[0]/2-10, dyn_crop1.acf.shape[0]/2+11)
fig.colorbar(CS)
plt.clim([-0.5, 1])
# plt.title("Residual_FFTMedian_TrueACF")
# plt.savefig(
#     "/Users/jacobaskew/Desktop/2DACF_Residual_FFTMedian_TrueACF.pdf",
#     dpi=400)
# plt.savefig(
#     "/Users/jacobaskew/Desktop/2DACF_Residual_FFTMedian_TrueACF.png",
#     dpi=400)
plt.show()
plt.close()  # 18
#############

Residual2 = autocorr - dyn_crop1.acf

fig, ax1 = plt.subplots(figsize=(9, 9))
CS = ax1.contourf(t_delays, f_shifts, Residual2,
                  levels=np.linspace(-0.5, 1, 10))
fig.colorbar(CS)
plt.title("Residual_SlowMasked_FFTZeroed")
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.savefig(
    "/Users/jacobaskew/Desktop/2DACF_Residual_SlowMasked_FFTZeroed.pdf",
    dpi=400)
plt.savefig(
    "/Users/jacobaskew/Desktop/2DACF_Residual_SlowMasked_FFTZeroed.png",
    dpi=400)
plt.show()
plt.close()  # 8

Residual6 = autocorr - true_acf

fig, ax1 = plt.subplots(figsize=(9, 9))
CS = ax1.contourf(t_delays, f_shifts, Residual6)  # ,
#                  levels=np.linspace(-0.5, 1, 10))
fig.colorbar(CS)
plt.title("Residual_SlowMasked_TrueACF")
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.savefig(
    "/Users/jacobaskew/Desktop/2DACF_Residual_SlowMasked_TrueACF.pdf",
    dpi=400)
plt.savefig(
    "/Users/jacobaskew/Desktop/2DACF_Residual_SlowMasked_TrueACF.png",
    dpi=400)
plt.show()
plt.close()  # 9

Residual7 = dyn_crop1.acf - true_acf

fig, ax1 = plt.subplots(figsize=(9, 9))
CS = ax1.contourf(t_delays, f_shifts, Residual7,
                  levels=np.linspace(-0.5, 1, 10))
fig.colorbar(CS)
plt.title("Residual_FFTZeroed_TrueACF")
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.savefig(
    "/Users/jacobaskew/Desktop/2DACF_Residual_FFTZeroed_TrueACF.pdf",
    dpi=400)
plt.savefig(
    "/Users/jacobaskew/Desktop/2DACF_Residual_FFTZeroed_TrueACF.png",
    dpi=400)
plt.show()
plt.close()  # 10

dyn_crop1.refill()
dyn_crop1.plot_dyn(filename="/Users/jacobaskew/Desktop/Spectra_biharmonic.png",
                   dpi=400)  # 11
# data = dyn_crop1.dyn
# autocorr = autocorr_func(data)

# fig, ax1 = plt.subplots(figsize=(9, 9))
# CS = ax1.contourf(t_delays, f_shifts, autocorr)
# fig.colorbar(CS)
# plt.title("SlowACF_refilled")
# plt.savefig("/Users/jacobaskew/Desktop/2DACF_Test_masked.pdf", dpi=400)
# plt.savefig("/Users/jacobaskew/Desktop/2DACF_Test_masked.png", dpi=400)
# plt.show()
# plt.close()  #

dyn_crop1.calc_acf()
acf_biharmonic = dyn_crop1.acf
dyn_crop1.get_scint_params(method="acf2d_approx")
FFT_refilled_biharmonic_dnu = round(dyn_crop1.dnu, 3)
FFT_refilled_biharmonic_dnuerr = round(dyn_crop1.dnuerr, 3)


fig, ax1 = plt.subplots(figsize=(9, 9))
CS = ax1.contourf(t_delays, f_shifts, dyn_crop1.acf,
                  levels=np.linspace(-0.5, 1, 10))
fig.colorbar(CS)
plt.title("FFT_refilled_biharmonic, " +
          str(FFT_refilled_biharmonic_dnu) + " +/- " +
          str(FFT_refilled_biharmonic_dnuerr))
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.savefig(
    "/Users/jacobaskew/Desktop/2DACF_FFT_refilled_biharmonic.pdf",
    dpi=400)
plt.savefig(
    "/Users/jacobaskew/Desktop/2DACF_FFT_refilled_biharmonic.png",
    dpi=400)
plt.show()
plt.close()  # 12

Residual5 = dyn_crop1.acf - true_acf

fig, ax1 = plt.subplots(figsize=(9, 9))
CS = ax1.contourf(t_delays, f_shifts, Residual5,
                  levels=np.linspace(-0.5, 1, 10))
fig.colorbar(CS)
plt.title("Residual_FFTBiharmonic_TrueACF")
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.savefig(
    "/Users/jacobaskew/Desktop/2DACF_Residual_FFTBiharmonic_TrueACF.pdf",
    dpi=400)
plt.savefig(
    "/Users/jacobaskew/Desktop/2DACF_Residual_FFTBiharmonic_TrueACF.png",
    dpi=400)
plt.show()
plt.close()  # 13

dyn_crop1 = cp(dyn)
dyn_crop1.crop_dyn(tmin=95, tmax=105, fmin=1000, fmax=1030)
r1 = np.argmin(abs(RFI_beg - dyn_crop1.freqs))
r2 = np.argmin(abs(RFI_end - dyn_crop1.freqs))
dyn_crop1.dyn[r1:r2, :] = 0

dyn_crop1.refill(method='median')
dyn_crop1.plot_dyn(filename="/Users/jacobaskew/Desktop/Spectra_median.png",
                   dpi=400)  # 14

dyn_crop1.calc_acf()
dyn_crop1.get_scint_params(method="acf2d_approx")
FFT_refilled_median_dnu = round(dyn_crop1.dnu, 3)
FFT_refilled_median_dnuerr = round(dyn_crop1.dnuerr, 3)

fig, ax1 = plt.subplots(figsize=(9, 9))
CS = ax1.contourf(t_delays, f_shifts, dyn_crop1.acf,
                  levels=np.linspace(-0.5, 1, 10))
fig.colorbar(CS)
plt.title("FFT_refilled_median, " +
          str(FFT_refilled_median_dnu) + " +/- " +
          str(FFT_refilled_median_dnuerr))
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.savefig(
    "/Users/jacobaskew/Desktop/2DACF_FFT_refilled_median.pdf",
    dpi=400)
plt.savefig(
    "/Users/jacobaskew/Desktop/2DACF_FFT_refilled_median.png",
    dpi=400)
plt.show()
plt.close()  # 15

Residual8 = dyn_crop1.acf - true_acf

fig, ax1 = plt.subplots(figsize=(9, 9))
CS = ax1.contourf(t_delays, f_shifts, Residual8)  # ,
#                  levels=np.linspace(-0.5, 1, 10))
fig.colorbar(CS)
plt.title("Residual_FFTMedian_TrueACF")
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.savefig(
    "/Users/jacobaskew/Desktop/2DACF_Residual_FFTMedian_TrueACF.pdf",
    dpi=400)
plt.savefig(
    "/Users/jacobaskew/Desktop/2DACF_Residual_FFTMedian_TrueACF.png",
    dpi=400)
plt.show()
plt.close()  # 16

Residual9 = dyn_crop1.acf - acf_biharmonic

fig, ax1 = plt.subplots(figsize=(9, 9))
CS = ax1.contourf(t_delays, f_shifts, Residual9)  # ,
#                  levels=np.linspace(-0.5, 1, 10))
fig.colorbar(CS)
plt.title("Residual_FFTMedian_FFTBiharmonic")
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.savefig(
    "/Users/jacobaskew/Desktop/2DACF_Residual_FFTMedian_TrueACF.pdf",
    dpi=400)
plt.savefig(
    "/Users/jacobaskew/Desktop/2DACF_Residual_FFTMedian_TrueACF.png",
    dpi=400)
plt.show()
plt.close()  # 17

###############################################################################

# # Here we create an array for the ACF that can be turned into the 3D plot #

# dyn_initial.get_scint_params(method="acf2d_approx")
# data = dyn_initial.dyn

# test_list_time = []
# for i in range(0, np.shape(data)[1]*2):
#     if i == 0 or i == np.shape(data)[1]:
#         x_1 = data[:, :]
#         x_2 = data[:, :]
#     elif i < np.shape(data)[1]:
#         x_1 = data[:, i:]
#         x_2 = data[:, :-i]
#     else:
#         x_1 = data[:, :i-np.shape(data)[1]]
#         x_2 = data[:, -i+np.shape(data)[1]:]
#     test_list_time.append(np.mean((x_1 - np.mean(x_1)) *
#                           (x_2 - np.mean(x_2))) /
#                           (np.std(x_1) * np.std(x_2)))

# plt.plot(test_list_time)
# plt.show()
# plt.close()
# # THE ABOVE GETS ME UP TO AND INCLUDING STEP 3

# data = dyn_initial.dyn
# test_list_time = []
# for i in range(0, len(data[1])*2):
#     for ii in range(0, np.shape(data)[0]*2):
#         if i == 0 or i == len(data[1]):
#             x_1 = data[:, :]
#             x_2 = data[:, :]
#         elif i < len(data[1]):
#             x_1 = data[:, i:]
#             x_2 = data[:, :-i]
#         else:
#             x_1 = data[:, :i-np.shape(data)[1]]
#             x_2 = data[:, -i+np.shape(data)[1]:]
#         test_list_time.append(np.mean((x_1 - np.mean(x_1)) *
#                               (x_2 - np.mean(x_2))) /
#                               (np.std(x_1) * np.std(x_2)))

# plt.plot(test_list_time)
# plt.show()
# plt.close()
# # This is time but I loop over the frequency as well.

# test_list_freq = []
# for ii in range(0, np.shape(data)[0]*2):
#     if ii == 0 or ii == np.shape(data)[0]:
#         x_1 = data[:, :]
#         x_2 = data[:, :]
#     elif i < np.shape(data)[0]:
#         x_1 = data[ii:, :]
#         x_2 = data[:-ii, :]
#     else:
#         x_1 = data[ii-np.shape(data)[0]:, :]
#         x_2 = data[:-ii+np.shape(data)[0], :]
#     test_list_freq.append(np.mean((x_1 - np.mean(x_1)) *
#                           (x_2 - np.mean(x_2))) /
#                           (np.std(x_1) * np.std(x_2)))

# plt.plot(test_list_freq)
# plt.show()
# plt.close()
# # This plot should show the freq lags

# test_list_both = []


# if i == 0 or i == np.shape(data)[1] \
#         and ii != 0 and ii != np.shape(data)[0]:
#     x_1 = data[ii:, :]
#     x_2 = data[:-ii, :]
# elif ii == 0 or ii == np.shape(data)[0] \
#         and i != 0 and i != np.shape(data)[1]:
#     x_1 = data[:, i:]
#     x_2 = data[:, :-i]
# elif i == 0 and ii == 0:
#     x_1 = data[:, :]
#     x_2 = data[:, :]
# elif i == 0 and ii == np.shape(data)[0]:
#     x_1 = data[:, :]
#     x_2 = data[:, :]
# elif i == np.shape(data)[0] and ii == 0:
#     x_1 = data[:, :]
#     x_2 = data[:, :]
# elif i == np.shape(data)[1] and ii == np.shape(data)[0]:
#     x_1 = data[:, :]
#     x_2 = data[:, :]

# test_array_both = np.zeros((512, 256))
# alternate_array_both = np.zeros((512, 256))
# nr, nc = np.shape(data)

# for i in range(0, nc*2):
#     for ii in range(0, nr*2):
#         # if i == 0 or i == nc or \
#         #           ii == 0 or ii == nr:
#         #     x_1 = data[:, :]
#         #     x_2 = data[:, :]
#         if i < nc and ii < nr:
#             x_1 = data[ii:, i:]
#             x_2 = data[:-ii or None, :-i or None]
#         elif i >= nc and ii < nr:
#             x_1 = data[ii:, :i-nc or None]
#             x_2 = data[:(-ii or None), (-i or None)+nc or None:]
#         elif i < nc and ii >= nr:
#             x_1 = data[:ii-nr or None, i:]
#             x_2 = data[(-ii or None)+nr or None:, :-i or None]
#         else:
#             x_1 = data[:ii-nr or None, :i-nc or None]
#             x_2 = data[(-ii or None)+nr or None:,
#                        (-i or None)+nc or None:]
#         test_array_both[ii, i] = np.mean((x_1 - np.mean(x_1)) *
#                                          (x_2 - np.mean(x_2))) / \
#             (np.std(x_1) * np.std(x_2))
#         mean = np.mean(np.concatenate((x_1, x_2)))
#         std = np.std(np.concatenate((x_1, x_2)))
#         prod = (x_1 - mean) * (x_2 - mean)
#         alternate_array_both[ii, i] = np.sum(prod) / (std ** 2)
# # plt.plot(test_list_both)
# plt.plot(test_array_both)
# plt.show()
# plt.close()
# This plot should show both the frequency and the time lags

###############################################################################

# Here is when I attempt to do a 2D plot myself

# wn = test_array_both[0][0] - max([test_array_both[1][0],
#                                   test_array_both[0][1]])
# test_array_both[0][0] = test_array_both[0][0] - wn

# test_array_both = np.fft.fftshift(test_array_both)
# test_array_both = np.fft.fftshift(alternate_array_both)

# tspan = dyn_initial.tobs
# fspan = dyn_initial.bw

# t_delays = np.linspace(-tspan/60, tspan/60, np.shape(test_array_both)[1])
# f_shifts = np.linspace(-fspan, fspan, np.shape(test_array_both)[0])

# fig, ax1 = plt.subplots(figsize=(9, 9))
# ax1.contourf(t_delays, f_shifts, test_array_both)
# plt.savefig("/Users/jacobaskew/Desktop/2DACF_Test.pdf", dpi=400)
# plt.savefig("/Users/jacobaskew/Desktop/2DACF_Test.png", dpi=400)
# plt.show()
# plt.close()


###############################################################################

# Here is where I will plot the difference between the two ACFs scintools
# and mine

# diff_array = dyn_initial.acf - test_array_both

# fig, ax1 = plt.subplots(figsize=(9, 9))
# ax1.contourf(t_delays, f_shifts, diff_array)
# plt.savefig("/Users/jacobaskew/Desktop/2DACF_Diff.pdf", dpi=400)
# plt.savefig("/Users/jacobaskew/Desktop/2DACF_Diff.png", dpi=400)
# plt.show()
# plt.close()

###############################################################################
# Everything below this line is the proper method of doing things

# dyn_initial.get_scint_params(method="acf2d_approx")


# def autocorr(data):
#     nr, nc = np.shape(data)
#     autocorr = np.zeros((nr*2, nc*2))
#     mean = np.mean(data)
#     std = np.std(data)

#     for i in range(0, nc*2):
#         for ii in range(0, nr*2):
#             if i < nc and ii < nr:
#                 x_1 = data[ii:, i:]
#                 x_2 = data[:-ii or None, :-i or None]
#             elif i >= nc and ii < nr:
#                 x_1 = data[ii:, :i-nc or None]
#                 x_2 = data[:-ii or None, -i+nc or None:]
#             elif i < nc and ii >= nr:
#                 x_1 = data[:ii-nr or None, i:]
#                 x_2 = data[-ii+nr or None:, :-i or None]
#             else:
#                 x_1 = data[:ii-nr or None, :i-nc or None]
#                 x_2 = data[-ii+nr or None:,
#                            -i+nc or None:]
#             prod = (x_1 - mean) * (x_2 - mean)
#             autocorr[ii, i] = np.sum(prod) / (std ** 2)

#     autocorr /= np.max(autocorr)
#     return autocorr


# def autocorr(data):
#     mean = np.ma.mean(data)
#     std = np.ma.std(data)
#     nr, nc = np.shape(data)
#     autocorr = np.zeros((2*nr, 2*nc))
#     for x in range(-nr, nr):
#         for y in range(-nc, nc):
#             segment = (data[max(0, x):min(x+nr, nr),
#                             max(0, y):min(y+nc, nc)] - mean) \
#                     * (data[max(0, -x):min(-x+nr, nr),
#                             max(0, -y):min(-y+nc, nc)] - mean)
#             numerator = np.ma.sum(segment)
#             autocorr[x+nr][y+nc] = numerator / (std ** 2)
#     autocorr /= np.nanmax(autocorr)
#     return autocorr

# autocorr = autocorr(dyn_initial.dyn)
# ACF_2D = autocorr
# # for i in range(0, 512):
# # autocorr[511][0] = 0
# # autocorr[0][255] = 0
# # ACF_2D = np.fft.fftshift(autocorr)
# plt.plot(ACF_2D)
# plt.show()
# plt.close()

# # wn = autocorr[0][0] - max([autocorr[1][0], autocorr[0][1]])
# # autocorr[0][0] = autocorr[0][0] - wn

# tspan = dyn_initial.tobs
# fspan = dyn_initial.bw

# t_delays = np.linspace(-tspan/60, tspan/60, np.shape(ACF_2D)[1])
# f_shifts = np.linspace(-fspan, fspan, np.shape(ACF_2D)[0])

# fig, ax1 = plt.subplots(figsize=(9, 9))
# CS = ax1.contourf(t_delays, f_shifts, ACF_2D)
# fig.colorbar(CS)
# plt.savefig("/Users/jacobaskew/Desktop/2DACF_Test.pdf", dpi=400)
# plt.savefig("/Users/jacobaskew/Desktop/2DACF_Test.png", dpi=400)
# plt.show()
# plt.close()

# ACF_2D_Scintools = dyn_initial.acf

# fig, ax1 = plt.subplots(figsize=(9, 9))
# CS = ax1.contourf(t_delays, f_shifts, ACF_2D_Scintools)
# fig.colorbar(CS)
# plt.savefig("/Users/jacobaskew/Desktop/2DACF_Scintools.pdf", dpi=400)
# plt.savefig("/Users/jacobaskew/Desktop/2DACF_Scintools.png", dpi=400)
# plt.show()
# plt.close()

# diff_array = dyn_initial.acf - ACF_2D

# fig, ax1 = plt.subplots(figsize=(9, 9))
# CS = ax1.contourf(t_delays, f_shifts, diff_array)
# fig.colorbar(CS)
# plt.savefig("/Users/jacobaskew/Desktop/2DACF_Diff.pdf", dpi=400)
# plt.savefig("/Users/jacobaskew/Desktop/2DACF_Diff.png", dpi=400)
# plt.show()
# plt.close()
###############################################################################
# Everything below here is for creating a 2D plot using dynspec code as a base

# # arr = np.asarray(test_list_both)
# arr = test_array_both
# arr = arr.reshape((512, 256))
# # arr = arr.reshape((512, 1))
# # arr = arr.reshape((1, 256))

# arr = np.fft.ifftshift(arr)
# subtract the white noise spike
# wn = arr[0][0] - max([arr[1][0], arr[0][1]])
# arr[0][0] = arr[0][0] - wn  # Set the noise spike to zero for plotting
# arr = np.fft.fftshift(arr)

# tspan = dyn_initial.tobs
# fspan = dyn_initial.bw

# t_delays = np.linspace(-tspan/60, tspan/60, np.shape(arr)[1])
# f_shifts = np.linspace(-fspan, fspan, np.shape(arr)[0])

# nscale = 4

# tlim = nscale * dyn_initial.tau / 60
# flim = nscale * dyn_initial.dnu
# if tlim > dyn_initial.tobs / 60:
#     tlim = dyn_initial.tobs / 60
# if flim > dyn_initial.bw:
#     flim = dyn_initial.bw

#     t_inds = np.argwhere(np.abs(t_delays) <= tlim).squeeze()
#     f_inds = np.argwhere(np.abs(f_shifts) <= flim).squeeze()
#     t_delays = t_delays[t_inds]
#     f_shifts = f_shifts[f_inds]

#     arr = arr[f_inds, :]
#     arr = arr[:, t_inds]

# fig, ax1 = plt.subplots(figsize=(9, 9))
# ax1.contourf(t_delays, f_shifts, arr)
# ax1.set_ylabel(r'Frequency shift, $\Delta\nu$ (MHz)')
# ax1.set_xlabel(r'Time lag, $\tau$ (mins)')
# miny, maxy = ax1.get_ylim()
# ax2 = ax1.twinx()
# ax2.set_ylim(miny/dyn_initial.dnu, maxy/dyn_initial.dnu)
# ax2.set_ylabel(r'$\Delta\nu$ / ($\Delta\nu_d = {0}\,$MHz)'.
#                format(round(dyn_initial.dnu, 2)))
# ax3 = ax1.twiny()
# minx, maxx = ax1.get_xlim()
# ax3.set_xlim(minx/(dyn_initial.tau/60), maxx/(dyn_initial.tau/60))
# ax3.set_xlabel(r'$\tau$/($\tau_d={0}\,$min)'.format(round(
#                                                     dyn_initial.tau/60, 2)))
# plt.contourf(t_delays, f_shifts, arr)
# plt.ylabel('Frequency lag (MHz)')
# plt.xlabel('Time lag (mins)')

# plt.show()
# plt.close()
