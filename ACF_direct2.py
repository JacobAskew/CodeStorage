#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 22:52:35 2020

@author: jacobaskew
"""
from scintools.dynspec import Dynspec
from scintools.scint_sim import Simulation
from itertools import product
# from scintools.scint_utils import write_results, read_results, read_par, \
#         float_array_from_dict, get_ssb_delay, get_earth_velocity, \
#         get_true_anomaly, is_valid, svd_model, interp_nan_2d, \
#         centres_to_edges

# import glob
import numpy as np
from numpy import empty, roll
# # import random

# import math
# # from astropy.time import Time
import matplotlib.pyplot as plt
# import matplotlib
# # import scipy.signal as sig
# from copy import deepcopy as cp
# from lmfit import Parameters, minimize
# import os
# from scipy.optimize import curve_fit
# from scipy.interpolate import griddata
# from scipy import interpolate
# from scipy.interpolate import interp2d


##############################################################################


def array_to_dynspec(flux, times, freqs, filename=None):

    if filename is None:
        fname = 'placeholder.dynspec'
    else:
        fname = filename
    # now write to file
    with open(fname, 'w') as fn:
        fn.write("# Scintools-modified dynamic spectrum " +
                 "in psrflux format\n")
        fn.write("# Created using write_file method in Dynspec class\n")
        fn.write("# Original header begins below:\n")
        fn.write("#\n")

        for i in range(len(times)-1):
            fn.write("# {} \n".format(i))
            ti = times[i]/60
            for j in range(len(freqs)):
                fi = freqs[j]
                di = flux[j, i]
                # di_err = self.dyn_err[j, i]
                fn.write("{0} {1} {2} {3} {4}\n".  # {5}
                         format(i, j, ti, fi, di))  # , di_err))


##############################################################################
z = 1000
c = 3*10**8
f = 800*10**6
wavelength = c/f
k = (2*np.pi)/wavelength
rf = np.sqrt(z/k)
# mb2 = 0.773*(rf) * 100
dnu_c = 1
mb2 = 0.773*(1100/dnu_c)**(5/6)
outfile = '/Users/jacobaskew/Desktop/test.txt'
resolution = 0.1
NF = int(round(1100/(resolution*dnu_c), -1))
measure = False
sim = False
load_data = True
plotting = True
compare = True
# 1360
if not '/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/Simulation/Dynspec/SimDynspec_'+str(resolution)+'.dynspec':
    sim = Simulation(mb2=mb2, rf=1, ds=0.01, alpha=5/3, ar=1, psi=0,
                      inner=0.001, ns=128, nf=256,
                      dlam=0.0375, lamsteps=False, seed=64, nx=None,
                      ny=None, dx=None, dy=None, plot=False, verbose=False,
                      freq=800, dt=8, mjd=50000, nsub=None, efield=False,
                      noise=None)
    dyn_initial = Dynspec(dyn=sim, process=False)
    dyn_initial.trim_edges()
    dyn_initial.plot_dyn(filename='/Users/jacobaskew/Desktop/SimSpectra_' +
                          str(resolution) + '.png', dpi=400)
    dyn_initial.write_file(filename='/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/Simulation/Dynspec/SimDynspec_'+str(resolution)+'.dynspec')
else:
    sim = Simulation()
    dyn_initial = Dynspec(dyn=sim, process=False)
    dyn_initial.load_file(filename='/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/Simulation/Dynspec/SimDynspec_'+str(resolution)+'.dynspec')
    dyn_initial.trim_edges()
    dyn_initial.plot_dyn(filename='/Users/jacobaskew/Desktop/SimSpectra_' +
                         str(resolution) + '.png', dpi=400)

dyn_initial.get_scint_params(method="nofit", plot=True, display=True)
print()
print("nofit")
print("dnu=", round(dyn_initial.dnu, 3), "+/-", round(dyn_initial.dnuerr, 3))
print("tau=", round(dyn_initial.tau, 3), "+/-", round(dyn_initial.tauerr, 3))

dyn_initial.get_scint_params(method="acf1d", plot=True, display=True)
print()
print("acf1d")
print("dnu=", round(dyn_initial.dnu, 3), "+/-", round(dyn_initial.dnuerr, 3))
print("tau=", round(dyn_initial.tau, 3), "+/-", round(dyn_initial.tauerr, 3))

dyn_initial.get_scint_params(method="acf2d_approx", plot=True, display=True)
print()
print("acf2d_approx")
print("dnu=", round(dyn_initial.dnu, 3), "+/-", round(dyn_initial.dnuerr, 3))
print("tau=", round(dyn_initial.tau, 3), "+/-", round(dyn_initial.tauerr, 3))

dyn_initial.get_acf_tilt(plot=True, display=True)

dyn_initial.plot_acf(contour=True, crop=False)


# arr = data - np.mean(data)
# nf = np.shape(data)[1]
# nt = np.shape(data)[0]
# arr = np.fft.fft2(arr, s=[2*nf, 2*nt])  # zero-padded
# arr = np.abs(arr)  # absolute value
# arr **= 2  # Squared manitude
# arr = np.fft.ifft2(arr)
# arr = np.fft.fftshift(arr)
# arr = np.real(arr)  # real component, just in case
# arr /= np.max(arr)  # normalise
# plt.plot(arr)


# def normalization(x):
#     # (1056, 128) this shape
#     acor = np.zeros((x.shape))
#     for i in range(0, x.shape[1]-1):
#         for ii in range(0, x.shape[1]):
#             # for iii in range(0, x.shape[0]):
#             x1 = x[ii, i]
#             x2 = x[ii, i+1]
#             acor[ii, i] = np.mean((x1 - np.mean(x1)) *
#                                   (x2 - np.mean(x2))) / \
#                 (np.std(x1) * np.std(x2))
#     return acor


data = dyn_initial.dyn
acor = np.zeros((data.shape[0], data.shape[1]))
data0 = data[50, :]

for i in range(1, data.shape[1]):
    x1 = data0[:-i]
    x2 = data0[i:]
    acor[50, i] = np.mean((x1 - np.mean(x1)) *
                          (x2 - np.mean(x2))) / \
        (np.std(x1) * np.std(x2))

plt.plot(dyn_initial.acf)
plt.show()
plt.close()
plt.plot(acor)
plt.show()
plt.close()

data = dyn_initial.dyn
acor = np.zeros((data.shape[0]*2, data.shape[1]*2))

for i in range(1, data.shape[1]-1):
    for ii in range(1, data.shape[0]-1):
        x1 = data[ii, i]
        x2 = data[ii+1, i+1]
        x1_ = data[data.shape[0]-ii, data.shape[1]-i]
        x2_ = data[(data.shape[0]-ii)-1, (data.shape[1]-i)-1]
        acor[ii, i] = np.mean((x1 - np.mean(x1)) *
                              (x2 - np.mean(x2))) / \
            (np.std(x1) * np.std(x2))
        acor[ii*2, i*2] = np.mean((x1_ - np.mean(x1_)) *
                                  (x2_ - np.mean(x2_))) / \
            (np.std(x1_) * np.std(x2_))

plt.plot(dyn_initial.acf)
plt.show()
plt.close()
plt.plot(acor)
plt.show()
plt.close()


# data = dyn_initial.dyn
# acor = np.zeros((data.shape[0], data.shape[1]))

# for ii in range(1, data.shape[0]):
#     data0 = data[ii, :]
#     for i in range(1, data.shape[1]):
#         x1 = data0[i]
#         x2 = data0[i:]
#         acor[ii, i] = np.corrcoef(x1, x2)[0, 1]

# plt.plot(dyn_initial.acf)
# plt.show()
# plt.close()
# plt.plot(acor)
# plt.show()
# plt.close()
# plt.xlim(0, 2)


# def autocovariance(Xi, N, k, Xs):
#     autoCov = 0
#     for i in np.arange(0, N-k):
#         autoCov += ((Xi[i+k])-Xs)*(Xi[i]-Xs)
#     return (1/(N-1))*autoCov


# data = dyn_initial.dyn
# acor = np.zeros((data.shape[0]*2, data.shape[1]*2))
# for i in range(1, data.shape[1]):
#     for ii in range(1, data.shape[0]):
#         Xi = data[ii, :-i]
#         N = np.size(Xi)
#         if N == 1:
#             continue
#         k = 1
#         Xs = np.average(Xi)
#         acor[ii, i] = autocovariance(Xi, N, k, Xs)

# plt.plot(dyn_initial.acf)
# plt.show()
# plt.close()
# plt.plot(acor)
# plt.show()
# plt.close()


# data = dyn_initial.dyn

# acor = np.zeros((data.shape))
# for i in range(1, data.shape[1]):
#     for ii in range(1, data.shape[0]):
#         acor[ii, i] = np.corrcoef(data[:-ii, :-i], data[ii:, i:])[0, 1]

# plt.plot(acor)
# def covariance(x, y):
#     # Finding the mean of the series x and y
#     mean_x = sum(x)/float(len(x))
#     mean_y = sum(y)/float(len(y))
#     # Subtracting mean from the individual elements
#     sub_x = [i - mean_x for i in x]
#     sub_y = [i - mean_y for i in y]
#     numerator = sum([sub_x[i]*sub_y[i] for i in range(len(sub_x))])
#     denominator = len(x)-1
#     cov = numerator/denominator
#     return cov

# Test = np.zeros((dyn_initial.dyn.shape))

# for i in range(1, len(dyn_initial.times)):
#     for ii in range(1, len(dyn_initial.freqs)):
#         # covariance(dyn_initial.dyn[i, :], dyn_initial.dyn[:, ii])
#         Test[ii, i] = (autocorrelate(dyn_initial.times[0:i],
#                                   dyn_initial.dyn[0:ii, 0:i]))

# Test = autocorrelate(dyn_initial.dyn)
# Test = normalization(dyn_initial.dyn)
Test = acor
arr = Test
# plt.plot(Test)
# arr = dyn_initial.acf
print("done")
# arr = np.reshape(Test, dyn_initial.dyn.shape)
tspan = dyn_initial.tobs
fspan = dyn_initial.bw
arr = np.fft.ifftshift(arr)
# subtract the white noise spike
wn = arr[0][0] - max([arr[1][0], arr[0][1]])
arr[0][0] = arr[0][0] - wn  # Set the noise spike to zero for plotting
arr = np.fft.fftshift(arr)

t_delays = np.linspace(-tspan/60, tspan/60, np.shape(arr)[1])
f_shifts = np.linspace(-fspan, fspan, np.shape(arr)[0])

# if crop or (tlim is not None):
# Set limits automatically
tlim = 4 * dyn_initial.tau / 60
flim = 4 * dyn_initial.dnu
if tlim > dyn_initial.tobs / 60:
    tlim = dyn_initial.tobs / 60
if flim > dyn_initial.bw:
    flim = dyn_initial.bw
    
    
   #jacob sucks big time 

#     t_inds = np.argwhere(np.abs(t_delays) <= tlim).squeeze()
#     f_inds = np.argwhere(np.abs(f_shifts) <= flim).squeeze()
#     t_delays = t_delays[t_inds]
#     f_shifts = f_shifts[f_inds]

#     arr = arr[f_inds, :]
#     arr = arr[:, t_inds]

# if input_acf is None:  # Also plot scintillation scales axes

fig, ax1 = plt.subplots(figsize=(15, 15))
ax1.contourf(t_delays, f_shifts, arr)
ax1.set_ylabel(r'Frequency shift, $\Delta\nu$ (MHz)')
ax1.set_xlabel(r'Time lag, $\tau$ (mins)')
miny, maxy = ax1.get_ylim()
ax2 = ax1.twinx()
ax2.set_ylim(miny/dyn_initial.dnu, maxy/dyn_initial.dnu)
ax2.set_ylabel(r'$\Delta\nu$ / ($\Delta\nu_d = {0}\,$MHz)'.
               format(round(dyn_initial.dnu, 2)))
ax3 = ax1.twiny()
minx, maxx = ax1.get_xlim()
ax3.set_xlim(minx/(dyn_initial.tau/60), maxx/(dyn_initial.tau/60))
ax3.set_xlabel(r'$\tau$/($\tau_d={0}\,$min)'.format(round(
                                              dyn_initial.tau/60, 2)))
# else:  # just plot acf without scales
plt.contourf(t_delays, f_shifts, arr)
plt.ylabel('Frequency lag (MHz)')
plt.xlabel('Time lag (mins)')
# plt.xlim(-3, 3)
# plt.ylim(-0.5, 0.5)
plt.show()

dyn_initial.plot_acf(contour=True)
