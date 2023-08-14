#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 22:52:35 2020

@author: jacobaskew
"""
from scintools.dynspec import Dynspec
from scintools.scint_sim import Simulation
from scintools.scint_utils import centres_to_edges
import numpy as np
import matplotlib.pyplot as plt
# from copy import deepcopy as cp
import matplotlib
# import os
# import time
# import sympy as sym
weights_2dacf = None
###############################################################################
Size = 80*np.pi  # Determines the size of the datapoints used
font = {'size': 10}
matplotlib.rc('font', **font)
# ###############################################################################
# sim = Simulation(mb2=20, ns=64, nf=64, seed=64)
# dyn_sim = Dynspec(dyn=sim, process=False)
# dyn_sim.plot_dyn()
# dyn_sim.get_scint_params(method='acf2d_approx', plot=True, cutoff=True)
###############################################################################
# Here I am testing the Bartlett errors using the actual equations
# First the 1d and then the 2d equations.
sim = Simulation(mb2=50, ns=8, nf=8, seed=64)
dyn_sim = Dynspec(dyn=sim, process=False)
# dyn_sim.plot_dyn()

# dyn_sim.get_scint_params(method='acf2d_approx', cutoff=False,
# redchisqr=False,
#                          weights_2dacf=None, plot=True)
# # dyn_sim.get_scint_params(method='acf1d', plot=True)
# dyn_sim.plot_acf(crop=True)
# dyn_sim.plot_acf(crop=True, contour=True)
dyn_sim.calc_acf()
ACF_DATA = dyn_sim.acf
ACF_DATA_NEW = ACF_DATA[1:, 1:]
ACF_DATA_NEW = ACF_DATA
# 1D Bartlett

# TIME
var_t = []
nf, nt = np.shape(ACF_DATA_NEW)
dyn_nf, dyn_nt = np.shape(dyn_sim.dyn)
ydata_t = ACF_DATA_NEW[int(nf/2), :]
xdata_t = np.zeros((nt))
xdata_t[:int(nt/2)] = (-dyn_sim.dt * np.linspace(int(nt/2)-1, 0,
                                                 int(nt/2))) / 60
xdata_t[int(nt/2):] = (dyn_sim.dt * np.linspace(0, int(nt/2),
                                                int(nt/2))) / 60
for i in range(nt):
    var_dum = 0
    for k in range(-nt, nt):
        if k+i < nt and k+i >= 0:
            var_dum += ydata_t[k+i]**2
        if k-i >= 0 and k-i < nt and k+i < nt and k+i >= 0:
            var_dum += ydata_t[k-i]*ydata_t[k+i]
        if k >= 0 and k < nt:
            var_dum += 2*ydata_t[i]**2*ydata_t[k]**2
        if k+i < nt and k+i >= 0 and k >= 0 and k < nt:
            var_dum -= 4*ydata_t[i] * ydata_t[k]*ydata_t[k+i]
    var_t.append(var_dum)

var_t_arr = np.asarray(var_t)
sigma_sqr_t = np.sqrt(var_t_arr / (dyn_nt))  # N should be the len(dyn)

plt.scatter(xdata_t, ydata_t, c='C0', label=r'ACF1D for $\tau_d$')
plt.fill_between(xdata_t, ydata_t-sigma_sqr_t, ydata_t+sigma_sqr_t, alpha=0.2,
                 label='Bartlett Errors 1D')
xl = plt.xlim()
plt.xlabel('Lags (mins)')
plt.ylabel('ACF Value')
# plt.xlim(np.max(xdata_t)/2, xl[1]*0.96)
plt.show()

sigma_sqr_t_solv = sigma_sqr_t[int(nt/2):]

# Online solution on time
ydata_t_solv = ydata_t[int(nt/2):]
xdata_t_solv = xdata_t[int(nt/2):]
variance_t_solv = (np.ones(np.shape(ydata_t_solv)) / (nt / 2))
variance_t_solv[0] = 1e-10
variance_t_solv[2:] *= 1 + 2 * np.cumsum(ydata_t_solv[1:-1] ** 2)
t_errors = np.sqrt(variance_t_solv)

plt.scatter(xdata_t_solv, ydata_t_solv, s=1, c='k',
            label=r'ACF1D for $\tau_d$')
plt.errorbar(xdata_t_solv, ydata_t_solv, yerr=sigma_sqr_t_solv, fmt=' ',
             ecolor='C0', elinewidth=0.5, capsize=0.5, alpha=0.3)
plt.errorbar(xdata_t_solv, ydata_t_solv, yerr=t_errors, fmt=' ',
             ecolor='C1', elinewidth=0.5, capsize=0.5, alpha=0.3)
# plt.fill_between(xdata_t[int(nt/2):], ydata_t[int(nt/2):]-t_errors,
# ydata_t[int(nt/2):]+t_errors, alpha=0.2,
#                  label='Bartlett Errors 1D')
xl = plt.xlim()
plt.xlabel('Time Lags (mins)')
plt.ylabel('ACF Value')
# plt.xlim(np.max(xdata_t_solv)/2, xl[1]*0.96)
plt.show()

# FREQ
var_f = []
dyn_nf, dyn_nt = np.shape(dyn_sim.dyn)
ydata_f = ACF_DATA_NEW[:, int(nt/2)]
xdata_f = np.zeros((nf))
xdata_f[0:int(nf/2)] = (-dyn_sim.df * np.linspace(int(nf/2)-1, 0,
                                                  int(nf/2)))
xdata_f[int(nf/2):] = (dyn_sim.df * np.linspace(0, int(nf/2)-1,
                                                int(nf/2)))
for i in range(nt):
    var_dum = 0
    for k in range(-nt, nt):
        if k+i < nt and k+i >= 0:
            var_dum += ydata_f[k+i]**2
        if k-i >= 0 and k-i < nt and k+i < nt and k+i >= 0:
            var_dum += ydata_f[k-i]*ydata_f[k+i]
        if k >= 0 and k < nt:
            var_dum += 2*ydata_f[i]**2*ydata_f[k]**2
        if k+i < nt and k+i >= 0 and k >= 0 and k < nt:
            var_dum -= 4*ydata_f[i]*ydata_f[k]*ydata_f[k+i]
    var_f.append(var_dum)

var_f_arr = np.asarray(var_f)
sigma_sqr_f = np.sqrt(var_f_arr / (dyn_nf))  # N should be the len(dyn)

sigma_sqr_f_solv = sigma_sqr_f[int(nf/2):]

plt.scatter(xdata_f, ydata_f, c='C0', label=r'ACF1D for $\dnu_d$')
plt.fill_between(xdata_f, ydata_f-sigma_sqr_f, ydata_f+sigma_sqr_f, alpha=0.2,
                 label='Bartlett Errors 1D')
xl = plt.xlim()
plt.xlabel('Lags (MHz)')
plt.ylabel('ACF Value')
# plt.xlim(np.max(xdata_f)/2, xl[1]*0.96)
plt.show()

# Online solution on frequency
ydata_f_solv = ydata_f[int(nf/2):]
xdata_f_solv = xdata_f[int(nf/2):]
variance_f_solv = (np.ones(np.shape(ydata_f_solv)) / (nf / 2))
variance_f_solv[0] = 1e-10
variance_f_solv[2:] *= 1 + 2 * np.cumsum(ydata_f_solv[1:-1] ** 2)
f_errors = np.sqrt(variance_f_solv)

plt.scatter(xdata_f_solv, ydata_f_solv, s=1, c='k',
            label=r'ACF1D for $\tau_d$')
plt.errorbar(xdata_f_solv, ydata_f_solv, yerr=sigma_sqr_f_solv, fmt=' ',
             ecolor='C0', elinewidth=0.5, capsize=0.5, alpha=0.3)
plt.errorbar(xdata_f_solv, ydata_f_solv, yerr=t_errors, fmt=' ',
             ecolor='C1', elinewidth=0.5, capsize=0.5, alpha=0.3)
# plt.fill_between(xdata_f[int(nt/2):], ydata_f[int(nt/2):]-t_errors,
# ydata_f[int(nt/2):]+t_errors, alpha=0.2,
#                  label='Bartlett Errors 1D')
xl = plt.xlim()
plt.xlabel('Frequency Lags (MHz)')
plt.ylabel('ACF Value')
# plt.xlim(np.max(xdata_f_solv)/2, xl[1]*0.96)
plt.show()

sigma_sqr_f_solv = sigma_sqr_f[int(nf/2):]

# Everything PLOT

fig, ax = plt.subplots(2)
fig.suptitle('     ACF1D with Bartlett Errors')
ax[0].scatter(xdata_t_solv, sigma_sqr_t_solv, s=5, c='C0', alpha=0.3,
              label=r'Bartlett Errors 1D')
ax[0].scatter(xdata_t_solv, t_errors, s=5, c='C1', alpha=0.3,
              label=r'Expected Errors 1D')
ax[0].scatter(xdata_t_solv[1:], (sigma_sqr_t_solv/t_errors)[1:], s=5, c='C2',
              alpha=1,
              label=r'Bartlett/Expected')
ax[0].set_xlabel('Time Lags (s)')
ax[0].set_ylabel('ACF Value')
# ax[0].set_xlim(np.max(xdata_t)/2, np.max(xdata_t)*1.04)
# ax[0].set_ylim(0, 2)
ax[0].legend(fontsize='x-small')

ax[1].scatter(xdata_f_solv, sigma_sqr_f_solv, s=5, c='C0', alpha=0.3,
              label=r'Expected Errors 1D')
ax[1].scatter(xdata_f_solv, f_errors, s=5, c='C1', alpha=0.3,
              label=r'Bartlett Errors 1D')
ax[1].scatter(xdata_f_solv[1:], (sigma_sqr_f_solv/f_errors)[1:], s=5, c='C2',
              alpha=1,
              label=r'Bartlett/Expected')
ax[1].set_xlabel('Frequency Lags (MHz)')
ax[1].set_ylabel('ACF Value')
# ax[1].set_xlim(np.max(xdata_f)/2, np.max(xdata_f)*1.04)
# ax[1].set_ylim(0, 2)
ax[1].legend(fontsize='x-small')
plt.tight_layout()
plt.show()

# fig.suptitle('ACF1D with Bartlett Errors')
# # ax[0].scatter(xdata_t, ydata_t, s=5, c='k', alpha=0.5,
# #               label=r'ACF1D for tau')
# plt.scatter(xdata_t, sigma_sqr_t, s=5, c='C0', alpha=1,
#               label=r'Bartlett Errors 1D')
# plt.scatter(xdata_t, t_errors, s=5, c='C1', alpha=1,
#               label=r'Expected Errors 1D')
# # ax[0].scatter(xdata_t, t_errors/sigma_sqr_t, s=5, c='C2', alpha=1,
# #               label=r'Expected/Bartlett')
# plt.scatter(xdata_t[1:], (sigma_sqr_t/t_errors)[1:], s=5, c='C2', alpha=1,
#               label=r'Bartlett/Expected')
# # ax[0].fill_between(xdata_t, ydata_t-sigma_sqr_t, ydata_t+sigma_sqr_t,
# #                    alpha=0.3, label='Bartlett Errors 1D', color='C0')
# # ax[0].fill_between(xdata_t, ydata_t-t_errors, ydata_t+t_errors, alpha=0.3,
# #                  label='Expected Errors 1D', color='C1')
# ax.set_xlabel('Lags in time (s)')
# ax.set_ylabel('ACF Value')
# ax.set_xlim(np.max(xdata_t)/2, np.max(xdata_t)*1.04)
# ax.set_ylim(0, 2)
# ax.legend(fontsize='xx-small')
# plt.show()


# 2D Bartlett

nf, nt = np.shape(ACF_DATA_NEW)


var2D_init = []
var2D = np.zeros((nf, nt))


acf_data = ACF_DATA_NEW

# This kind of works 2D Bartlett

for i in range(0, nf):
    for j in range(0, nt):
        var_dum = 0
        for k in range(-nf, nf):
            for m in range(-nt, nt):
                if k+i >= 0 and k+i < nf and m+j >= 0 and m+j < nt:
                    var_dum += acf_data[k+i, m+j]**2
                if k+i >= 0 and k+i < nf and m+j >= 0 and m+j < nt and \
                        k-i >= 0 and k-i < nf and m-j >= 0 and m-j < nt:
                    var_dum += acf_data[k-i, m-j]*acf_data[k+i, m+j]
                if k >= 0 and k < nf and m >= 0 and m < nt:
                    var_dum += 2*acf_data[i, j]**2*acf_data[k, m]**2
                if k >= 0 and k < nf and m >= 0 and m < nt and k+i >= 0 and \
                        k+i < nf and m+j >= 0 and m+j < nt:
                    var_dum -= 4*acf_data[i, j]*acf_data[k, m]*acf_data[k+i,
                                                                        m+j]
        var2D[i, j] = var_dum / ((nf/2 - abs(i - nf/2))*(nt/2 - abs(j - nt/2)))
sigma_sqr2D = np.sqrt(var2D)
weights2D = 1/sigma_sqr2D

# for i in range(0, nf):
#     for j in range(0, nt):
#         var_dum = 0
#         for k in range(-nf, nf):
#             for m in range(-nt, nt):
#                 var_dum1 = 0
#                 var_dum2 = 1
#                 var_dum3 = 1
#                 var_dum4 = 1
#                 var_dum5 = 1
#                 var_dum6 = 1
#                 var_dum7 = 1
#                 var_dum8 = 1
#                 if k+i >= 0 and k+i < nf and m+j >= 0 and m+j < nt:
#                     var_dum1 = acf_data[k+i, m+j]**2
#                 if k-i >= 0 and k-i < nf and m-j >= 0 and m-j < nt:
#                     var_dum2 = acf_data[k-i, m-j]
#                 if k+i >= 0 and k+i < nf and m+j >= 0 and m+j < nt:
#                     var_dum3 = acf_data[k+i, m+j]
#                 var_dum4 = 2*acf_data[i, j]**2
#                 if k >= 0 and k < nf and m >= 0 and m < nt:
#                     var_dum5 = acf_data[k, m]**2
#                 var_dum6 = 4*acf_data[i, j]
#                 if k >= 0 and k < nf and m >= 0 and m < nt:
#                     var_dum7 = acf_data[k, m]
#                 if k+i >= 0 and k+i < nf and m+j >= 0 and m+j < nt:
#                     var_dum8 = acf_data[k+i, m+j]
#                 if var_dum2 == 1 and var_dum3 == 1:
#                     var_dum2 = 0
#                 if var_dum4 == 1 and var_dum5 == 1:
#                     var_dum4 = 0
#                 if var_dum6 == 1 and var_dum7 == 1 and var_dum8 == 1:
#                     var_dum6 = 0
#                 var_dum = var_dum1 + (var_dum2*var_dum3) + \
#                     (var_dum4*var_dum5) - (var_dum6*var_dum7*var_dum8)
#         var2D[i, j] = var_dum / ((nf-i)*(nt-j))
# sigma_sqr2D = np.sqrt(var2D)
# weights2D = 1/sigma_sqr2D

# I think we need to go negative!!!

# for i in range(-nf, nf):
#     for j in range(-nt, nt):
#         var_dum = 0
#         for k in range(-nf, nf):
#             for m in range(-nt, nt):
#                 if k+i < nf and k+i > -nf and m+j > -nt and m+j < nt \
#                         and k-i > -nf and k-i < nf and m-j > \
# -nt and m-j < nt \
#                         and k > -nf and k < nf and m < nt and m > -nt:
#                     var_dum += acf_data[k+i, m+j]**2 + \
#                         acf_data[k-i, m-j]*acf_data[k+i, m+j] + \
#                             2*acf_data[i, j]**2*acf_data[k, m]**2 - \
#                                 4*acf_data[i, j]*acf_data[k, m] * \
#                                     acf_data[k+i, m+j]
#         var2D[i, j] = var_dum / ((nf-i)*(nt-j))
# sigma_sqr2D = np.sqrt(var2D)
# weights2D = 1/sigma_sqr2D


# for i in range(0, nf):
#     for j in range(0, nt):
#         var_dum = 0
#         for k in range(-nf, nf):
#             for m in range(-nt, nt):
#                 if k+i >= 0 and k+i < nf and m+j >= 0 and m+j < nt:
#                     var_dum += acf_data[k+i, m+j]**2
#                 if k+i >= 0 and k+i < nf and m+j >= 0 and m+j < nt and \
#                         k-i >= 0 and k-i < nf and m-j >= 0 and m-j < nt:
#                     var_dum += acf_data[k-i, m-j]*acf_data[k+i, m+j]
#                 if k >= 0 and k < nf and m >= 0 and m < nt:
#                     var_dum += 2*acf_data[i, j]**2*acf_data[k, m]**2
#                 if k >= 0 and k < nf and m >= 0 and m < nt and k+i >= 0 and \
#                         k+i < nf and m+j >= 0 and m+j < nt:
#                     var_dum -= 4*acf_data[i, j]*acf_data[k, m]*acf_data[k+i,
# m+j]
#         var2D[i, j] = var_dum / ((nf-i)*(nt-j))

# NUM = nf*nt*(nf*2)*(nt*2)
# valid_indices = []
# for i in range(0, nf):
#     for j in range(0, nt):
#         for k in range(-nf, nf):
#             for m in range(-nt, nt):
#                 valid_indices.append(np.asarray([i, j, k, m]))

# valid_indices = np.asarray(valid_indices)

# plt.plot(valid_indices[:, 0], c='C0')
# plt.plot(valid_indices[:, 1], c='C1')
# plt.plot(valid_indices[:, 2], c='C2', alpha=0.2)
# plt.plot(valid_indices[:, 3], c='C3', alpha=0.2)


# var2D = var_dum / ((nf-np.arange(nf)[:,None])*(nt-np.arange(nt)))
# # weights2D = 1/sigma_sqr2D
# error_2dacf_topright = var2D[int(nt/2):, int(nf/2):]
# error_2dacf_topleft = np.flip(error_2dacf_topright, axis=0)
# error_2dacf_top = np.concatenate((error_2dacf_topleft, error_2dacf_topright),
#                                  axis=0)
# error_2dacf_bottom = np.flip(error_2dacf_top, axis=1)
# sigma_sqr2D = np.concatenate((error_2dacf_bottom, error_2dacf_top),
#                              axis=1)
# weights2D = 1/(sigma_sqr2D)


tticks = np.linspace(-dyn_sim.tobs, dyn_sim.tobs, nt + 1)[:-1]  # +1
fticks = np.linspace(-dyn_sim.bw, dyn_sim.bw, nf + 1)[:-1]  # +1
tedges = centres_to_edges(tticks/60)[1:]
fedges = centres_to_edges(fticks)[1:]

cb = plt.pcolormesh(tedges, fedges, ACF_DATA_NEW,
                    linewidth=0, rasterized=True, shading='auto')
plt.colorbar(cb)
plt.show()

cb = plt.pcolormesh(tedges, fedges, sigma_sqr2D,
                    linewidth=0, rasterized=True, shading='auto')
plt.colorbar(cb)
plt.show()

cb = plt.pcolormesh(tedges, fedges, weights2D,
                    linewidth=0, rasterized=True, shading='auto')
plt.colorbar(cb)
plt.show()

error_2d = np.ones((nf, nt)) * np.inf

for i in range(0, nf):
    for j in range(0, nt):
        if ((nf/2 - abs(i - nf/2))*(nt/2 - abs(j - nt/2))) != 0:
            error_2d[i, j] = 1 / ((nf/2 - abs(i - nf/2))*(nt/2 - abs(j -
                                                                     nt/2)))

cb = plt.pcolormesh(tedges, fedges, error_2d,
                    linewidth=0, rasterized=True, shading='auto')
plt.colorbar(cb)
plt.show()

cb = plt.pcolormesh(tedges, fedges, 1/error_2d,
                    linewidth=0, rasterized=True, shading='auto')
plt.colorbar(cb)
plt.show()

###############################################################################
# sim = Simulation(mb2=200, seed=64)
# dyn_sim = Dynspec(dyn=sim, process=False)
# dyn_sim.plot_dyn()
# # dyn.get_scint_params(method='acf1d', plot=True)
# # dyn_sim.get_scint_params(method='acf2d_approx', cutoff=True,
# # weights_2dacf=True,
# #                          redchisqr=True, plot=True)
# dyn_sim.get_acf_tilt(method='acf2d_approx', plot=True, phasewrapper=True)
# # dyn_cropped.plot_acf(input_acf=dyn_cropped.acf)
# dyn_sim.get_scint_params(method='acf2d_approx', plot=True, cutoff=True,
#                          weights_2dacf=weights_2dacf,
#                          redchisqr=weights_2dacf, phasewrapper=True)
# print("Scint timescale:", dyn_sim.tau, "+/-", dyn_sim.tauerr, "s")
# print("Scint bandwidth:", dyn_sim.dnu, "+/-", dyn_sim.dnuerr, "MHz")
# print("tilt in the acf:", dyn_sim.acf_tilt, "+/-", dyn_sim.acf_tilt_err,
#       "mins/MHz")
# print("phasegrad in the acf:", dyn_sim.phasegrad, "+/-",
# dyn_sim.phasegraderr,
#       "mins/MHz")

# ###############################################################################
# # Here I am testing the bartlett formula for determining weights ...

# sim = Simulation(mb2=10, seed=64)
# dyn = Dynspec(dyn=sim, process=False)
# dyn.plot_dyn()
# dyn.get_scint_params(method='acf2d_approx', plot=True)

# nf, nt = np.shape(dyn.acf)

# true_acf = dyn.acf

# # Use Bartlett's formula from Brockwell and Davis (1991),
# # Eqn 7.2.5
# ydata_f = dyn.acf[int(nf/2):, int(nt/2)]
# xdata_f = dyn.df * np.linspace(0, len(ydata_f)-1, len(ydata_f))
# ydata_t = dyn.acf[int(nf/2), int(nt/2):]
# xdata_t = dyn.dt * np.linspace(0, len(ydata_t)-1, len(ydata_t))

# variance_t = np.ones(np.shape(ydata_t)) / int(nt / 2)
# # variance_t[0] = 1/(nf * nt)
# variance_t[2:] *= 1 + 2 * np.cumsum(ydata_t[1:-1] ** 2)
# t_errors = np.sqrt(variance_t)

# variance_f = np.ones(np.shape(ydata_f)) / int(nf / 2)
# # variance_f[0] = 1/(nf * nt)
# variance_f[2:] *= 1 + 2 * np.cumsum(ydata_f[1:-1] ** 2)
# f_errors = np.sqrt(variance_f)
# error_2dacf = np.zeros((int(nf/2), int(nt/2)))

# fig, (ax0, ax1) = plt.subplots(nrows=2)
# ax0.plot(xdata_t/60, ydata_t)
# ax0.fill_between(xdata_t/60, ydata_t+t_errors, ydata_t-t_errors, color='C0',
#                  alpha=0.4, label='error')
# ax1.plot(xdata_f, ydata_f)
# ax1.fill_between(xdata_f, ydata_f+f_errors, ydata_f-f_errors, color='C0',
#                  alpha=0.4, label='error')
# ax0.set_xlabel(r"$\tau_d=$"+str(round(dyn.tau, 4)))
# ax0.set_ylabel("ACF Value")
# ax1.set_xlabel(r"$\Delta\nu_d=$"+str(round(dyn.dnu, 4)))
# ax1.set_ylabel("ACF Value")
# fig.tight_layout()
# plt.show()
# plt.close()

# true_acf_pos = true_acf[int(nf/2):, int(nt/2):]

# for i in range(0, int(nf / 2)):
#     for ii in range(0, int(nt / 2)):
#         error_2dacf[i, ii] = \
#             np.sqrt(((t_errors[ii]*true_acf_pos[i, ii])/ydata_t[ii])**2 +
#                     ((f_errors[i]*true_acf_pos[i, ii])/ydata_f[i])**2)

# error_2dacf_topright = error_2dacf
# error_2dacf_topleft = np.flip(error_2dacf_topright, axis=0)
# error_2dacf_top = np.concatenate((error_2dacf_topleft, error_2dacf_topright),
#                                  axis=0)
# error_2dacf_bottom = np.flip(error_2dacf_top, axis=1)
# error_2dacf = np.concatenate((error_2dacf_bottom, error_2dacf_top),
#                              axis=1)
# weights_2dacf = 1/(error_2dacf)

# tticks = np.linspace(-dyn.tobs, dyn.tobs, nt + 1)[:-1]  # +1
# fticks = np.linspace(-dyn.bw, dyn.bw, nf + 1)[:-1]  # +1
# tedges = centres_to_edges(tticks/60)[1:]
# fedges = centres_to_edges(fticks)[1:]
# #
# index_min = np.argmin(abs(-30 - fedges))
# index_max = np.argmin(abs(30 - fedges))
# zoomed = np.arange(index_min, index_max)
# zoomed2, zoomed3 = np.mgrid[index_min:index_max, index_min:index_max]
# #
# fig, (ax0, ax1, ax2) = plt.subplots(ncols=3, sharex=True, sharey=True,
#                                     figsize=(9, 3))
# ax0.pcolormesh(tedges[zoomed], fedges[zoomed], true_acf[zoomed2, zoomed3],
#                linewidth=0, rasterized=True, shading='auto')
# ax0.set_ylabel("Frequency lags (MHz)")
# ax1.pcolormesh(tedges[zoomed], fedges[zoomed], error_2dacf[zoomed2, zoomed3],
#                linewidth=0, rasterized=True, shading='auto')
# ax1.set_xlabel("Time lags (mins)")
# ax2.pcolormesh(tedges[zoomed], fedges[zoomed], weights_2dacf[zoomed2,
# zoomed3],
#                linewidth=0, rasterized=True, shading='auto')
# plt.tight_layout()
# plt.show()
# plt.close()
# #
# # plt.contourf(tedges, fedges, error_2dacf)
# # plt.show()
# # plt.close()

# # fig, ax = plt.subplots()
# # cf = plt.pcolormesh(tedges, fedges, weights_2dacf, linewidth=0,
# #                     rasterized=True, shading='auto', vmin=-1, vmax=1)
# # fig.colorbar(cf)
# # plt.xlabel("Time lags (mins)")
# # plt.ylabel("Frequency lags (MHz)")
# # plt.xlim(-40, 40)
# # plt.ylim(-1000/60, 1000/60)
# # plt.show()
# # plt.close()
# # # plt.contourf(tedges, fedges, weights_2dacf)
# # # plt.show()
# # # plt.close()

# # plt.plot(tedges, error_2dacf[0, :])
# # plt.xlabel("Time lags (mins)")
# # plt.ylabel("Error Values")
# # plt.show()
# # plt.close()
# # plt.plot(fedges, error_2dacf[:, 0])
# # plt.xlabel("Frequency lags (MHz)")
# # plt.ylabel("Error Values")
# # plt.show()
# # plt.close()

# # variable = (50, 50)

# # find central indices
# # i1, j1 = np.array(dyn.acf.shape) // 2
# # i2, j2 = np.array(variable) // 2
# # error_2dacf = error_2dacf[i1-i2:i1-i2+variable[0],
# #                           j1-j2:j1-j2+variable[1]]
# # weights_2dacf = weights_2dacf[i1-i2:i1-i2+variable[0],
# #                               j1-j2:j1-j2+variable[1]]
# # tedges = tedges[i1-i2:i1-i2+variable[0], j1-j2:j1-j2+variable[1]]
# # fedges = fedges[i1-i2:i1-i2+variable[0], j1-j2:j1-j2+variable[1]]

# # plt.pcolormesh(tedges, fedges, error_2dacf, linewidth=0, rasterized=True,
# #                shading='auto')
# # plt.show()
# # plt.close()
# # plt.contourf(tedges, fedges, error_2dacf)
# # plt.show()
# # plt.close()
# ###############################################################################


# # def scint_acf_model_2d_approx(tdata, fdata, ydata, weights, Alpha):
# #     """
# #     Fit an approximate 2D ACF function
# #     """

# #     amp = 1
# #     dnu = 2
# #     tau = 120
# #     alpha = Alpha
# #     mu = 0*60  # min/MHz to s/MHz
# #     tobs = 6000
# #     bw = 100
# #     wn = 0
# #     nt = len(tdata)
# #     nf = len(fdata)

# #     tdata = np.reshape(tdata, (nt, 1))
# #     fdata = np.reshape(fdata, (1, nf))

# #     # model = amp * np.exp(-(abs((tdata / tau) + 2 * phasegrad *
# #     #                           ((dnu / np.log(2)) / freq)**(1 / 6) *
# #     #                           (fdata / (dnu / np.log(2))))**(3 * alpha
# / 2) +
# #     #                     abs(fdata / (dnu / np.log(2)))**(3 / 2))**(2
# / 3))
# #     model = amp * np.exp(-(abs((tdata - mu*fdata)/tau)**(3 * alpha / 2) +
# #                          abs(fdata / (dnu / np.log(2)))**(3 / 2))**(2 / 3))

# #     # multiply by triangle function
# #     model = np.multiply(model, 1-np.divide(abs(tdata), tobs))
# #     model = np.multiply(model, 1-np.divide(abs(fdata), bw))
# #     model = np.fft.fftshift(model)
# #     model[-1, -1] += wn  # add white noise spike
# #     model = np.fft.ifftshift(model)
# #     model = np.transpose(model)

# #     if weights is None:
# #         weights = np.ones(np.shape(ydata))

# #     return model


# sim = Simulation(mb2=10, ns=200, freq=400, nf=200, seed=64)
# dyn = Dynspec(dyn=sim, process=False)
# dyn.get_scint_params(method='acf2d_approx', plot=False)

# nf, nt = np.shape(dyn.acf)

# tticks = np.linspace(-6000, 6000, nt + 1)[:-1]  # +1
# fticks = np.linspace(-100, 100, nf + 1)[:-1]  # +1

# tdata = tticks
# fdata = fticks
# ydata = dyn.acf
# weights = None
# model_2 = scint_acf_model_2d_approx(tdata, fdata, ydata, weights, Alpha=2)
# model_5 = scint_acf_model_2d_approx(tdata, fdata, ydata, weights, Alpha=5/3)
# model_1 = scint_acf_model_2d_approx(tdata, fdata, ydata, weights, Alpha=1)


# tedges = centres_to_edges(tticks/60)[1:]
# fedges = centres_to_edges(fticks)[1:]

# fig, ax = plt.subplots()
# cf = plt.pcolormesh(tedges, fedges, model_2, linewidth=0, rasterized=True,
#                     shading='auto')
# fig.colorbar(cf)
# plt.xlabel("Time lags (mins)")
# plt.ylabel("Frequency lags (MHz)")
# plt.xlim(-10, 10)
# plt.ylim(-10, 10)
# plt.show()
# plt.close()

# fig, ax = plt.subplots()
# cf = plt.pcolormesh(tedges, fedges, model_5, linewidth=0, rasterized=True,
#                     shading='auto')
# fig.colorbar(cf)
# plt.xlabel("Time lags (mins)")
# plt.ylabel("Frequency lags (MHz)")
# plt.xlim(-10, 10)
# plt.ylim(-10, 10)
# plt.show()
# plt.close()

# fig, ax = plt.subplots()
# cf = plt.pcolormesh(tedges, fedges, model_1, linewidth=0, rasterized=True,
#                     shading='auto')
# fig.colorbar(cf)
# plt.xlabel("Time lags (mins)")
# plt.ylabel("Frequency lags (MHz)")
# plt.xlim(-10, 10)
# plt.ylim(-10, 10)
# plt.show()
# plt.close()

# ###############################################################################
# sim = Simulation(mb2=10, seed=64)
# dyn = Dynspec(dyn=sim, process=False)
# dyn.plot_dyn()
# dyn.get_scint_params(method='acf2d_approx', redchisqr=True, plot=True,
#                      weights_2dacf=True, cutoff=True)
# dyn.get_scint_params(method='acf2d_approx', redchisqr=True, plot=True,
#                      weights_2dacf=True, cutoff=True)
# dyn.get_scint_params(method='acf1d', plot=True)
# # dyn.get_scint_params(method='acf2d_approx', plot=True)

# tspan = dyn.tobs
# fspan = dyn.bw

# t_delays = np.linspace(-tspan/60, tspan/60, np.shape(dyn.acf)[1])
# f_shifts = np.linspace(-fspan, fspan, np.shape(dyn.acf)[0])

# nr, nc = np.shape(dyn.dyn)
# nr2, nc2 = np.shape(dyn.acf)
# weights_2dacf = np.zeros((nr2, nc2))
# N_2dacf = weights_2dacf
# error_2dacf = weights_2dacf
# # N_dnu = nr  # the number of frequnecy channels in the dynspec
# N_dnu = np.unique(np.diff(f_shifts))[0]
# # N_tau = nc  # the number of subints in the dynspec
# N_tau = np.unique(np.diff(t_delays))[0]
# for i in range(-nr, nr):
#     for ii in range(-nc, nc):
#         if i == 0 and ii != 0:
#             N_2dacf[i, ii] = abs((N_tau-abs(ii))*(N_dnu-nr))
#         elif ii == 0 and i != 0:
#             N_2dacf[i, ii] = abs((N_dnu-abs(i))*(N_tau-nc))
#         elif i == 0 and ii == 0:
#             N_2dacf[i, ii] = nr*nc
#         else:
#             N_2dacf[i, ii] = abs((N_tau-abs(i))*(N_dnu-abs(ii)))
#         if N_2dacf[i, ii] == 0:
#             N_2dacf[i, ii] = 1e-3
# error_2dacf = 1/(np.sqrt(N_2dacf))
# weights_2dacf = 1/(error_2dacf)


# fig, ax1 = plt.subplots(figsize=(15, 15))
# ax1.contourf(t_delays, f_shifts, N_2dacf)
# ax1.set_ylabel(r'Frequency shift, $\Delta\nu$ (MHz)')
# ax1.set_xlabel(r'Time lag, $\tau$ (mins)')

# fig, ax1 = plt.subplots(figsize=(15, 15))
# ax1.contourf(t_delays, f_shifts, error_2dacf)
# ax1.set_ylabel(r'Frequency shift, $\Delta\nu$ (MHz)')
# ax1.set_xlabel(r'Time lag, $\tau$ (mins)')

# fig, ax1 = plt.subplots(figsize=(15, 15))
# ax1.contourf(t_delays, f_shifts, weights_2dacf)
# ax1.set_ylabel(r'Frequency shift, $\Delta\nu$ (MHz)')
# ax1.set_xlabel(r'Time lag, $\tau$ (mins)')

# Weights2D = np.zeros((nr2, nc2))
# N_dnu = np.unique(np.diff(f_shifts))[0]
# # N_dnu = nr
# N_tau = np.unique(np.diff(t_delays))[0]
# # N_tau = nc
# for i in range(-nr, nr):
#     for ii in range(-nc, nc):
#         if i == 0 and ii != 0:
#             Weights2D[i, ii] = 1/(1/(np.sqrt((N_dnu-abs(ii))*(N_tau-nr))))
#         elif ii == 0 and i != 0:
#             Weights2D[i, ii] = 1/(1/(np.sqrt((N_tau-abs(i))*(N_dnu-nc))))
#         elif ii == 0 and i == 0:
#             Weights2D[i, ii] = 1/(1/(np.sqrt(nr*nc)))
#         else:
#        Weights2D[i, ii] = 1/(1/(np.sqrt((N_tau-abs(i))*(N_dnu-abs(ii)))))

# fig, ax1 = plt.subplots(figsize=(15, 15))
# ax1.contourf(t_delays, f_shifts, Weights2D)
# ax1.set_ylabel(r'Frequency shift, $\Delta\nu$ (MHz)')
# ax1.set_xlabel(r'Time lag, $\tau$ (mins)')
# ###############################################################################
# # # This section creates the simulated spectra that we can play with #

# z = 735
# c = 3*10**8
# f_MHz = 800
# f = f_MHz*10**6
# wavelength = c/f
# k = (2*np.pi)/wavelength
# rf = np.sqrt(z/k)
# dnu_c = 0.05
# mb2 = 0.773*(f_MHz/dnu_c)**(5/6)
# outfile = '/Users/jacobaskew/Desktop/test.txt'
# resolution = 0.1
# NF = int(round(f_MHz/(resolution*dnu_c), -1))
# wd = '/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/'
# simspecfile = wd+'Simulation/Dynspec/SimDynspec_'+str(resolution)+'.dynspec'
# if os.path.exists(simspecfile):
#     sim = Simulation()
#     dyn_initial = Dynspec(dyn=sim, process=False)
#     dyn_initial.load_file(filename=simspecfile)
#     dyn_initial.trim_edges()
#     dyn_initial.plot_dyn(filename='/Users/jacobaskew/Desktop/SimSpectra_' +
#                          str(resolution) + '.png', dpi=400)
# else:
#     sim = Simulation(mb2=mb2, rf=rf, ds=0.01, alpha=5/3, ar=1, psi=0,
#                      inner=0.001, ns=2048, nf=NF,
#                      dlam=0.0375, lamsteps=False, seed=64, nx=None,
#                      ny=None, dx=None, dy=None, plot=False, verbose=False,
#                      freq=800, dt=8, mjd=50000, nsub=None, efield=False,
#                      noise=None)
#     dyn_initial = Dynspec(dyn=sim, process=False)
#     dyn_initial.trim_edges()
#     dyn_initial.plot_dyn(filename='/Users/jacobaskew/Desktop/SimSpectra_' +
#                          str(resolution) + '.png', dpi=400)
#     dyn_initial.write_file(filename=simspecfile)

# ###############################################################################

# # # Here we create an array for the ACF that can be turned into the 3D plot #

# dyn_initial.get_scint_params(method="acf2d_approx")
# data = dyn_initial.dyn
# acor = np.zeros((data.shape[0], data.shape[1]))
# # data0 = data[50, :]

# # This for loop just does the time lags
# acf_list = []
# for i in range(1, data.shape[1]):
#     x1 = data[50, :-i]
#     x2 = data[50, i:]
#     acor[50, i] = np.mean((x1 - np.mean(x1)) *
#                           (x2 - np.mean(x2))) / \
#         (np.std(x1) * np.std(x2))
#     acf_list.append(np.mean((x1 - np.mean(x1)) *
#                             (x2 - np.mean(x2))) /
#                     (np.std(x1) * np.std(x2)))

# plt.plot(acor)
# plt.xlim(40, 60)
# plt.show()
# plt.close()

# plt.pcolormesh(acor)
# plt.show()
# plt.close()

# # This does the same however indexes every freq lag
# acor = np.zeros((data.shape[0], data.shape[1]))

# for i in range(1, data.shape[1]):
#     for ii in range(1, data.shape[0]):
#         x1 = data[ii, :-i]
#         x2 = data[ii, i:]
#         acor[ii, i] = np.mean((x1 - np.mean(x1)) *
#                               (x2 - np.mean(x2))) / \
#             (np.std(x1) * np.std(x2))

# plt.plot(acor)
# plt.show()
# plt.close()

# plt.pcolormesh(acor)
# plt.show()
# plt.close()

# # This does the same but indexes the freq at the same time as the time lags

# acor = np.zeros((data.shape[0], data.shape[1]))

# for i in range(1, data.shape[1]):
#     for ii in range(1, data.shape[0]):
#         x1 = data[:-ii, :-i]
#         x2 = data[ii:, i:]
#         acor[ii, i] = np.mean((x1 - np.mean(x1)) *
#                               (x2 - np.mean(x2))) / \
#             (np.std(x1) * np.std(x2))

# plt.plot(acor)
# plt.show()
# plt.close()

# plt.pcolormesh(acor)
# plt.show()
# plt.close()

# # X = np.asarray(([2, 1, 5, 4, 2], [3, 4, 2, 1, 5]))
# # # X = np.random.random_sample((100, 100))

# # test_list = []
# # for i in range(0, X.shape[1]):
# #     for ii in range(0, X.shape[0]):
# #         if i == 0 and ii == 0:
# #             x_1 = X
# #             x_2 = X
# #         elif i == 0:
# #             x_1 = X[:ii, :]
# #             x_2 = X[ii:, :]
# #         elif ii == 0:
# #             x_1 = X[:, :i]
# #             x_2 = X[:, i:]
# #         else:
# #             x_1 = X[:ii, :i]
# #             x_2 = X[ii:, i:]
# #         print("test")
# #         print(x_1)
# #         print(x_2)
# #         test_list.append(np.mean((x_1 - np.mean(x_1)) *
# #                                  (x_2 - np.mean(x_2))) /
# #                                 (np.std(x_1) * np.std(x_2)))

# # plt.plot(test_list)
# # plt.show()
# # plt.close()

# # plt.pcolormesh(test_list)
# # plt.show()
# # plt.close()

# ###############################################################################

# # # Here we create the plots and outputs scintools makes without using it #


# Test = acor
# arr = Test
# print("done")
# tspan = dyn_initial.tobs
# fspan = dyn_initial.bw
# arr = np.fft.ifftshift(arr)
# wn = arr[0][0] - max([arr[1][0], arr[0][1]])
# arr[0][0] = arr[0][0] - wn
# arr = np.fft.fftshift(arr)

# t_delays = np.linspace(-tspan/60, tspan/60, np.shape(arr)[1])
# f_shifts = np.linspace(-fspan, fspan, np.shape(arr)[0])

# tlim = 4 * dyn_initial.tau / 60
# flim = 4 * dyn_initial.dnu
# if tlim > dyn_initial.tobs / 60:
#     tlim = dyn_initial.tobs / 60
# if flim > dyn_initial.bw:
#     flim = dyn_initial.bw


# fig, ax1 = plt.subplots(figsize=(15, 15))
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
#                                               dyn_initial.tau/60, 2)))
# plt.contourf(t_delays, f_shifts, arr)
# plt.ylabel('Frequency lag (MHz)')
# plt.xlabel('Time lag (mins)')
# plt.show()

# ###############################################################################

# # # This section is all about testing the script against scintools #

# dyn_initial.get_scint_params(method="nofit", plot=True, display=True)
# print()
# print("nofit")
# print("dnu=", round(dyn_initial.dnu, 3), "+/-", round(dyn_initial.dnuerr, 3))
# print("tau=", round(dyn_initial.tau, 3), "+/-", round(dyn_initial.tauerr, 3))

# dyn_initial.get_scint_params(method="acf1d", plot=True, display=True)
# print()
# print("acf1d")
# print("dnu=", round(dyn_initial.dnu, 3), "+/-", round(dyn_initial.dnuerr, 3))
# print("tau=", round(dyn_initial.tau, 3), "+/-", round(dyn_initial.tauerr, 3))

# dyn_initial.get_scint_params(method="acf2d_approx", plot=True, display=True)
# print()
# print("acf2d_approx")
# print("dnu=", round(dyn_initial.dnu, 3), "+/-", round(dyn_initial.dnuerr, 3))
# print("tau=", round(dyn_initial.tau, 3), "+/-", round(dyn_initial.tauerr, 3))

# dyn_initial.get_acf_tilt(plot=True, display=True)

# dyn_initial.plot_acf(contour=True, crop=False)
# ###############################################################################
# # In this area I am testing how cropping the acf could work
# z = 735
# c = 3*10**8
# f_MHz = 1000
# f = f_MHz*10**6
# wavelength = c/f
# k = (2*np.pi)/wavelength
# rf = np.sqrt(z/k)
# # mb2 = 0.773*(rf) * 100
# dnu_c = 0.1
# mb2 = 0.773*(f_MHz/dnu_c)**(5/6)

# sim = Simulation(mb2=mb2/4, ns=264, nf=264, dt=8, freq=f_MHz)
# dyn = Dynspec(dyn=sim, process=False)
# dyn.plot_dyn()
# dyn.get_scint_params(method='acf2d_approx')
# dyn.plot_acf(crop=True, contour=False)
# dyn.get_scint_params(method='acf1d', plot=True, nscale=10)
# dyn.get_scint_params(method='acf2d_approx', plot=True, nscale=10)
# print("acf2d_approx dnu:", dyn.dnu, '+/-', dyn.dnuerr)
# print("acf2d_approx tau:", dyn.tau, '+/-', dyn.tauerr)
# dyn.get_scint_params(method='acf1d', plot=True, nscale=5)
# dyn.get_scint_params(method='acf2d_approx', plot=True, nscale=5)
# print("acf2d_approx dnu:", dyn.dnu, '+/-', dyn.dnuerr)
# print("acf2d_approx tau:", dyn.tau, '+/-', dyn.tauerr)
# dyn.get_scint_params(method='acf1d', plot=True, nscale=4)
# dyn.get_scint_params(method='acf2d_approx', plot=True, nscale=4)
# print("acf2d_approx dnu:", dyn.dnu, '+/-', dyn.dnuerr)
# print("acf2d_approx tau:", dyn.tau, '+/-', dyn.tauerr)
# dyn.get_scint_params(method='acf1d', plot=True, nscale=3)
# dyn.get_scint_params(method='acf2d_approx', plot=True, nscale=3)
# print("acf2d_approx dnu:", dyn.dnu, '+/-', dyn.dnuerr)
# print("acf2d_approx tau:", dyn.tau, '+/-', dyn.tauerr)
# dyn.get_scint_params(method='acf1d', plot=True, nscale=2)
# dyn.get_scint_params(method='acf2d_approx', plot=True, nscale=2)
# print("acf2d_approx dnu:", dyn.dnu, '+/-', dyn.dnuerr)
# print("acf2d_approx tau:", dyn.tau, '+/-', dyn.tauerr)
# dyn.get_scint_params(method='acf1d', plot=True, nscale=1)
# dyn.get_scint_params(method='acf2d_approx', plot=True, nscale=1)
# print("acf2d_approx dnu:", dyn.dnu, '+/-', dyn.dnuerr)
# print("acf2d_approx tau:", dyn.tau, '+/-', dyn.tauerr)
# # plt.plot(dyn.acf)
# start_time = time.time()
# dyn.get_scint_params(method='acf1d', nscale=200, plot=True)
# print("===== %s seconds =====" % (time.time() - start_time))
# print("acf1d dnu:", dyn.dnu, '+/-', dyn.dnuerr)
# print("acf1d tau:", dyn.tau, '+/-', dyn.tauerr)
# start_time = time.time()
# dyn.get_scint_params(method='acf1d', nscale=10, plot=True)
# print("===== %s seconds =====" % (time.time() - start_time))
# print("acf1d dnu:", dyn.dnu, '+/-', dyn.dnuerr)
# print("acf1d tau:", dyn.tau, '+/-', dyn.tauerr)
# plt.plot(dyn.acf)
# plt.show()
# plt.close()
# ###############################################################################
# # In this area I am testing how cropping the acf could work
# z = 735
# c = 3*10**8
# f_MHz = 1000
# f = f_MHz*10**6
# wavelength = c/f
# k = (2*np.pi)/wavelength
# rf = np.sqrt(z/k)
# # mb2 = 0.773*(rf) * 100
# dnu_c = 0.1
# mb2 = 0.773*(f_MHz/dnu_c)**(5/6)

# sim = Simulation(mb2=mb2/4, ns=264, nf=264, dt=8, freq=f_MHz)
# dyn = Dynspec(dyn=sim, process=False)
# dyn.plot_dyn()
# dyn.get_scint_params(method='acf2d_approx', weights_2dacf=None)
# dyn.plot_acf(crop=True)
# dyn.get_scint_params(method='acf2d_approx', weights_2dacf=True)
# dyn.plot_acf(crop=True)


# def autocorr_func(data):
#     mean = np.ma.mean(data)
#     std = np.ma.std(data)
#     nr, nc = np.shape(data)
#     autocorr = np.zeros((2*nr, 2*nc))
#     for x in range(-nr+1, nr):
#         for y in range(-nc+1, nc):
#             segment1 = (data[max(0, x):min(x+nr, nr),
#                              max(0, y):min(y+nc, nc)] - mean)
#             segment2 = (data[max(0, -x):min(-x+nr, nr),
#                              max(0, -y):min(-y+nc, nc)] - mean)
#             numerator = np.ma.sum(np.ma.multiply(segment1, segment2))
#             autocorr[x+nr][y+nc] = numerator / (std ** 2)
#     print("len(segment1)", len(segment1))
#     print(len(segment2))
#     autocorr /= np.nanmax(autocorr)
#     return autocorr


# data = dyn.acf
# nr, nc = np.shape(data)
# # N = np.zeros((nr, nc))
# # Errors2D = np.zeros((nr, nc))
# tspan = dyn.tobs
# fspan = dyn.bw
# t_delays = np.linspace(-tspan/60, tspan/60, np.shape(data)[1])
# f_shifts = np.linspace(-fspan, fspan, np.shape(data)[0])
# Weights2D = np.zeros((nr, nc))
# N_dnu = np.unique(np.diff(f_shifts))[0]
# N_tau = np.unique(np.diff(t_delays))[0]
# nr, nc = np.shape(dyn.dyn)
# for i in range(-nr, nr):
#     for ii in range(-nc, nc):
#         if i == 0 and ii != 0:
#             Weights2D[i, ii] = 1/(1/(np.sqrt((N_dnu-abs(ii))*(N_tau-nr))))
#         elif ii == 0 and i != 0:
#             Weights2D[i, ii] = 1/(1/(np.sqrt((N_tau-abs(i))*(N_dnu-nc))))
#         elif ii == 0 and i == 0:
#             Weights2D[i, ii] = 1/(1/(np.sqrt(nr*nc)))
#         else:
#             Weights2D[i, ii] = \
# 1/(1/(np.sqrt((N_tau-abs(i))*(N_dnu-abs(ii)))))
# # nr, nc = np.shape(self.dyn)
# # nr2, nc2 = np.shape(self.acf)
# # weights_2dacf = np.zeros((nr2, nc2))
# # N_2dacf = weights_2dacf
# # error_2dacf = weights_2dacf
# # N_dnu = nr  # the number of frequnecy channels in the dynspec
# # N_tau = nc  # the number of subints in the dynspec
# # for i in range(-nr, nr):
# #     for ii in range(-nc, nc):
# #         if i == 0 and ii != 0:
# #             N_2dacf[i, ii] = abs((N_tau-abs(ii))*(N_dnu-nr2))
# #         elif ii == 0 and i != 0:
# #             N_2dacf[i, ii] = abs((N_dnu-abs(i))*(N_tau-nc2))
# #         elif ii == 0 and i == 0:
# #             N_2dacf[i, ii] = nr*nc
# #         else:
# #             N_2dacf[i, ii] = abs((N_tau-abs(i))*(N_dnu-abs(ii)))
# #         if N_2dacf[i, ii] == 0:
# #             N_2dacf[i, ii] = 1e-3
# # error_2dacf = 1/(np.sqrt(N_2dacf))
# # weights_2dacf = 1/(error_2dacf)

# # N_fft = np.fft.fftshift(N)
# Weights2D_fft = np.fft.fftshift(Weights2D)

# # fig, ax = plt.subplots(figsize=(15, 15))
# # # ax.contourf(t_delays, f_shifts, N_fft)
# # # ax.contourf(t_delays, f_shifts, N)
# # ax.pcolormesh(t_delays, f_shifts, N, linewidth=0, rasterized=True,
# #               shading='auto')
# # # ax.pcolormesh(t_delays, f_shifts, N_fft, linewidth=0, rasterized=True,
# # #               shading='auto')
# # plt.show()
# # plt.close()

# fig, ax = plt.subplots(figsize=(15, 15))
# # ax.contourf(t_delays, f_shifts, data)
# ax.pcolormesh(t_delays, f_shifts, data, linewidth=0, rasterized=True,
#               shading='auto')
# # plt.xlim(-10, 10)
# # plt.ylim(-50, 50)
# # plt.xlim(-2.5, 2.5)
# # plt.ylim(-5, 5)
# plt.show()
# plt.close()

# fig, ax = plt.subplots(figsize=(15, 15))
# # ax.contourf(t_delays, f_shifts, Weights2D_fft)
# # ax.contourf(t_delays, f_shifts, Weights2D)
# ax.pcolormesh(t_delays, f_shifts, Weights2D, linewidth=0, rasterized=True,
#               shading='auto')
# # ax.pcolormesh(t_delays, f_shifts, Weights2D_fft, linewidth=0,
# # rasterized=True,
# #               shading='auto')
# # plt.xlim(-10, 10)
# # plt.ylim(-50, 50)
# # plt.xlim(-2.5, 2.5)
# # plt.ylim(-5, 5)
# plt.show()
# plt.close()
# ###############################################################################
# # Here I want to run 1000 sims and see the difference between a fit using
# # weights and not using weights
# dnu = []
# dnuerr = []
# tau = []
# tauerr = []
# phasegrad = []
# phasegraderr = []
# scint_param_method = []
# dnu_weights = []
# dnuerr_weights = []
# tau_weights = []
# tauerr_weights = []
# phasegrad_weights = []
# phasegraderr_weights = []
# redchisqr_weights = []
# scint_param_method_weights = []
# for i in range(0, 1000):
#     sim = Simulation(mb2=10, freq=1000)
#     dyn = Dynspec(dyn=sim, process=False)
#     dyn.get_scint_params(method='acf2d_approx')
#     dnu.append(dyn.dnu)
#     dnuerr.append(dyn.dnuerr)
#     tau.append(dyn.tau)
#     tauerr.append(dyn.tauerr)
#     phasegrad.append(dyn.phasegrad)
#     phasegraderr.append(dyn.phasegraderr)
#     scint_param_method.append(dyn.scint_param_method)
#     dyn.get_scint_params(method='acf2d_approx', redchisqr=True,
#                          weights_2dacf=True, cutoff=True)
#     dnu_weights.append(dyn.dnu)
#     dnuerr_weights.append(dyn.dnuerr)
#     tau_weights.append(dyn.tau)
#     tauerr_weights.append(dyn.tauerr)
#     phasegrad_weights.append(dyn.phasegrad)
#     phasegraderr_weights.append(dyn.phasegraderr)
#     if dyn.scint_param_method == 'nofit':
#         redchisqr_weights.append(1e8)
#     else:
#         redchisqr_weights.append(dyn.acf_redchisqr)
#     scint_param_method_weights.append(dyn.scint_param_method)

# dnu = np.asarray(dnu)
# dnuerr = np.asarray(dnuerr)
# tau = np.asarray(tau)
# tauerr = np.asarray(tauerr)
# phasegrad = np.asarray(phasegrad)
# phasegraderr = np.asarray(phasegraderr)
# scint_param_method = np.asarray(scint_param_method)
# dnu_weights = np.asarray(dnu_weights)
# dnuerr_weights = np.asarray(dnuerr_weights)
# tau_weights = np.asarray(tau_weights)
# tauerr_weights = np.asarray(tauerr_weights)
# phasegrad_weights = np.asarray(phasegrad_weights)
# phasegraderr_weights = np.asarray(phasegraderr_weights)
# redchisqr_weights = np.asarray(redchisqr_weights)
# scint_param_method_weights = np.asarray(scint_param_method_weights)

# indicies = np.argwhere((1500 > tau) * (50 > dnu) *
#                        (scint_param_method == "acf2d_approx") *
#                        (1500 > tau_weights) *
#                        (50 > dnu_weights) *
#                        (scint_param_method_weights == "acf2d_approx"))

# dnu_filt = dnu[indicies].squeeze()
# dnuerr_filt = dnuerr[indicies].squeeze()
# tau_filt = tau[indicies].squeeze()
# tauerr_filt = tauerr[indicies].squeeze()
# phasegrad_filt = phasegrad[indicies].squeeze()
# phasegraderr_filt = phasegraderr[indicies].squeeze()
# scint_param_method_filt = scint_param_method[indicies].squeeze()
# dnu_weights_filt = dnu_weights[indicies].squeeze()
# dnuerr_weights_filt = dnuerr_weights[indicies].squeeze()
# tau_weights_filt = tau_weights[indicies].squeeze()
# tauerr_weights_filt = tauerr_weights[indicies].squeeze()
# phasegrad_weights_filt = phasegrad_weights[indicies].squeeze()
# phasegraderr_weights_filt = phasegraderr_weights[indicies].squeeze()
# redchisqr_weights_filt = redchisqr_weights[indicies].squeeze()
# scint_param_method_weights_filt = \
#     scint_param_method_weights[indicies].squeeze()

# # redchisqr_weights_norm = \
# (redchisqr_weights - np.min(redchisqr_weights)) / \
# #     (np.max(redchisqr_weights) - np.min(redchisqr_weights))

# Size = 80*np.pi  # Determines the size of the datapoints used
# font = {'size': 28}
# matplotlib.rc('font', **font)

# fig, ax = plt.subplots(figsize=(15, 15))
# cm = plt.cm.get_cmap('viridis')
# z = redchisqr_weights_filt
# sc = plt.scatter(dnu_filt, dnu_weights_filt, c=z, cmap=cm, s=Size, alpha=0.3)
# plt.errorbar(dnu_filt, dnu_weights_filt, xerr=dnuerr_filt,
#              yerr=dnuerr_weights_filt, fmt=' ', ecolor='k', elinewidth=2,
#              capsize=3, alpha=0.01)
# plt.colorbar(sc)
# xl = plt.xlim()
# yl = plt.ylim()
# plt.plot([xl[0], xl[1]],
#          [yl[0], yl[1]], color='k')
# plt.xlim(xl)
# plt.ylim(yl)
# plt.title("Scintillation Bandwidth")
# plt.xlabel(r"$\Delta\nu_d$ (MHz)")
# plt.ylabel(r"$\Delta\nu_d$ (Weighted) (MHz)")
# plt.show()
# plt.close()

# fig, ax = plt.subplots(figsize=(15, 15))
# cm = plt.cm.get_cmap('viridis')
# z = redchisqr_weights_filt
# sc = plt.scatter(tau_filt, tau_weights_filt, c=z, s=Size, cmap=cm, alpha=0.3)
# plt.errorbar(tau_filt, tau_weights_filt, xerr=tauerr_filt,
#              yerr=tauerr_weights_filt, fmt=' ', ecolor='k', elinewidth=2,
#              capsize=3, alpha=0.01)
# plt.colorbar(sc)
# xl = plt.xlim()
# yl = plt.ylim()
# plt.plot([xl[0], xl[1]],
#          [yl[0], yl[1]], color='k')
# plt.xlim(xl)
# plt.ylim(yl)
# plt.title("Scintillation Timescale")
# plt.xlabel(r"$\tau_d$ (s)")
# plt.ylabel(r"$\tau_d$ (Weighted) (s)")
# plt.show()
# plt.close()

# fig, ax = plt.subplots(figsize=(15, 15))
# cm = plt.cm.get_cmap('viridis')
# z = redchisqr_weights_filt
# sc = plt.scatter(phasegrad_filt, phasegrad_weights_filt, s=Size, c=z,
# cmap=cm,
#                  alpha=0.3)
# plt.errorbar(phasegrad_filt, phasegrad_weights_filt, xerr=phasegraderr_filt,
#              yerr=phasegraderr_weights_filt, fmt=' ', ecolor='k',
# elinewidth=2,
#              capsize=3, alpha=0.01)
# plt.colorbar(sc)
# xl = plt.xlim()
# yl = plt.ylim()
# plt.plot([xl[0], xl[1]],
#          [yl[0], yl[1]], color='k')
# plt.xlim(xl)
# plt.ylim(yl)
# plt.title("Phase Gradient")
# plt.xlabel(r"$\phi$ (degrees)")
# plt.ylabel(r"$\phi$ (Weighted) (degrees)")
# plt.show()
# plt.close()

# fig, ax = plt.subplots(figsize=(15, 15))
# plt.hist(redchisqr_weights_filt, bins=10, color='C0')
# plt.xlabel("Reduced chi-sqr")
# plt.ylabel("Frequency")
# plt.show()
# plt.close()
# ###############################################################################
