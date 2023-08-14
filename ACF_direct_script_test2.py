#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 22:52:35 2020

@author: jacobaskew
"""
###############################################################################
from scintools.dynspec import Dynspec
# from scintools.scint_utils import read_results, float_array_from_dict, \
# from scintools.scint_utils import write_results
from scintools.scint_sim import Simulation
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy as cp
# from scipy.optimize import curve_fit
from scipy.odr import ODR, Model, RealData
# import os
# import csv
###############################################################################


def autocorr_func(data):
    mean = np.ma.mean(data)
    std = np.ma.std(data)
    nr, nc = np.shape(data)
    autocorr = np.zeros((2*nr, 2*nc))
    for x in range(-nr+1, nr):
        for y in range(-nc+1, nc):
            segment1 = (data[max(0, x):min(x+nr, nr),
                        max(0, y):min(y+nc, nc)] - mean)
            segment2 = (data[max(0, -x):min(-x+nr, nr),
                        max(0, -y):min(-y+nc, nc)] - mean)
            numerator = np.ma.sum(np.ma.multiply(segment1, segment2))
            autocorr[x+nr][y+nc] = numerator / (std ** 2)
    autocorr /= np.nanmax(autocorr)
    return autocorr


def line_fit(beta, x):
    y = beta[0]+beta[1]*x
    return y


# def line_fit_with_errors(x, y, x_err, y_err):
#     def fit_func(x, a, b):
#         return a * x + b
#     sigma = np.concatenate([x_err, y_err], axis=0)
#     print(sigma.shape)
#     params, cov = curve_fit(fit_func, x, y,
#                             sigma=sigma,
#                             absolute_sigma=True)
#     a, b = params
#     a_err, b_err = np.sqrt(np.diag(cov))
#     return a, b, a_err, b_err


###############################################################################
# simfile0 = "/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/New/"
# simfile = simfile0 + "Simulations/"
# if not os.path.exists(str(simfile)+"simulation.dynspec"):
#     sim = Simulation(mb2=150, seed=128, ns=1024, nf=1024, freq=800)
#     # sim = Simulation()
#     dyn = Dynspec(dyn=sim, process=False)
#     dyn.write_file(filename=str(simfile)+"simulation.dynspec")
# else:
#     sim = Simulation()
#     dyn = Dynspec(dyn=sim, process=False)
#     dyn.load_file(filename=str(simfile)+"simulation.dynspec")

# dyn.plot_dyn(dpi=400)

# dyn_crop1 = cp(dyn)
# tspan = dyn_crop1.tobs
# fspan = dyn_crop1.bw

# dyn_crop1.calc_acf()
# FFT_acf = dyn_crop1.acf
# t_delays = np.linspace(-tspan/60, tspan/60, np.shape(FFT_acf)[1])
# f_shifts = np.linspace(-fspan, fspan, np.shape(FFT_acf)[0])

# #

# fig, ax1 = plt.subplots(figsize=(9, 9))
# CS = ax1.contourf(t_delays, f_shifts, FFT_acf)  # ,
# #                  levels=np.linspace(-1, 1, 10))
# fig.colorbar(CS)
# plt.title("FFT_spectra")
# plt.savefig(
#     "/Users/jacobaskew/Desktop/2DACF_FFT_spectra.pdf",
#     dpi=400)
# plt.savefig(
#     "/Users/jacobaskew/Desktop/2DACF_FFT_spectra.png",
#     dpi=400)
# plt.show()
# plt.close()  # 15

# RFI_beg = 1000
# RFI_end = 1050
# tspan = dyn_crop1.tobs
# fspan = dyn_crop1.bw

# r1 = np.argmin(abs(RFI_beg - dyn_crop1.freqs))
# r2 = np.argmin(abs(RFI_end - dyn_crop1.freqs))
# dyn_crop1.dyn[r1:r2, :] = 0

# dyn_crop1.refill(method='median')
# dyn_crop1.plot_dyn(filename="/Users/jacobaskew/Desktop/Spectra_median.png",
#                    dpi=400)  # 14

# dyn_crop1.calc_acf()
# median_acf = dyn_crop1.acf
# t_delays = np.linspace(-tspan/60, tspan/60, np.shape(median_acf)[1])
# f_shifts = np.linspace(-fspan, fspan, np.shape(median_acf)[0])

# fig, ax1 = plt.subplots(figsize=(9, 9))
# CS = ax1.contourf(t_delays, f_shifts, median_acf)
# fig.colorbar(CS)
# plt.title("FFT_refilled_median")
# plt.xlim(-5, 5)
# plt.ylim(-5, 5)
# plt.savefig(
#     "/Users/jacobaskew/Desktop/2DACF_FFT_refilled_median.pdf",
#     dpi=400)
# plt.savefig(
#     "/Users/jacobaskew/Desktop/2DACF_FFT_refilled_median.png",
#     dpi=400)
# plt.show()
# plt.close()  # 15
###############################################################################
# I want to create 1000 simulated spectra of double pulsar as if they are
# chunks of an observation. I want to then test different methods and plot
# the resulting histograms

dnu_c = 0.2
centre_freq = 800
mb2 = 0.773*(centre_freq/dnu_c)**(5/6)

dnu1 = []
dnu2 = []
# dnu3 = []
dnuerr1 = []
dnuerr2 = []
# dnuerr3 = []
for i in range(0, 1000):
    sim = Simulation(mb2=mb2, rf=2, ns=76, nf=1000, dlam=0.0375,
                     freq=centre_freq, dt=8)
    dyn = Dynspec(dyn=sim, process=False)
    dyn.get_scint_params(method='acf2d_approx')
    dnu1.append(dyn.dnu)
    dnuerr1.append(dyn.dnuerr)
    #
    RFI_beg = 805
    RFI_end = 815
    r1 = np.argmin(abs(RFI_beg - dyn.freqs))
    r2 = np.argmin(abs(RFI_end - dyn.freqs))
    dyn.dyn[r1:r2, :] = 0
    # data = np.ma.masked_where(dyn.dyn == 0, dyn.dyn)
    # autocorr = autocorr_func(data)
    # dyn.acf = autocorr
    # dyn.get_scint_params(method='acf2d_approx')
    # dnu3.append(dyn.dnu)
    # dnuerr3.append(dyn.dnuerr)
    #
    dyn.refill(method='Mean')
    dyn.calc_acf()
    dyn.get_scint_params(method='acf2d_approx')
    dnu2.append(dyn.dnu)
    dnuerr2.append(dyn.dnuerr)
dnu1 = np.asarray(dnu1)
dnu2 = np.asarray(dnu2)
# dnu3 = np.asarray(dnu3)
dnuerr1 = np.asarray(dnuerr1)
dnuerr2 = np.asarray(dnuerr2)
# dnuerr3 = np.asarray(dnuerr3)

plt.hist(dnu1, bins=20, color='C0', alpha=0.4, density=True)
plt.hist(dnu2, bins=20, color='C1', alpha=0.4, density=True)
# plt.hist(dnu3, bins=20, color='C3', alpha=0.4, density=True)
plt.show()
plt.close()

data = RealData(dnu1, dnu2, dnuerr1, dnuerr2)
model = Model(line_fit)

odr = ODR(data, model, [1, 0, 0])
odr.set_job(fit_type=0)
output = odr.run()

fig = plt.figure(figsize=(9, 9))
ax = fig.add_subplot(1, 1, 1)
plt.scatter(dnu1, dnu2, color='C0', alpha=0.2)
plt.errorbar(dnu1, dnu2, yerr=dnuerr2, xerr=dnuerr1, fmt=' ', ecolor='C0',
             elinewidth=1, capsize=1, alpha=0.2)
yl = plt.ylim()
xl = plt.xlim()
plt.plot([0, xl[1]], [0, xl[1]], color='k')
plt.ylim(0, yl[1])
plt.xlim(0, xl[1])
xn = np.linspace(0, xl[1], 100)
yn = line_fit(output.beta, xn)
gradient = output.beta[1]
intercept = output.beta[0]
plt.plot(xn, yn, 'g-', label='m = ' + str(round(gradient, 3)) +
         '  b = ' + str(round(intercept, 3)))
plt.xlabel(r"'True' ACF $\Delta\nu_d$")
plt.ylabel(r"'Mean' ACF $\Delta\nu_d$")
ax.legend()
plt.show()
plt.close()

# a, b, aerr, berr = line_fit_with_errors(dnu1, dnu2, dnuerr1, dnuerr2)
###############################################################################
# In this section I wish to create code that takes a spectrum, finds the
# scint bandwidth and timescale with uncertainties.
# Then compares that to the same method but taking random chuncks of frequency
# or time of different sizes

dnu_c = 0.2
mb2 = 0.773*(1100/dnu_c)**(5/6)
sim = Simulation(mb2=mb2, rf=2, ns=76, nf=1000, dlam=0.0375, freq=800, dt=8,
                 seed=64)
dyn_true = Dynspec(dyn=sim, process=False)
dyn_true.plot_dyn(dpi=400)
dyn_true.get_scint_params(method='acf2d_approx')
dyn_true_dnu = dyn_true.dnu
dyn_true_dnuerr = dyn_true.dnuerr
dyn_true_tau = dyn_true.tau
dyn_true_tauerr = dyn_true.tauerr
print("dyn_true_dnu", dyn_true_dnu)
print("dyn_true_dnuerr", dyn_true_dnuerr)
print("dyn_true_tau", dyn_true_tau)
print("dyn_true_tauerr", dyn_true_tauerr)
for i in range(0, 31):
    step_freq = int(i)  # Decide how many MHz we want to flag
    step = int(step_freq / dyn_true.df)

    dyn_crop_dnu1 = []
    dyn_crop_dnuerr1 = []
    dyn_crop_dnu2 = []
    dyn_crop_dnuerr2 = []
    for i in range(1, dyn_true.dyn.shape[0]):
        if i != 0:
            i += step - 1
            if i + step - 1 > dyn_true.dyn.shape[0]:
                continue
        dyn_crop = cp(dyn_true)
        # f1 = np.argmin(abs(i - dyn_crop.freqs))
        # f2 = np.argmin(abs(i+step - dyn_crop.freqs))
        dyn_crop.dyn[i:i+step, :] = 0
        # dyn_crop.plot_dyn()
        # data = np.ma.masked_where(dyn_crop.dyn == 0, dyn_crop.dyn)
        # autocorr = autocorr_func(data)
        # dyn_crop.acf = autocorr
        # dyn_crop.get_scint_params(method='acf2d_approx')
        # dyn_crop_dnu1.append(dyn_crop.dnu)
        # dyn_crop_dnuerr1.append(dyn_crop.dnuerr)
        dyn_crop.refill('median')
        dyn_crop.calc_acf()
        dyn_crop.get_scint_params(method='acf2d_approx')
        dyn_crop_dnu1.append(dyn_crop.dnu)
        dyn_crop_dnuerr1.append(dyn_crop.dnuerr)
        #
        dyn_crop.dyn[i:i+step, :] = 0
        dyn_crop.refill('mean')
        dyn_crop.calc_acf()
        dyn_crop.get_scint_params(method='acf2d_approx')
        dyn_crop_dnu2.append(dyn_crop.dnu)
        dyn_crop_dnuerr2.append(dyn_crop.dnuerr)

    dyn_crop_dnu2 = np.asarray(dyn_crop_dnu2)
    dyn_crop_dnuerr2 = np.asarray(dyn_crop_dnuerr2)

    plt.scatter(dyn_crop_dnu1, dyn_crop_dnu2, color='C0', alpha=0.2)
    plt.errorbar(dyn_crop_dnu1, dyn_crop_dnu2, yerr=dyn_crop_dnuerr2,
                 xerr=dyn_crop_dnuerr2, fmt=' ', ecolor='C0', elinewidth=1,
                 capsize=1, alpha=0.2)
    plt.scatter(dyn_true_dnu, dyn_true_dnu, color='k')
    plt.errorbar(dyn_true_dnu, dyn_true_dnu, yerr=dyn_true_dnuerr,
                 xerr=dyn_true_dnuerr, fmt=' ', ecolor='k', elinewidth=1,
                 capsize=1)
    yl = plt.ylim()
    xl = plt.xlim()
    plt.plot([0, 0.6], [0, 0.6], color='k')
    plt.ylim(0.3, 0.6)
    plt.xlim(0.3, 0.6)
    plt.xlabel(r"'Median' ACF $\Delta\nu_d$")
    plt.ylabel(r"'Mean' ACF $\Delta\nu_d$")
    plt.title("Setting different Channels to 0, step=" +
              str(step_freq) + "MHz")
    plt.savefig("/Users/jacobaskew/Desktop/Tmp/ZappingChannels_" +
                str(step_freq))
    plt.show()
    plt.close()


###############################################################################
# plt.plot(x,y,'C3')
# # plot(xn,yn,'k-',label='leastsq')
# odr.set_job(fit_type=0)
# output = odr.run()
# yn = func(output.beta, xn)
# plot(xn,yn,'g-',label='odr')
# legend(loc=0)
