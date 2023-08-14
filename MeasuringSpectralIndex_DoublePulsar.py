#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 14:59:20 2023

@author: jacobaskew
"""
###############################################################################
# Importing neccessary things #
from scintools.scint_utils import read_par, pars_to_params
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
###############################################################################

desktopdir = '/Users/jacobaskew/Desktop/'
datadir = desktopdir + 'DoublePulsar_Project/datasets/FullDataStorage/'
par_dir = '/Users/jacobaskew/Desktop/Swinburne_Daniel/ParFiles/'
psrname = 'J0737-3039A'
wd0 = "/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/New/"
outdir = wd0 + "Modelling"
pars = read_par(str(par_dir) + str(psrname) + '.par')
params = pars_to_params(pars)

viss = np.loadtxt(datadir + 'Full_VissData.txt', dtype='float')
visserr = np.loadtxt(datadir + 'Full_VisserrData.txt', dtype='float')
mjd = np.loadtxt(datadir + 'Full_MJDData.txt', dtype='float')
freqMHz = np.loadtxt(datadir + 'Full_FreqData.txt', dtype='float')
freqGHz = freqMHz / 1e3
dnu = np.loadtxt(datadir + 'Full_DnuData.txt', dtype='float')
dnuerr = np.loadtxt(datadir + 'Full_DnuerrData.txt', dtype='float')
tau = np.loadtxt(datadir + 'Full_TauData.txt', dtype='float')
tauerr = np.loadtxt(datadir + 'Full_TauerrData.txt', dtype='float')
phase = np.loadtxt(datadir + 'Full_PhaseData.txt', dtype='float')
U = np.loadtxt(datadir + 'Full_UData.txt', dtype='float')
ve_ra = np.loadtxt(datadir + 'Full_ve_raData.txt', dtype='float')
ve_dec = np.loadtxt(datadir + 'Full_ve_decData.txt', dtype='float')

kwargs = {"U": U, "ve_ra": ve_ra, "ve_dec": ve_dec, "params": params}
###############################################################################
alpha_prime = 4.4
Beta = (2 * alpha_prime)/(alpha_prime - 2)
alpha = Beta - 2
###############################################################################

Size = 80*np.pi  # Determines the size of the datapoints used
font = {'size': 20}
matplotlib.rc('font', **font)
coeff = np.polynomial.polynomial.polyfit(np.log(freqMHz), np.log(dnu), 1,
                                         rcond=None, full=False,
                                         w=1/(dnuerr/dnu))
fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(1, 1, 1)
plt.errorbar(np.log(freqMHz), np.log(dnu), yerr=dnuerr/dnu, fmt='o')
plt.plot(np.log(freqMHz), coeff[1] * np.log(freqMHz) + coeff[0],
         label=r'Measured: $\alpha$='+str(round(coeff[1], 2)))
plt.plot(np.log(freqMHz), 4 * np.log(freqMHz) + -29.8,
         label=r'Kolmogorov: $\alpha$='+str(4), c='k')
plt.xlabel("Natural log Frequency (MHz)")
plt.ylabel("Natural log Scintilation Bandwidth (MHz)")
ax.legend()
plt.savefig("/Users/jacobaskew/Desktop/NaturalLogSpectralIndex.png")
plt.show()
plt.close()

freq_sort = np.argsort(freqMHz)
fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(1, 1, 1)
plt.errorbar(freqMHz, dnu, yerr=dnuerr, fmt='o')
plt.plot(freqMHz[freq_sort], (0.05*(freqMHz/800)**(coeff[1]))[freq_sort],
         label=r'Measured: $\alpha$='+str(round(coeff[1], 2)))
plt.plot(freqMHz[freq_sort], (0.05*(freqMHz/800)**(4))[freq_sort],
         label=r'Kolmogorov: $\alpha$='+str(4), c='k')
plt.xlabel("Frequency (MHz)")
plt.ylabel("Scintilation Bandwidth (MHz)")
ax.legend()
plt.savefig("/Users/jacobaskew/Desktop/SpectralIndex.png")
plt.show()
plt.close()
