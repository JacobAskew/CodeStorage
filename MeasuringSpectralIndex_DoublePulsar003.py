#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 10:41:43 2023

@author: jacobaskew
"""

###############################################################################
# Importing neccessary things #
from scintools.scint_utils import read_par, pars_to_params, get_earth_velocity, \
    get_true_anomaly
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
import bilby
# from astropy.time import Time
from Alpha_Likelihood import AlphaLikelihood2
# import corner
###############################################################################
desktopdir = '/Users/jacobaskew/Desktop/'
datadir = '/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/New/DataFiles/'
datadir_Lband = '/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/New/Lband/HiRes/Data/'
datadir_Sband = '/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/New/Sband/HiRes/Data/'
par_dir = '/Users/jacobaskew/Desktop/Swinburne_Daniel/ParFiles/'
psrname = 'J0737-3039A'
wd0 = "/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/New/"
outdir = wd0 + "Modelling"
pars = read_par(str(par_dir) + str(psrname) + '.par')
params = pars_to_params(pars)
# UHF
mjd_UHF = np.loadtxt(datadir + '_mjd.txt', dtype='float')
freqMHz_UHF = np.loadtxt(datadir + '_freqMHz.txt', dtype='float')
dnu_UHF = np.loadtxt(datadir + '_dnu.txt', dtype='float')
dnuerr_UHF = np.loadtxt(datadir + '_dnuerr.txt', dtype='float')
# L-Band
mjd_L = np.loadtxt(datadir_Lband + '_mjd.txt', dtype='float')
freqMHz_L = np.loadtxt(datadir_Lband + '_freqMHz.txt', dtype='float')
dnu_L = np.loadtxt(datadir_Lband + '_dnu.txt', dtype='float')
dnuerr_L = np.loadtxt(datadir_Lband + '_dnuerr.txt', dtype='float')
# S-Band
mjd_S = np.loadtxt(datadir_Sband + '_mjd.txt', dtype='float')
freqMHz_S = np.loadtxt(datadir_Sband + '_freqMHz.txt', dtype='float')
dnu_S = np.loadtxt(datadir_Sband + '_dnu.txt', dtype='float')
dnuerr_S = np.loadtxt(datadir_Sband + '_dnuerr.txt', dtype='float')
# Combine
mjd = np.concatenate((mjd_UHF, mjd_L, mjd_S))
freqMHz = np.concatenate((freqMHz_UHF, freqMHz_L, freqMHz_S))
dnu = np.concatenate((dnu_UHF, dnu_L, dnu_S))
dnuerr = np.concatenate((dnuerr_UHF, dnuerr_L, dnuerr_S))
#
###############################################################################
# dnu_old = dnu.copy()
dnu_mean = []
freqMHz_mean = []
mjd_mean = []

num_obs = 9

# I want to take the average of dnu at each frequency at each observation
for i in range(0, len(np.unique(freqMHz))):
    for ii in range(0, num_obs):
        indices = \
            np.argwhere((np.unique(freqMHz)[i] == freqMHz) *
                        (np.unique(np.round(mjd, -1))[ii] > mjd-20) *
                        (np.unique(np.round(mjd, -1))[ii] < mjd+20))
        dnu[indices] = float(np.mean(dnu[indices]))
        dnu_mean.append(float(np.mean(dnu[indices])))
        freqMHz_mean.append(np.unique(freqMHz)[i])
        mjd_mean.append(np.unique(np.round(mjd, -1))[ii])
dnu_mean = np.asarray(dnu_mean)
freqMHz_mean = np.asarray(freqMHz_mean)
mjd_mean = np.asarray(mjd_mean)

mjd_ciel = [59780, 59800, 59830, 59880, 59910, 59950, 60010, 60090, 70000]
mjd_flor = [0,     59780, 59810, 59870, 59900, 59940, 59990, 60070, 60110]

mjd_old = mjd.copy()
freqMHz_old = freqMHz.copy()
dnu_old = dnu.copy()
dnuerr_old = dnuerr.copy()

alphaPs = []
alphaPerrs = []

ref_dnus = []
ref_dnuerrs = []

ref_freqs = []
ref_freqerrs = []

for i in range(0, num_obs):
    
    one_obs_sort = [(mjd_old < mjd_ciel[i]) * (mjd_old > mjd_flor[i])]
    mjd = mjd_old[one_obs_sort]
    freqMHz = freqMHz_old[one_obs_sort]
    dnu = dnu_old[one_obs_sort]
    dnuerr = dnuerr_old[one_obs_sort]
###############################################################################
    # A few simple setup steps
    label = "Observation_" + str(i)
    outdir = "/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/New/Spectral_Index/Bilby/" + str(label) + "/"
    bilby.utils.check_directory_exists_and_if_not_mkdir(outdir)
    
    df_lim = 0.0332 * 1.1  # This is the channel bandwidth X 1.1 and our floor limit
    
    xdata = freqMHz[dnu > df_lim]
    ydata = dnu[dnu > df_lim]
    sigma = dnuerr[dnu > df_lim]
    
    c_init = np.median(ydata[np.argwhere((xdata > 800) * (xdata < 840))])
    
    # First, we define our "signal model", in this case a simple linear function
    def model(xdata, alpha, ref_freq, ref_dnu, EFAC, EQUAD):
        return (xdata/ref_freq)**alpha * ref_dnu
    
    # Now lets instantiate a version of our GaussianLikelihood, giving it
    # the xdata, ydata and signal model
    likelihood = AlphaLikelihood2(xdata, ydata, model, sigma)
    
    # From hereon, the syntax is exactly equivalent to other bilby examples
    # We make a prior
    priors = dict()
    priors["alpha"] = bilby.core.prior.Uniform(0, 6, "alpha")
    priors["ref_dnu"] = bilby.core.prior.Uniform(np.min(ydata), np.max(ydata),
                                                 "ref_dnu")
    priors["ref_freq"] = bilby.core.prior.Uniform(np.min(xdata), np.max(xdata),
                                                  "ref_freq")
    priors["EFAC"] = bilby.core.prior.Uniform(-10, 10, "EFAC")
    priors["EQUAD"] = bilby.core.prior.Uniform(-10, 10, "EQUAD")
    
    # And run sampler
    result = bilby.run_sampler(
        likelihood=likelihood,
        priors=priors,
        sampler="dynesty",
        nlive=250,
        outdir=outdir,
        label=label,
    )
    
    # Finally plot a corner plot: all outputs are stored in outdir
    font = {'size': 16}
    matplotlib.rc('font', **font)
    result.plot_corner()
    plt.show()
    plt.close()
    
    NUM = np.argmax(result.posterior['log_likelihood'])
    alpha_result = result.posterior['alpha'][NUM]
    alpha_std = np.std(result.posterior['alpha'].values)
    ref_dnu_result = result.posterior['ref_dnu'][NUM]
    ref_dnu_std = np.std(result.posterior['ref_dnu'].values)
    ref_freq_result = result.posterior['ref_freq'][NUM]
    ref_freq_std = np.std(result.posterior['ref_freq'].values)
    EFAC_result = result.posterior['EFAC'][NUM]
    EFAC_std = np.std(result.posterior['EFAC'].values)
    EQUAD_result = result.posterior['EQUAD'][NUM]
    EQUAD_std = np.std(result.posterior['EQUAD'].values)
    
    alpha_std = np.sqrt((alpha_std * 10**EFAC_result)**2+(10**EQUAD_result)**2)
    
    xsort = np.argsort(xdata)
    
    Size = 80*np.pi  # Determines the size of the datapoints used
    font = {'size': 28}
    matplotlib.rc('font', **font)
    
    # fig = plt.figure(figsize=(20, 10))
    # fig.subplots_adjust(hspace=0.5, wspace=0.5)
    # ax = fig.add_subplot(1, 1, 1)
    # plt.hist(result.posterior['alpha'], bins=20, color='C0', alpha=0.9)
    # yl = plt.ylim()
    # plt.vlines(alpha_result, yl[0], yl[1], colors='C3', label=r"$\alpha^\prime$="+str(round(alpha_result, 3))+"$\pm$"+str(round(alpha_std, 3)))
    # plt.vlines(alpha_result - alpha_std, yl[0], yl[1], linestyles='dashed', colors='C3')
    # plt.vlines(alpha_result + alpha_std, yl[0], yl[1], linestyles='dashed', colors='C3')
    # ax.legend(fontsize='small')
    # ax.set_xlabel(r"Spectral Index, $\alpha^\prime$ (MHz)")
    # plt.ylim(yl)
    # fig.savefig("{}/{}_alpha_dist.png".format(outdir, label))
    # plt.show()
    # plt.close()
    
    # We quickly plot the data to check it looks sensible
    fig = plt.figure(figsize=(20, 10))
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(xdata, ydata, c='C0', label="data", alpha=0.2)
    ax.errorbar(xdata, ydata, yerr=sigma, fmt=' ', color='k', linewidth=1, capsize=1, alpha=0.2)
    xl = plt.xlim()
    xrange = np.linspace(xl[0], xl[1], 1000)
    yresults = model(xrange, alpha=alpha_result, ref_freq=ref_freq_result, ref_dnu=ref_dnu_result, EFAC=EFAC_result, EQUAD=EQUAD_result)
    yresults_min = model(xrange, alpha=alpha_result-alpha_std, ref_freq=ref_freq_result, ref_dnu=ref_dnu_result, EFAC=EFAC_result, EQUAD=EQUAD_result)
    yresults_max = model(xrange, alpha=alpha_result+alpha_std, ref_freq=ref_freq_result, ref_dnu=ref_dnu_result, EFAC=EFAC_result, EQUAD=EQUAD_result)
    ax.plot(xrange, yresults, "C3", label=r"$\alpha^\prime$="+str(round(alpha_result, 3))+"$\pm$"+str(round(alpha_std, 3)))
    plt.fill_between(xrange, yresults_min, yresults_max, color='C3', alpha=0.2)
    ax.set_xlabel(r"Observational Frequency, $\nu$ (MHz)")
    ax.set_ylabel(r"Scintillation Bandwidth, $\Delta\nu_d$ (MHz)")
    # ax.set_title("Frequency dependence and day of year")
    ax.legend(fontsize='small')
    plt.xlim(xl)
    fig.savefig("{}/{}_data_model.png".format(outdir, label))
    plt.show()
    plt.close()
    
    alpha_prime = alpha_result
    Beta = (2 * alpha_prime)/(alpha_prime - 2)
    Beta_err = np.sqrt(((-4)/((alpha_prime - 2)**2))**2*(alpha_std)**2)
    alpha = Beta - 2
    alpha_err = Beta_err
    
    model2 = (xdata/1000)**alpha_result * c_init
    chisqr = np.sum((ydata - model2)**2/(model2))
    Ndata = len(xdata)
    Nparam = 1
    Nfree = Ndata - Nparam
    red_chisqr = chisqr / Nfree
    
    print("========== We found the relationship between our data and observational frequency to be ==========")
    print("========== alpha_prime = "+str(round(alpha_result, 3))+" +/- "+str(round(alpha_std, 3))+" ==========")
    print("========== alpha       = "+str(round(alpha, 3))+" +/- "+str(round(alpha_err, 3))+" ==========")
    print("========== Beta        = "+str(round(Beta, 3))+" +/- "+str(round(Beta_err, 3))+" ==========")
    print("========== Chisqr      = "+str(round(chisqr, 3))+" ====================")
    print("========== RedChisqr   = "+str(round(red_chisqr, 3))+" ====================")
    
    alphaPs.append(alpha_result)
    alphaPerrs.append(alpha_std)
    
    ref_dnus.append(ref_dnu_result)
    ref_dnuerrs.append(ref_dnu_std)
    
    ref_freqs.append(ref_freq_result)
    ref_freqerrs.append(ref_freq_std)
###############################################################################
# UHF
mjd_UHF = np.loadtxt(datadir + '_mjd.txt', dtype='float')
freqMHz_UHF = np.loadtxt(datadir + '_freqMHz.txt', dtype='float')
dnu_UHF = np.loadtxt(datadir + '_dnu.txt', dtype='float')
dnuerr_UHF = np.loadtxt(datadir + '_dnuerr.txt', dtype='float')
# L-Band
mjd_L = np.loadtxt(datadir_Lband + '_mjd.txt', dtype='float')
freqMHz_L = np.loadtxt(datadir_Lband + '_freqMHz.txt', dtype='float')
dnu_L = np.loadtxt(datadir_Lband + '_dnu.txt', dtype='float')
dnuerr_L = np.loadtxt(datadir_Lband + '_dnuerr.txt', dtype='float')
# S-Band
mjd_S = np.loadtxt(datadir_Sband + '_mjd.txt', dtype='float')
freqMHz_S = np.loadtxt(datadir_Sband + '_freqMHz.txt', dtype='float')
dnu_S = np.loadtxt(datadir_Sband + '_dnu.txt', dtype='float')
dnuerr_S = np.loadtxt(datadir_Sband + '_dnuerr.txt', dtype='float')
# Combine
mjd = np.concatenate((mjd_UHF, mjd_L, mjd_S))
freqMHz = np.concatenate((freqMHz_UHF, freqMHz_L, freqMHz_S))
dnu = np.concatenate((dnu_UHF, dnu_L, dnu_S))
dnuerr = np.concatenate((dnuerr_UHF, dnuerr_L, dnuerr_S))
###############################################################################
# Here I will plot some things

alpha_prime = np.median(alphaPs)
alpha_primeerr = alpha_prime * np.sqrt(np.sum(alphaPerrs) / len(alphaPerrs))

ref_dnu = np.median(ref_dnus)
ref_dnuerr = np.sqrt(np.std(ref_dnus)**2 + np.median(ref_dnuerrs)**2)

ref_freq = np.median(ref_freqs)
ref_freqerr = np.sqrt(np.std(ref_freqs)**2 + np.median(ref_freqerrs)**2)

xdata = freqMHz_old
ydata = dnu_old
sigma = dnuerr_old

# We quickly plot the data to check it looks sensible
fig = plt.figure(figsize=(20, 10))
fig.subplots_adjust(hspace=0.5, wspace=0.5)
ax = fig.add_subplot(1, 1, 1)
cm = plt.cm.get_cmap('viridis')
z = mjd
ax.scatter(xdata, ydata, c=z, cmap=cm, edgecolors='k', s=Size, label="data", alpha=0.2)
ax.errorbar(xdata, ydata, yerr=sigma, fmt=' ', ecolor='k', linewidth=2, capsize=2, alpha=0.2)
# xl = plt.xlim()
xrange = np.linspace(np.min(xdata)*0.95, np.max(xdata)*1.05, 1000)
yresults = model(xrange, alpha=alpha_prime, ref_freq=ref_freq, ref_dnu=ref_dnu, EFAC=1, EQUAD=1)
ytheory = model(xrange, alpha=4.4, ref_freq=ref_freq, ref_dnu=ref_dnu, EFAC=1, EQUAD=1)
yresults_min = model(xrange, alpha=alpha_prime-alpha_primeerr, ref_freq=ref_freq, ref_dnu=ref_dnu, EFAC=1, EQUAD=1)
yresults_max = model(xrange, alpha=alpha_prime+alpha_primeerr, ref_freq=ref_freq, ref_dnu=ref_dnu, EFAC=1, EQUAD=1)
ax.plot(xrange, yresults, "C3", label=r"$\alpha^\prime$="+str(round(alpha_prime, 3))+"$\pm$"+str(round(alpha_primeerr, 3)))
ax.plot(xrange, ytheory, "k", label=r"$\alpha^\prime$=4.4 (Kolmogorov)")
plt.fill_between(xrange, yresults_min, yresults_max, color='C3', alpha=0.2)
ax.set_title(r"Total $\alpha^\prime$ Measurement")
ax.set_xlabel(r"Observational Frequency, $\nu$ (MHz)")
ax.set_ylabel(r"Scintillation Bandwidth, $\Delta\nu_d$ (MHz)")
# ax.set_title("Frequency dependence and day of year")
ax.legend(fontsize='small')
ax.loglog()
plt.xlim(np.min(xrange), np.max(xrange))
plt.savefig("/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/New/Spectral_Index/Bilby/Total_Alpha_Prime.png")
plt.show()
plt.close()

Beta = (2 * alpha_prime)/(alpha_prime - 2)
Beta_err = np.sqrt(((-4)/((alpha_prime - 2)**2))**2*(alpha_primeerr)**2)
alpha = Beta - 2
alpha_err = Beta_err

alpha_prime_min = alpha_prime - alpha_primeerr
Beta_min = (2 * alpha_prime_min)/(alpha_prime_min  - 2)
Beta_err_min = np.sqrt(((-4)/((alpha_prime_min  - 2)**2))**2*(alpha_primeerr)**2)
alpha_min = Beta_min  - 2
alpha_err_min = Beta_err_min 

alpha_prime_max = alpha_prime + alpha_primeerr
Beta_max = (2 * alpha_prime_max)/(alpha_prime_max  - 2)
Beta_err_max = np.sqrt(((-4)/((alpha_prime_max  - 2)**2))**2*(alpha_primeerr)**2)
alpha_max = Beta_max  - 2
alpha_err_max = Beta_err_max 

alpha_range = abs(alpha_max - alpha_min)
###############################################################################
