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
par_dir = '/Users/jacobaskew/Desktop/Swinburne_Daniel/ParFiles/'
psrname = 'J0737-3039A'
wd0 = "/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/New/"
outdir = wd0 + "Modelling"
pars = read_par(str(par_dir) + str(psrname) + '.par')
params = pars_to_params(pars)

viss = np.loadtxt(datadir + '_viss.txt', dtype='float')
visserr = np.loadtxt(datadir + '_visserr.txt', dtype='float')
mjd = np.loadtxt(datadir + '_mjd.txt', dtype='float')
freqMHz = np.loadtxt(datadir + '_freqMHz.txt', dtype='float')
freqGHz = freqMHz / 1e3
dnu = np.loadtxt(datadir + '_dnu.txt', dtype='float')
dnuerr = np.loadtxt(datadir + '_dnuerr.txt', dtype='float')
tau = np.loadtxt(datadir + '_tau.txt', dtype='float')
tauerr = np.loadtxt(datadir + '_tauerr.txt', dtype='float')
phase = np.loadtxt(datadir + '_phase.txt', dtype='float')
U = np.loadtxt(datadir + '_U.txt', dtype='float')
ve_ra = np.loadtxt(datadir + '_ve_ra.txt', dtype='float')
ve_dec = np.loadtxt(datadir + '_ve_dec.txt', dtype='float')

# Loading in 11 extra bits around each mjd for the 10min window effect ...
if os.path.exists(datadir + '10minMJD.txt'):
    mjd_range = np.loadtxt(datadir + '10minMJD.txt', dtype='float')
    U_range = np.loadtxt(datadir + '10minU.txt', dtype='float')
    ve_ra_range = np.loadtxt(datadir + '10minVE_RA.txt', dtype='float')
    ve_dec_range = np.loadtxt(datadir + '10minVE_DEC.txt', dtype='float')
    phase_range = np.loadtxt(datadir + '10minPhase.txt', dtype='float')
else:
    mjd_10min = []
    step = 5 / 1440
    for i in range(0, len(mjd)):
        mjd_10min.append(np.linspace(mjd[i]-step, mjd[i]+step, 11))
    mjd_10min = np.asarray(mjd_10min).flatten()
    np.savetxt(datadir+'10minMJD.txt', mjd_10min, delimiter=',', fmt='%s')
    mjd_range = mjd_10min
    vearth_ra_10min, vearth_dec_10min = get_earth_velocity(mjd_10min, pars['RAJ'], pars['DECJ'])
    U_10min = get_true_anomaly(mjd_10min, pars)
    vearth_ra_10min = vearth_ra_10min.squeeze()
    vearth_dec_10min = vearth_dec_10min.squeeze()
    om_10min = pars['OM'] + pars['OMDOT']*(mjd_10min - pars['T0'])/365.2425
    phase_10min = U_10min*180/np.pi + om_10min
    phase_10min = phase_10min % 360
    U_range = U_10min
    ve_ra_range = vearth_ra_10min
    ve_dec_range = vearth_dec_10min
    phase_range = phase_10min
    np.savetxt(datadir+'10minU.txt', U_range, delimiter=',', fmt='%s')
    np.savetxt(datadir+'10minVE_RA.txt', ve_ra_range, delimiter=',', fmt='%s')
    np.savetxt(datadir+'10minVE_DEC.txt', ve_dec_range, delimiter=',', fmt='%s')
    np.savetxt(datadir+'10minPhase.txt', phase_range, delimiter=',', fmt='%s')

# Loading in model data across min to max mjd, 10000 steps ...
if os.path.exists(datadir + 'Model_mjdData.txt'):
    Model_mjd = np.loadtxt(datadir + 'Model_mjdData.txt', dtype='float')
    Model_phase = np.loadtxt(datadir + 'Model_phaseData.txt', dtype='float')
    Model_U = np.loadtxt(datadir + 'Model_UData.txt', dtype='float')
    Model_vearth_ra = np.loadtxt(datadir + 'Model_vearth_raData.txt', dtype='float')
    Model_vearth_dec = np.loadtxt(datadir + 'Model_vearth_decData.txt', dtype='float')
else:
    # NUM_MJD = round(11*10*1440*(np.max(mjd)-np.min(mjd)), -3)  # Expensive time to calc using this
    NUM_MJD = 8000
    Model_mjd = np.linspace(np.min(mjd), np.max(mjd), NUM_MJD)
    np.savetxt(datadir+'Model_mjdData.txt', Model_mjd, delimiter=',', fmt='%s')
    Model_vearth_ra, Model_vearth_dec = get_earth_velocity(Model_mjd, pars['RAJ'], pars['DECJ'])
    Model_U = get_true_anomaly(Model_mjd, pars)
    Model_vearth_ra = Model_vearth_ra.squeeze()
    Model_vearth_dec = Model_vearth_dec.squeeze()
    om_model = pars['OM'] + pars['OMDOT']*(Model_mjd - pars['T0'])/365.2425
    Model_phase = Model_U*180/np.pi + om_model
    Model_phase = Model_phase % 360
    np.savetxt(datadir+'Model_UData.txt', Model_U, delimiter=',', fmt='%s')
    np.savetxt(datadir+'Model_vearth_raData.txt', Model_vearth_ra, delimiter=',', fmt='%s')
    np.savetxt(datadir+'Model_vearth_decData.txt', Model_vearth_dec, delimiter=',', fmt='%s')
    np.savetxt(datadir+'Model_phaseData.txt', Model_phase, delimiter=',', fmt='%s')

dnu_old = dnu.copy()
dnu_mean = []
freqMHz_mean = []
mjd_mean = []
# I want to take the average of dnu at each frequency at each observation
for i in range(0, len(np.unique(freqMHz))):
    for ii in range(0, 7):
        indices = \
            np.argwhere((np.unique(freqMHz)[i] == freqMHz) *
                        (np.unique(np.round(mjd, -1))[ii] > mjd-20) *
                        (np.unique(np.round(mjd, -1))[ii] < mjd+20))
        dnu[indices] = float(np.mean(dnu[indices]))
        dnu_mean.append(float(np.mean(dnu[indices])))
        freqMHz_mean.append(np.unique(freqMHz)[i])
        mjd_mean.append(np.unique(np.round(mjd, -1))[ii])
cm = plt.cm.get_cmap('viridis')
z = freqMHz
plt.scatter(dnu_old, dnu, c=z, cmap=cm)
plt.xlabel(r"Original $\Delta\nu_d$ (MHz)")
plt.ylabel(r"Mean $\Delta\nu_d$ (MHz)")
plt.show()
plt.close()



num_obs = 12
phase_ciel = [30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 361]
phase_flor = [-1, 30, 60,  90, 120, 150, 180, 210, 240, 270, 300, 330]


viss_old = viss.copy()
visserr_old = visserr.copy()
mjd_old = mjd.copy()
freqMHz_old = freqMHz.copy()
freqGHz_old = freqMHz_old / 1e3
dnu_old = dnu.copy()
dnuerr_old = dnuerr.copy()
tau_old = tau.copy()
tauerr_old = tauerr.copy()
phase_old = phase.copy()
U_old = U.copy()
ve_ra_old = ve_ra.copy()
ve_dec_old = ve_dec.copy()
mjd_range_old = mjd_range.copy()
U_range_old = U_range.copy()
ve_ra_range_old = ve_ra_range.copy()
ve_dec_range_old = ve_dec_range.copy()
phase_range_old = phase_range.copy()

alphaPs = []
alphaPerrs = []

ref_dnus = []
ref_dnuerrs = []

ref_freqs = []
ref_freqerrs = []

for i in range(0, num_obs):
    
    one_obs_sort = [(phase_old < phase_ciel[i]) * (phase_old > phase_flor[i])]
    one_obs_sort2 = [(phase_range_old < np.max(phase_old[(phase_old < phase_ciel[i]) * (phase_old > phase_flor[i])])) *
                      ((phase_range_old > np.min(phase_old[(phase_old < phase_ciel[i]) * (phase_old > phase_flor[i])])))]
    viss = viss_old[one_obs_sort]
    visserr = visserr_old[one_obs_sort]
    mjd = mjd_old[one_obs_sort]
    freqMHz = freqMHz_old[one_obs_sort]
    freqGHz = freqGHz_old[one_obs_sort]
    dnu = dnu_old[one_obs_sort]
    dnuerr = dnuerr_old[one_obs_sort]
    tau = tau_old[one_obs_sort]
    tauerr = tauerr_old[one_obs_sort]
    phase = phase_old[one_obs_sort]
    U = U_old[one_obs_sort]
    ve_ra = ve_ra_old[one_obs_sort]
    ve_dec = ve_dec_old[one_obs_sort]
    mjd_range = mjd_range_old[one_obs_sort2]
    U_range = U_range_old[one_obs_sort2]
    ve_ra_range = ve_ra_range_old[one_obs_sort2]
    ve_dec_range = ve_dec_range_old[one_obs_sort2]
    phase_range = phase_range_old[one_obs_sort2]
###############################################################################
    # A few simple setup steps
    label = "Phase_" + str(i*30)
    outdir = "/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/New/Spectral_Index/Bilby/" + str(label) + "/"
    bilby.utils.check_directory_exists_and_if_not_mkdir(outdir)
    
    xdata = freqMHz[freqMHz > 800]
    ydata = dnu[freqMHz > 800]
    sigma = dnuerr[freqMHz > 800]
    
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
ax.scatter(xdata, ydata, c='C0', label="data", alpha=0.2)
ax.errorbar(xdata, ydata, yerr=sigma, fmt=' ', color='k', linewidth=1, capsize=1, alpha=0.2)
xl = plt.xlim()
xrange = np.linspace(xl[0], xl[1], 1000)
yresults = model(xrange, alpha=alpha_prime, ref_freq=ref_freq, ref_dnu=ref_dnu, EFAC=1, EQUAD=1)
yresults_min = model(xrange, alpha=alpha_prime-alpha_primeerr, ref_freq=ref_freq, ref_dnu=ref_dnu, EFAC=1, EQUAD=1)
yresults_max = model(xrange, alpha=alpha_prime+alpha_primeerr, ref_freq=ref_freq, ref_dnu=ref_dnu, EFAC=1, EQUAD=1)
ax.plot(xrange, yresults, "C3", label=r"$\alpha^\prime$="+str(round(alpha_prime, 3))+"$\pm$"+str(round(alpha_primeerr, 3)))
plt.fill_between(xrange, yresults_min, yresults_max, color='C3', alpha=0.2)
ax.set_title(r"Total $\alpha^\prime$ Measurement")
ax.set_xlabel(r"Observational Frequency, $\nu$ (MHz)")
ax.set_ylabel(r"Scintillation Bandwidth, $\Delta\nu_d$ (MHz)")
# ax.set_title("Frequency dependence and day of year")
ax.legend(fontsize='small')
plt.xlim(xl)
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


alpha_range = alpha_max - alpha_min







###############################################################################
