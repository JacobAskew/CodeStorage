#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 12:04:23 2022

@author: jacobaskew
"""

# I want to write a python script that can process my new data much like old

##############################################################################
# Common #
from scintools.dynspec import Dynspec
from scintools.scint_utils import write_results, read_results, read_par, \
        float_array_from_dict, get_ssb_delay, get_earth_velocity, \
        get_true_anomaly
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import glob
from copy import deepcopy as cp
from lmfit import Parameters, Minimizer
import os
from scipy.optimize import curve_fit
from scintools.scint_sim import Simulation
###############################################################################
# Fitting methods from Geray


def least_squares_Yerrors(X_vals, Y_vals, Y_error):
    '''linear fit for points with yerrors and returning error of fitline
    input: all points as 1D arrays'''

    Y = np.matrix(Y_vals)
    C = np.zeros((X_vals.size, X_vals.size))
    A = np.matrix(np.transpose(np.array([np.ones(X_vals.size), X_vals])))

    for i in range(0, X_vals.size):
        C[i, i] = Y_error[i]**2

    Cinv = np.linalg.inv(C)
    X = np.linalg.inv(A.T*Cinv*A)*(A.T*Cinv*Y.T)

    # variance
    Var = np.sqrt(np.linalg.inv(A.T*Cinv*A))
    return X.item(1), X.item(0), Var.item(1, 1), Var.item(0, 0)
    # gradient, error of gradient, intercept, error of intercept


def least_squares(x, y):
    '''when you dont have errrors but also dont need an error for the
    fitted line input: all points as 1D arrays'''
    # m = ( <xy> - <x><y> ) / ( <x^2> - <x>^2 )
    gradient = ((np.nanmean(x*y) - np.nanmean(x) * np.nanmean(y))
                (np.nanmean(x**2) - np.nanmean(x)**2))
    # c = <y> - m <x>
    intercept = np.nanmean(y) - gradient * np.nanmean(x)
    return gradient, intercept


def weighted_least_squares(x, y, dy):
    '''or if you need weights
    input: all points as 1D arrays'''
    w = 1. / dy**2  # weighting by inverse variance
    w /= np.sum(w)  # enforce normalisation sum( weights ) = 1

    gradient = ((np.sum(w*x*y) - np.sum(w*x) * np.sum(w*y))
                (np.sum(w*x**2) - np.sum(w*x)**2))
    intercept = np.sum(w*y) - gradient * np.sum(w*x)

    return gradient, intercept


###############################################################################


def powerlaw_fitter(xdata, ydata, weights, reffreq, amp_init=1, amp_min=0,
                    amp_max=np.inf, alpha_init=4, alpha_min=0,
                    alpha_max=np.inf, reffreq_min=0, reffreq_max=np.inf,
                    steps=10000, burn=0.2, return_amp=False):
    # reffreq_max = np.max(xdata)*0.95
    parameters = Parameters()
    parameters.add('amp', value=amp_init, vary=True, min=amp_min, max=amp_max)
    parameters.add('alpha', value=alpha_init, vary=True, min=alpha_min,
                   max=alpha_max)
    # parameters.add('reffreq', value=reffreq, vary=True, min=reffreq_min,
    #                max=reffreq_max)
    func = Minimizer(powerlaw, parameters, fcn_args=(xdata, ydata, weights,
                     reffreq, amp_init), nan_policy='raise')
    mcmc_results = func.emcee(steps=steps, burn=int(burn * steps))
    results = mcmc_results

    Slope = results.params['alpha'].value
    Slopeerr = results.params['alpha'].stderr
    Amp = results.params['amp'].value
    Amperr = results.params['amp'].stderr
    # Reffreq = results.params['reffreq'].value
    # Reffreqerr = results.params['reffreq'].stderr
    if return_amp:
        return Slope, Slopeerr, Amp, Amperr
    else:
        return Slope, Slopeerr


##############################################################################


def powerlaw(params, xdata, ydata, weights, reffreq, amp):

    if weights is None:
        weights = np.ones(np.shape(xdata))

    if ydata is None:
        ydata = np.zeros(np.shape(xdata))

    parvals = params.valuesdict()
    # if amp is None:
    #     parvals.add('amp', value=1, vary=True)
    #     amp = parvals['amp']
    # else:
    #     parvals.add('amp', value=amp, vary=True)
    amp = parvals['amp']
    # parvals.add('alpha', value=4, vary=True)
    alpha = parvals['alpha']
    # reffreq = parvals['reffreq']

    func = amp*(xdata/reffreq)**(alpha)

    return (ydata - func)


def func1(xdata, c, a):
    return a*(xdata/1000)**c


def func2(xdata, c, a):
    return a*(xdata/1300)**c


def func3(xdata, c, a):
    return a*(xdata/1600)**c


##############################################################################


def remove_eclipse(start_mjd, tobs, dyn, fluxes):
    end_mjd = start_mjd + tobs/86400
    Eclipse_mjd = np.genfromtxt(eclipsefile, delimiter=',',
                                encoding=None, dtype=float)
    Eclipse_events = np.array(np.where((Eclipse_mjd > start_mjd) *
                              (Eclipse_mjd < end_mjd)))
    if Eclipse_events.size == 0:
        median_flux_list = []
        for i in range(0, np.shape(fluxes)[1]):
            median_flux_list.append(np.median(fluxes[:, i]))
        median_fluxes = np.asarray(median_flux_list)
        Eclipse_index = int(np.argmin(median_fluxes))
        if Eclipse_index is not None:
            fluxes[:, Eclipse_index-3:Eclipse_index+3] = 0
            if linear:
                dyn.refill(method='linear')
            else:
                dyn.refill()
            print("Eclispe in dynspec")

        else:
            print("No Eclispe in dynspec")
    else:
        Eclipse_events_mjd = Eclipse_mjd[Eclipse_events]
        mjds = start_mjd + dyn.times/86400
        Eclipse_index = np.argmin(abs(mjds - Eclipse_events_mjd))
        if Eclipse_index is not None:
            print("Eclispe in dynspec")
            fluxes[:, Eclipse_index-3:Eclipse_index+3] = 0
            if linear:
                dyn.refill(method='linear')
            elif median:
                dyn.refill(method='median')
            else:
                dyn.refill()
        else:
            print("No Eclispe in dynspec")
    return Eclipse_index


##############################################################################
psrname = 'J0737-3039A'
pulsar = '0737-3039A'

wd0 = '/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/'
eclipsefile = wd0+'Datafiles/Eclipse_mjd.txt'
wd = wd0+'New/'
datadir = wd + 'Dynspec/'
outdir = wd + 'DataFiles/'
par_dir = '/Users/jacobaskew/Desktop/Swinburne_Daniel/ParFiles/'
dynspecs = sorted(glob.glob(datadir + '/*.dynspec'))
observations = []
for i in range(0, len(dynspecs)):
    observations.append(dynspecs[i].split(datadir)[1].split('-')[0]+'-' +
                        dynspecs[i].split(datadir)[1].split('-')[1]+'-' +
                        dynspecs[i].split(datadir)[1].split('-')[2])
observations = np.unique(np.asarray(observations))
for i in range(0, len(observations)):
    observation_date = observations[i]+'/'
    observation_date2 = observations[i]
    spectradir = wd + 'SpectraPlots/' + observation_date
    spectrabindir = wd + 'SpectraPlotsBin/' + observation_date
    ACFdir = wd + 'ACFPlots/' + observation_date
    ACFbindir = wd + 'ACFPlotsBin/' + observation_date
    plotdir = wd + 'Plots/' + observation_date
    dynspecs = sorted(glob.glob(datadir + str(observation_date.split('/')[0])
                                + '*.XPp.dynspec'))
    dynspecfile = \
        outdir+'DynspecPlotFiles/'+observation_date2+'_CompleteDynspec.dynspec'
    # Settings #
    time_bin = 30
    freq_bin = 30
    #
    measure = False
    model = False
    model2 = False
    compare = True
    load_data = True
    plotting = True
    #
    zap = True
    linear = False
    var = False
    median = True
    #
    filedir = str(outdir)+str(observation_date)
    filedir2 = str(outdir)
    try:
        os.mkdir(filedir)
    except OSError as error:
        print(error)
    outfile = str(filedir)+str(psrname)+'_'+str(observation_date2)+'_freq' + \
        str(freq_bin)+'_time'+str(time_bin)+'_ScintillationResults_UHF.txt'
    outfile_total = str(filedir2)+str(psrname)+'_freq'+str(freq_bin)+'_time' + \
        str(time_bin)+'_ScintillationResults_UHF_Total.txt'
    if os.path.exists(outfile) and measure:
        os.remove(outfile)
    if var:
        freq_bin_string = str(freq_bin)+'var'
    else:
        freq_bin_string = str(freq_bin)

    try:
        os.mkdir(spectradir)
    except OSError as error:
        print(error)
    try:
        os.mkdir(spectrabindir)
    except OSError as error:
        print(error)
    try:
        os.mkdir(ACFdir)
    except OSError as error:
        print(error)
    try:
        os.mkdir(ACFbindir)
    except OSError as error:
        print(error)
    try:
        os.mkdir(plotdir)
    except OSError as error:
        print(error)
    try:
        os.mkdir(spectradir+'time'+str(time_bin)+'freq'+str(freq_bin_string) +
                 '/')
    except OSError as error:
        print(error)
    try:
        os.mkdir(spectrabindir+'time'+str(time_bin)+'freq' +
                 str(freq_bin_string)+'/')
    except OSError as error:
        print(error)
    try:
        os.mkdir(ACFdir+'time'+str(time_bin)+'freq'+str(freq_bin_string) +
                 '/')
    except OSError as error:
        print(error)
    try:
        os.mkdir(ACFbindir+'time'+str(time_bin)+'freq' +
                 str(freq_bin_string)+'/')
    except OSError as error:
        print(error)
    try:
        os.mkdir(plotdir+'time'+str(time_bin)+'freq'+str(freq_bin_string)+'/')
    except OSError as error:
        print(error)

    plotdir = plotdir+'time'+str(time_bin)+'freq'+str(freq_bin_string)+'/'
    spectradir = spectradir+'time'+str(time_bin)+'freq' + \
        str(freq_bin_string)+'/'
    spectrabindir = spectrabindir+'time'+str(time_bin)+'freq' + \
        str(freq_bin_string)+'/'
    ACFdir = ACFdir+'time'+str(time_bin)+'freq' + \
        str(freq_bin_string)+'/'
    ACFbindir = ACFbindir+'time'+str(time_bin)+'freq' + \
        str(freq_bin_string)+'/'
if load_data:
    results_dir = outdir
    params = read_results(outfile_total)

    pars = read_par(str(par_dir) + str(psrname) + '.par')

    # Read in arrays
    mjd = float_array_from_dict(params, 'mjd')  # MJD for observation start
    df = float_array_from_dict(params, 'df')  # channel bandwidth
    dnu = float_array_from_dict(params, 'dnu')  # scint bandwidth
    dnu_est = float_array_from_dict(params, 'dnu_est')  # estimated bandwidth
    dnuerr = float_array_from_dict(params, 'dnuerr')
    tau = float_array_from_dict(params, 'tau')
    tauerr = float_array_from_dict(params, 'tauerr')
    freq = float_array_from_dict(params, 'freq')
    bw = float_array_from_dict(params, 'bw')
    name = np.asarray(params['name'])
    # scintle_num = float_array_from_dict(params, 'scintle_num')
    tobs = float_array_from_dict(params, 'tobs')  # tobs in second
    rcvrs = np.array([rcvr[0] for rcvr in params['name']])

    # Sort by MJD
    sort_ind = np.argsort(mjd)

    df = np.array(df[sort_ind]).squeeze()
    dnu = np.array(dnu[sort_ind]).squeeze()
    dnu_est = np.array(dnu_est[sort_ind]).squeeze()
    dnuerr = np.array(dnuerr[sort_ind]).squeeze()
    tau = np.array(tau[sort_ind]).squeeze()
    tauerr = np.array(tauerr[sort_ind]).squeeze()
    mjd = np.array(mjd[sort_ind]).squeeze()
    rcvrs = np.array(rcvrs[sort_ind]).squeeze()
    freq = np.array(freq[sort_ind]).squeeze()
    tobs = np.array(tobs[sort_ind]).squeeze()
    name = np.array(name[sort_ind]).squeeze()
    # scintle_num = np.array(scintle_num[sort_ind]).squeeze()
    bw = np.array(bw[sort_ind]).squeeze()

    # Used to filter the data
    indicies = np.argwhere((tauerr < 10*tau) * (dnuerr < 10*dnu))

    df = df[indicies].squeeze()
    dnu = dnu[indicies].squeeze()
    dnu_est = dnu_est[indicies].squeeze()
    dnuerr = dnuerr[indicies].squeeze()
    tau = tau[indicies].squeeze()
    tauerr = tauerr[indicies].squeeze()
    mjd = mjd[indicies].squeeze()
    rcvrs = rcvrs[indicies].squeeze()
    freq = freq[indicies].squeeze()
    tobs = tobs[indicies].squeeze()
    name = name[indicies].squeeze()
    # scintle_num = scintle_num[indicies].squeeze()
    bw = bw[indicies].squeeze()

    mjd_annual = mjd % 365.2425
    print('Getting SSB delays')
    ssb_delays = get_ssb_delay(mjd, pars['RAJ'], pars['DECJ'])
    mjd += np.divide(ssb_delays, 86400)  # add ssb delay

    dnu_std = np.std(dnu)

    """
    Model Viss
    """
    print('Getting Earth velocity')
    vearth_ra, vearth_dec = get_earth_velocity(mjd, pars['RAJ'], pars['DECJ'])
    print('Getting true anomaly')
    pars['PBDOT'] *= 10**-12  # add in factor dropped by tempo
    U = get_true_anomaly(mjd, pars)

    true_anomaly = U.squeeze()
    vearth_ra = vearth_ra.squeeze()
    vearth_dec = vearth_dec.squeeze()

    om = pars['OM'] + pars['OMDOT']*(mjd - pars['T0'])/365.2425
    # compute orbital phase
    phase = U*180/np.pi + om
    phase[phase > 360] = phase[phase > 360] - 360
###############################################################################
if plotting:

    Font = 30
    Size = 80*np.pi  # Determines the size of the datapoints used
    font = {'size': 28}
    matplotlib.rc('font', **font)
###############################################################################

    # MJD #

    fig = plt.figure(figsize=(20, 10))
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    cm = plt.cm.get_cmap('viridis')
    z = freq
    sc = plt.scatter(mjd, dnu, c=z, cmap=cm, s=Size, alpha=0.6)
    plt.colorbar(sc)
    plt.errorbar(mjd, dnu, yerr=dnuerr, fmt=' ', ecolor='k',
                 elinewidth=2, capsize=3, alpha=0.55)
    xl = plt.xlim()
    plt.plot(xl, (df[0], df[0]), color='C2')
    plt.xlabel('MJD')
    plt.ylabel('Scintillation Bandwidth (MHz)')
    plt.title(psrname + r', $\Delta\nu$')
    # plt.grid(True, which="both", ls="-", color='0.65')
    plt.xlim(xl)
    plt.ylim(0, 0.2)
    # plt.savefig(str(Dnudir) + "Dnu_Orbital_Freq.png")
    plt.show()
    plt.close()

    # ANNUAL #

    fig = plt.figure(figsize=(20, 10))
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    cm = plt.cm.get_cmap('viridis')
    z = freq
    sc = plt.scatter(mjd_annual, dnu, c=z, cmap=cm, s=Size, alpha=0.6)
    plt.colorbar(sc)
    plt.errorbar(mjd_annual, dnu, yerr=dnuerr, fmt=' ', ecolor='k',
                 elinewidth=2, capsize=3, alpha=0.55)
    xl = plt.xlim()
    plt.plot(xl, (df[0], df[0]), color='C2')
    plt.xlabel('Annual Phase (Days)')
    plt.ylabel('Scintillation Bandwidth (MHz)')
    plt.title(psrname + r', $\Delta\nu$')
    # plt.grid(True, which="both", ls="-", color='0.65')
    plt.xlim(xl)
    plt.ylim(0, 0.2)
    # plt.savefig(str(Dnudir) + "Dnu_Orbital_Freq.png")
    plt.show()
    plt.close()

    # PHASE #

    fig = plt.figure(figsize=(20, 10))
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    cm = plt.cm.get_cmap('viridis')
    z = freq
    sc = plt.scatter(phase, dnu, c=z, cmap=cm, s=Size, alpha=0.6)
    plt.colorbar(sc)
    plt.errorbar(phase, dnu, yerr=dnuerr, fmt=' ', ecolor='k',
                 elinewidth=2, capsize=3, alpha=0.55)
    xl = plt.xlim()
    plt.plot(xl, (df[0], df[0]), color='C2')
    plt.xlabel('Orbital Phase (degrees)')
    plt.ylabel('Scintillation Bandwidth (MHz)')
    plt.title(psrname + r', $\Delta\nu$')
    # plt.grid(True, which="both", ls="-", color='0.65')
    plt.xlim(xl)
    plt.ylim(0, 0.2)
    # plt.savefig(str(Dnudir) + "Dnu_Orbital_Freq.png")
    plt.show()
    plt.close()

    # PHASE and observation day #
    name_num = []
    for i in range(0, len(name)):
        for ii in range(0, len(np.unique(name))):
            if name[i] == np.unique(name)[ii]:
                name_num.append(ii)

    fig = plt.figure(figsize=(20, 10))
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    cm = plt.cm.get_cmap('viridis')
    z = name_num
    sc = plt.scatter(phase, dnu, c=z, cmap=cm, s=Size, alpha=0.6)
    plt.colorbar(sc)
    plt.errorbar(phase, dnu, yerr=dnuerr, fmt=' ', ecolor='k',
                 elinewidth=2, capsize=3, alpha=0.55)
    xl = plt.xlim()
    plt.plot(xl, (df[0], df[0]), color='C2')
    plt.xlabel('Orbital Phase (degrees)')
    plt.ylabel('Scintillation Bandwidth (MHz)')
    plt.title('Orbital Phase and "observation run"')
    # plt.grid(True, which="both", ls="-", color='0.65')
    plt.xlim(xl)
    plt.ylim(0, 0.2)
    # plt.savefig(str(Dnudir) + "Dnu_Orbital_Freq.png")
    plt.show()
    plt.close()
    # Powerlaw #
    fig = plt.figure(figsize=(20, 10))
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    ax = fig.add_subplot(1, 1, 1)
    #
    plt.scatter(freq, dnu, s=Size, alpha=0.6, c='C0')
    plt.errorbar(freq, dnu, yerr=dnuerr, fmt=' ',
                 ecolor='k', elinewidth=2, capsize=3, alpha=0.4)
    #
    xl = plt.xlim()
    xdata = np.linspace(xl[0], xl[1], 1000)
    #
    popt, pcov = curve_fit(func1, freq, dnu)
    perr = np.sqrt(np.diag(pcov))
    plt.plot(xdata, func1(xdata, *popt),
             'C1', label=r'$f_c=1000$, $\alpha$='+str(round(popt[0], 2)) +
             r'$\pm$'+str(round(perr[0], 2)))
    plt.fill_between(xdata.flatten(),
                     func1(xdata, *[popt[0]+perr[0], popt[1]]).flatten(),
                     func1(xdata, *[popt[0]-perr[0], popt[1]]).flatten(),
                     alpha=0.5, color='C1')
    #
    theory_dnu = popt[1]*(xdata/1000)**(4)
    plt.plot(xdata, theory_dnu, c='k', linewidth=2)
    plt.plot(xl, (df[0], df[0]), color='C2')
    #
    plt.xlabel('Orbital Phase (degrees)')
    plt.ylabel('Scintillation Bandwidth (MHz)')
    plt.title('initial powerlaw fit')
    plt.show()
    plt.close()

if model:
    Font = 30
    Size = 80*np.pi  # Determines the size of the datapoints used
    font = {'size': 28}
    matplotlib.rc('font', **font)

    Slopes = []
    Slopeerrs = []
    Amps = []
    Amperrs = []
    for i in range(0, len(np.unique(mjd))):
        freq_new = freq[mjd == np.unique(mjd)[i]]
        dnu_new = dnu[mjd == np.unique(mjd)[i]]
        dnuerr_new = dnuerr[mjd == np.unique(mjd)[i]]

        fig = plt.figure(figsize=(20, 10))
        fig.subplots_adjust(hspace=0.5, wspace=0.5)
        ax = fig.add_subplot(1, 1, 1)
        #
        plt.scatter(freq_new, dnu_new, s=Size, alpha=0.6, c='C0')
        plt.errorbar(freq_new, dnu_new, yerr=dnuerr_new, fmt=' ',
                     ecolor='k', elinewidth=2, capsize=3, alpha=0.4)
        #
        xl = plt.xlim()
        xdata = np.linspace(xl[0], xl[1], 1000)
        #
        popt, pcov = curve_fit(func1, freq_new, dnu_new)
        perr = np.sqrt(np.diag(pcov))
        plt.plot(xdata, func1(xdata, *popt),
                 'C1', label=r'$f_c=1000$, $\alpha$='+str(round(popt[0], 2)) +
                 r'$\pm$'+str(round(perr[0], 2)))
        plt.fill_between(xdata.flatten(),
                         func1(xdata, *[popt[0]+perr[0], popt[1]]).flatten(),
                         func1(xdata, *[popt[0]-perr[0], popt[1]]).flatten(),
                         alpha=0.5, color='C1')
        #
        theory_dnu = popt[1]*(xdata/1000)**(4)
        plt.plot(xdata, theory_dnu, c='k', linewidth=2)
        plt.plot(xl, (df[0], df[0]), color='C2')
        #
        ax.legend()
        plt.xlabel('Orbital Phase (degrees)')
        plt.ylabel('Scintillation Bandwidth (MHz)')
        plt.title('powerlaw best fit')
        plt.show()
        plt.close()

        Slopes.append(popt[0])
        Slopeerrs.append(perr[0])
        Amps.append(popt[1])
        Amperrs.append(perr[1])

    Slopes = np.asarray(Slopes)
    Slopeerrs = np.asarray(Slopeerrs)
    Amps = np.asarray(Amps)
    Amperrs = np.asarray(Amperrs)

    Average_Slope = np.average(Slopes, weights=Slopeerrs)
    Mean_Slopeerr = np.mean(Slopeerrs)
    Average_Amp = np.average(Amps, weights=Amperrs)
    Mean_Amperr = np.mean(Amperrs)

    fig = plt.figure(figsize=(20, 10))
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    ax = fig.add_subplot(1, 1, 1)
    #
    plt.scatter(freq, dnu, s=Size, alpha=0.6, c='C0')
    plt.errorbar(freq, dnu, yerr=dnuerr, fmt=' ',
                 ecolor='k', elinewidth=2, capsize=3, alpha=0.4)
    #
    xl = plt.xlim()
    xdata = np.linspace(xl[0], xl[1], 1000)
    #
    plt.plot(xdata, func1(xdata, *[Average_Slope, Average_Amp]),
             'C1', label=r'$f_c=1000$, $\alpha$='+str(round(Average_Slope,
                                                            2)) +
             r'$\pm$'+str(round(Mean_Slopeerr, 2)))
    plt.fill_between(xdata.flatten(),
                     func1(xdata, *[Average_Slope+Mean_Slopeerr,
                                    Average_Amp]).flatten(),
                     func1(xdata, *[Average_Slope-Mean_Slopeerr,
                                    Average_Amp]).flatten(),
                     alpha=0.5, color='C1')
    #
    theory_dnu = popt[1]*(xdata/1000)**(4)
    plt.plot(xdata, theory_dnu, c='k', linewidth=2)
    plt.plot(xl, (df[0], df[0]), color='C2')
    #
    ax.legend()
    plt.xlabel('Orbital Phase (degrees)')
    plt.ylabel('Scintillation Bandwidth (MHz)')
    plt.title('average power law fit')
    plt.xlim(xl)
    plt.show()
    plt.close()

    # # FREQ #

    # # Take the median dnu for each frequency. (DONE)
    # # Fit a power law to those median dnu values.
    # # Plot the expected values

    # unique_freqs = np.unique(freq[np.argwhere(freq > 800)])
    # median_dnu_freqs = []
    # median_dnuerr_freqs = []
    # for i in range(0, len(unique_freqs)):
    #     median_dnu_freqs.append(np.median(dnu[np.argwhere(freq ==
    #                                                       unique_freqs[i])]))
    #     median_dnuerr_freqs.append(np.median(dnuerr[
    #         np.argwhere(freq == unique_freqs[i])]))
    # median_dnu_freqs = np.asarray(median_dnu_freqs)
    # median_dnuerr_freqs = np.asarray(median_dnuerr_freqs)
    # median_dnu_freq1000 = np.median(dnu[np.argwhere((freq < 1020) * (freq >
    #                                                                  980))])

    # Slope, Slopeerr = powerlaw_fitter(unique_freqs, median_dnu_freqs,
    #                                   median_dnuerr_freqs, reffreq=1000,
    #                                   amp_init=0.07)

    # freqs = np.arange(0, 2000)
    # powerlaw_dnu = median_dnu_freq1000*(freqs/1000)**(Slope)
    # powerlaw_dnu_upper = median_dnu_freq1000*(freqs/1000)**(Slope+Slopeerr)
    # powerlaw_dnu_lower = median_dnu_freq1000*(freqs/1000)**(Slope-Slopeerr)
    # theory_dnu = median_dnu_freq1000*(freqs/1000)**(4)

    # fig = plt.figure(figsize=(20, 10))
    # fig.subplots_adjust(hspace=0.5, wspace=0.5)
    # cm = plt.cm.get_cmap('viridis')
    # z = phase
    # sc = plt.scatter(freq, dnu, c=z, cmap=cm, s=Size, alpha=0.6)
    # plt.colorbar(sc)
    # plt.errorbar(freq, dnu, yerr=dnuerr, fmt=' ', ecolor='k',
    #              elinewidth=2, capsize=3, alpha=0.55)
    # xl = plt.xlim()
    # plt.plot(xl, (df[0], df[0]), color='C2')
    # plt.plot(freqs, theory_dnu, color='k', linewidth=4)
    # plt.fill_between(freqs.flatten(),
    #                  powerlaw_dnu_upper.flatten(),
    #                  powerlaw_dnu_lower.flatten(),
    #                  alpha=0.6, color='C1')
    # plt.plot(freqs, powerlaw_dnu, color='C1', linewidth=4)
    # # plt.plot(unique_freqs, median_dnu_freqs, color='C1', linewidth=4)
    # # plt.errorbar(unique_freqs, median_dnu_freqs, fmt=' ',
    # # yerr=median_dnuerr_freqs,
    # #              color='C1', elinewidth=2, capsize=2)
    # plt.xlabel('Frequency (MHz)')
    # plt.ylabel('Scintillation Bandwidth (MHz)')
    # plt.title(psrname + r', $\Delta\nu$')
    # # plt.grid(True, which="both", ls="-", color='0.65')
    # plt.xlim(xl)
    # plt.ylim(0, 0.2)
    # # plt.savefig(str(Dnudir) + "Dnu_Orbital_Freq.png")
    # plt.show()
    # plt.close()
    # ###########################################################################
    # # Now I want to measure and fit the slope of the presumed power low to the
    # # data at each 'phase' i.e each 10min chunk of observing.

    # unique_mjd = np.unique(mjd)
    # unique_mjd_minimum = np.min(unique_mjd)
    # unique_mjd_diff = unique_mjd - unique_mjd_minimum
    # unique_mjd_min = (unique_mjd_diff)*1440

    # Slopes = []
    # Slopeerrs = []

    # for i in range(0, len(unique_mjd)):
    #     results_dir = outdir
    #     params = read_results(outfile)

    #     pars = read_par(str(par_dir) + str(psrname) + '.par')

    #     # Read in arrays
    #     mjd = float_array_from_dict(params, 'mjd')  # MJD for observation start
    #     df = float_array_from_dict(params, 'df')  # channel bandwidth
    #     dnu = float_array_from_dict(params, 'dnu')  # scint bandwidth
    #     dnu_est = float_array_from_dict(params, 'dnu_est')  # estimated
    #     dnuerr = float_array_from_dict(params, 'dnuerr')
    #     tau = float_array_from_dict(params, 'tau')
    #     tauerr = float_array_from_dict(params, 'tauerr')
    #     freq = float_array_from_dict(params, 'freq')
    #     bw = float_array_from_dict(params, 'bw')
    #     # scintle_num = float_array_from_dict(params, 'scintle_num')
    #     tobs = float_array_from_dict(params, 'tobs')  # tobs in second
    #     rcvrs = np.array([rcvr[0] for rcvr in params['name']])

    #     # Sort by MJD
    #     sort_ind = np.argsort(mjd)

    #     df = np.array(df[sort_ind]).squeeze()
    #     dnu = np.array(dnu[sort_ind]).squeeze()
    #     dnu_est = np.array(dnu_est[sort_ind]).squeeze()
    #     dnuerr = np.array(dnuerr[sort_ind]).squeeze()
    #     tau = np.array(tau[sort_ind]).squeeze()
    #     tauerr = np.array(tauerr[sort_ind]).squeeze()
    #     mjd = np.array(mjd[sort_ind]).squeeze()
    #     rcvrs = np.array(rcvrs[sort_ind]).squeeze()
    #     freq = np.array(freq[sort_ind]).squeeze()
    #     tobs = np.array(tobs[sort_ind]).squeeze()
    #     # scintle_num = np.array(scintle_num[sort_ind]).squeeze()
    #     bw = np.array(bw[sort_ind]).squeeze()

    #     mjd_annual = mjd % 365.2425
    #     print('Getting SSB delays')
    #     ssb_delays = get_ssb_delay(mjd, pars['RAJ'], pars['DECJ'])
    #     mjd += np.divide(ssb_delays, 86400)  # add ssb delay

    #     """
    #     Model Viss
    #     """
    #     print('Getting Earth velocity')
    #     vearth_ra, vearth_dec = get_earth_velocity(mjd, pars['RAJ'],
    #                                                pars['DECJ'])
    #     print('Getting true anomaly')
    #     pars['PBDOT'] *= 10**-12  # add in factor dropped by tempo
    #     U = get_true_anomaly(mjd, pars)

    #     true_anomaly = U.squeeze()
    #     vearth_ra = vearth_ra.squeeze()
    #     vearth_dec = vearth_dec.squeeze()

    #     om = pars['OM'] + pars['OMDOT']*(mjd - pars['T0'])/365.2425
    #     # compute orbital phase
    #     phase = U*180/np.pi + om
    #     phase[phase > 360] = phase[phase > 360] - 360

    #     small_change = 4.6295033826027066e-04

    #     # Used to filter the data
    #     indicies = np.argwhere((mjd < unique_mjd[i]+small_change) *
    #                            (mjd > unique_mjd[i]-small_change) *
    #                            (tauerr < 10*tau) * (dnuerr < 10*dnu))

    #     df = df[indicies].squeeze()
    #     dnu = dnu[indicies].squeeze()
    #     dnu_est = dnu_est[indicies].squeeze()
    #     dnuerr = dnuerr[indicies].squeeze()
    #     tau = tau[indicies].squeeze()
    #     tauerr = tauerr[indicies].squeeze()
    #     mjd = mjd[indicies].squeeze()
    #     rcvrs = rcvrs[indicies].squeeze()
    #     freq = freq[indicies].squeeze()
    #     tobs = tobs[indicies].squeeze()
    #     # scintle_num = scintle_num[indicies].squeeze()
    #     bw = bw[indicies].squeeze()

    #     print(len(mjd))
    #     print()
    #     if len(mjd) == 0:
    #         continue

    #     mjd_annual = mjd % 365.2425
    #     print('Getting SSB delays')
    #     ssb_delays = get_ssb_delay(mjd, pars['RAJ'], pars['DECJ'])
    #     mjd += np.divide(ssb_delays, 86400)  # add ssb delay

    #     """
    #     Model Viss
    #     """
    #     print('Getting Earth velocity')
    #     vearth_ra, vearth_dec = get_earth_velocity(mjd, pars['RAJ'],
    #                                                pars['DECJ'])
    #     print('Getting true anomaly')
    #     pars['PBDOT'] *= 10**-12  # add in factor dropped by tempo
    #     U = get_true_anomaly(mjd, pars)

    #     true_anomaly = U.squeeze()
    #     vearth_ra = vearth_ra.squeeze()
    #     vearth_dec = vearth_dec.squeeze()

    #     om = pars['OM'] + pars['OMDOT']*(mjd - pars['T0'])/365.2425
    #     # compute orbital phase
    #     phase = U*180/np.pi + om
    #     phase[phase > 360] = phase[phase > 360] - 360

    #     amp_init = np.average(dnu[np.argwhere((freq > 950))],
    #                           weights=dnuerr[np.argwhere((freq > 950))])
    #     freq_crop = freq[np.argwhere(freq > 800)]
    #     dnu_crop = dnu[np.argwhere(freq > 800)]
    #     dnuerr_crop = dnuerr[np.argwhere(freq > 800)]

    #     Slope, Slopeerr = \
    #         powerlaw_fitter(xdata=freq_crop, ydata=dnu_crop,
    #                         weights=dnuerr_crop, reffreq=1000,
    #                         amp_init=amp_init, steps=1000,
    #                         amp_max=1, alpha_max=6)
    #     Slopes.append(Slope)
    #     Slopeerrs.append(Slopeerr)
    #     freqs = np.arange(0, 2000)
    #     theory_dnu = amp_init*(freqs/1000)**(4)
    #     powerlaw_dnu = amp_init*(freqs/1000)**(Slope)
    #     powerlaw_dnu_upper = amp_init*(freqs/1000)**(Slope+Slopeerr)
    #     powerlaw_dnu_lower = amp_init*(freqs/1000)**(Slope-Slopeerr)

    #     Font = 30
    #     Size = 80*np.pi
    #     font = {'size': 28}
    #     matplotlib.rc('font', **font)

    #     fig = plt.figure(figsize=(20, 10))
    #     fig.subplots_adjust(hspace=0.5, wspace=0.5)
    #     plt.scatter(freq, dnu, c='C0', s=Size, alpha=0.6)
    #     plt.errorbar(freq, dnu, yerr=dnuerr, fmt=' ', ecolor='k',
    #                  elinewidth=2, capsize=3, alpha=0.55)
    #     xl = plt.xlim()
    #     plt.plot(xl, (df[0], df[0]), color='C2')
    #     plt.plot(freqs, theory_dnu, color='k', linewidth=4)
    #     plt.plot(freqs, powerlaw_dnu, color='C1', linewidth=4,
    #              label=r'$\alpha$ = '+str(round(Slope, 2))+r' $\pm$ ' +
    #              str(round(Slopeerr, 2)))
    #     plt.fill_between(freqs.flatten(),
    #                      powerlaw_dnu_upper.flatten(),
    #                      powerlaw_dnu_lower.flatten(),
    #                      alpha=0.6, color='C1')
    #     plt.xlabel('Frequency (MHz)')
    #     plt.ylabel('Scintillation Bandwidth (MHz)')
    #     plt.title(psrname + r', $\Delta\nu$')
    #     plt.xlim(xl)
    #     plt.ylim(0, 0.2)
    #     plt.savefig(str(plotdir) + "Dnu_Orbital_Freq" +
    #                 str(round(unique_mjd_min[i], 1)) + ".pdf")
    #     plt.show()
    #     plt.close()

    # Slopes = np.asarray(Slopes)
    # Slopeerrs = np.asarray(Slopeerrs)

    # Slope_average = np.average(Slopes, weights=Slopeerrs)
    # Slopeerr_median = np.median(Slopeerrs)

    # print("The average slope predicted with median uncertainty: " +
    #       str(round(np.average(Slopes, weights=Slopeerrs), 3)) + " +/- " +
    #       str(round(Slopeerr_median, 3)))

    # freqs = np.arange(0, 2000)
    # powerlaw_dnu = median_dnu_freq1000*(freqs/1000)**(Slope_average)
    # powerlaw_dnu_upper = median_dnu_freq1000*(freqs/1000)**(Slope +
    #                                                         Slopeerr_median)
    # powerlaw_dnu_lower = median_dnu_freq1000*(freqs/1000)**(Slope -
    #                                                         Slopeerr_median)
    # theory_dnu = median_dnu_freq1000*(freqs/1000)**(4)

    # results_dir = outdir
    # params = read_results(outfile)

    # pars = read_par(str(par_dir) + str(psrname) + '.par')

    # # Read in arrays
    # mjd = float_array_from_dict(params, 'mjd')  # MJD for observation start
    # df = float_array_from_dict(params, 'df')  # channel bandwidth
    # dnu = float_array_from_dict(params, 'dnu')  # scint bandwidth
    # dnu_est = float_array_from_dict(params, 'dnu_est')  # estimated bandwidth
    # dnuerr = float_array_from_dict(params, 'dnuerr')
    # tau = float_array_from_dict(params, 'tau')
    # tauerr = float_array_from_dict(params, 'tauerr')
    # freq = float_array_from_dict(params, 'freq')
    # bw = float_array_from_dict(params, 'bw')
    # # scintle_num = float_array_from_dict(params, 'scintle_num')
    # tobs = float_array_from_dict(params, 'tobs')  # tobs in second
    # rcvrs = np.array([rcvr[0] for rcvr in params['name']])

    # # Sort by MJD
    # sort_ind = np.argsort(mjd)

    # df = np.array(df[sort_ind]).squeeze()
    # dnu = np.array(dnu[sort_ind]).squeeze()
    # dnu_est = np.array(dnu_est[sort_ind]).squeeze()
    # dnuerr = np.array(dnuerr[sort_ind]).squeeze()
    # tau = np.array(tau[sort_ind]).squeeze()
    # tauerr = np.array(tauerr[sort_ind]).squeeze()
    # mjd = np.array(mjd[sort_ind]).squeeze()
    # rcvrs = np.array(rcvrs[sort_ind]).squeeze()
    # freq = np.array(freq[sort_ind]).squeeze()
    # tobs = np.array(tobs[sort_ind]).squeeze()
    # # scintle_num = np.array(scintle_num[sort_ind]).squeeze()
    # bw = np.array(bw[sort_ind]).squeeze()

    # # Used to filter the data
    # indicies = np.argwhere((tauerr < 10*tau) * (dnuerr < 10*dnu))

    # df = df[indicies].squeeze()
    # dnu = dnu[indicies].squeeze()
    # dnu_est = dnu_est[indicies].squeeze()
    # dnuerr = dnuerr[indicies].squeeze()
    # tau = tau[indicies].squeeze()
    # tauerr = tauerr[indicies].squeeze()
    # mjd = mjd[indicies].squeeze()
    # rcvrs = rcvrs[indicies].squeeze()
    # freq = freq[indicies].squeeze()
    # tobs = tobs[indicies].squeeze()
    # # scintle_num = scintle_num[indicies].squeeze()
    # bw = bw[indicies].squeeze()

    # mjd_annual = mjd % 365.2425
    # print('Getting SSB delays')
    # ssb_delays = get_ssb_delay(mjd, pars['RAJ'], pars['DECJ'])
    # mjd += np.divide(ssb_delays, 86400)  # add ssb delay

    # """
    # Model Viss
    # """
    # print('Getting Earth velocity')
    # vearth_ra, vearth_dec = get_earth_velocity(mjd, pars['RAJ'], pars['DECJ'])
    # print('Getting true anomaly')
    # pars['PBDOT'] *= 10**-12  # add in factor dropped by tempo
    # U = get_true_anomaly(mjd, pars)

    # true_anomaly = U.squeeze()
    # vearth_ra = vearth_ra.squeeze()
    # vearth_dec = vearth_dec.squeeze()

    # om = pars['OM'] + pars['OMDOT']*(mjd - pars['T0'])/365.2425
    # # compute orbital phase
    # phase = U*180/np.pi + om
    # phase[phase > 360] = phase[phase > 360] - 360

    # dnu_est2 = dnu_est - df[0]
    # dnu_est = np.sqrt(dnu_est**2 - df[0]**2)

    # freq_unique = np.unique(freq)
    # dnu_averages = []
    # dnu_est_averages = []
    # dnu_est2_averages = []
    # # residual_mjd_minutes = (mjd-np.min(mjd))*1400
    # # residual_mjd_minutes_unique = (unique_mjd-np.min(unique_mjd))*1400
    # for i in range(0, len(freq_unique)):
    #     target_dnu = np.argwhere(freq == freq_unique[i])
    #     dnu_averages.append(np.average(dnu[target_dnu],
    #                                    weights=dnuerr[target_dnu]))
    #     dnu_est_averages.append(np.average(dnu_est[target_dnu]))
    #     dnu_est2_averages.append(np.average(dnu_est2[target_dnu]))
    # dnu_averages = np.asarray(dnu_averages)
    # dnu_est_averages = np.asarray(dnu_est_averages)
    # dnu_est2_averages = np.asarray(dnu_est2_averages)

    # fig = plt.figure(figsize=(20, 10))
    # fig.subplots_adjust(hspace=0.5, wspace=0.5)
    # plt.scatter(freq_unique, dnu_averages, s=Size, alpha=0.6, c='C0')
    # plt.scatter(freq_unique, dnu_est_averages, s=Size, alpha=0.6, c='C4')
    # xl = plt.xlim()
    # plt.plot(xl, (df[0], df[0]), color='C2')
    # plt.plot(freqs, theory_dnu, color='k', linewidth=4)
    # plt.xlabel('Frequency (MHz)')
    # plt.ylabel('Scintillation Bandwidth (MHz)')
    # plt.title(psrname + r', $\Delta\nu$')
    # plt.xlim(xl)
    # plt.ylim(0, 0.2)
    # plt.show()
    # plt.close()

    # fig = plt.figure(figsize=(20, 10))
    # ax = fig.add_subplot(1, 1, 1)
    # fig.subplots_adjust(hspace=0.5, wspace=0.5)
    # cm = plt.cm.get_cmap('viridis')
    # z = phase
    # sc = plt.scatter(freq, dnu, c=z, cmap=cm, s=Size, alpha=0.5)
    # plt.colorbar(sc)
    # plt.errorbar(freq, dnu, yerr=dnuerr, fmt=' ', ecolor='k',
    #              elinewidth=2, capsize=3, alpha=0.4)
    # # plt.scatter(freq_unique, dnu_averages, s=Size, alpha=0.7, c='C3')
    # # plt.scatter(freq_unique, dnu_est_averages, s=Size, alpha=0.7, c='C4')
    # xl = plt.xlim()
    # plt.plot(xl, (df[0], df[0]), color='C2')
    # plt.plot(freqs, theory_dnu, color='k', linewidth=4, alpha=0.3)
    # plt.fill_between(freqs.flatten(),
    #                  powerlaw_dnu_upper.flatten(),
    #                  powerlaw_dnu_lower.flatten(),
    #                  alpha=0.2, color='C1')
    # plt.plot(freqs, powerlaw_dnu, color='C1', linewidth=4, alpha=0.6,
    #          label=r'New UHF $\alpha$ = '+str(round(Slope_average, 2)) +
    #          r' $\pm$ '+str(round(Slopeerr_median, 2)))
    # plt.xlabel('Frequency (MHz)')
    # plt.ylabel('Scintillation Bandwidth (MHz)')
    # plt.title(psrname + r', $\Delta\nu$')
    # plt.xlim(xl)
    # plt.ylim(0, 0.2)
    # ax.legend()
    # plt.savefig(str(plotdir) + "Dnu_Orbital_Freq.pdf")
    # plt.show()
    # plt.close()
    # ###########################################################################

    # # can you plot both dnu_new = dnu_est - channel_bw
    # # and dnu_new = sqrt(dnu_est**2 - channel_bw**2) ?

    # fig = plt.figure(figsize=(20, 10))
    # fig.subplots_adjust(hspace=0.5, wspace=0.5)
    # plt.scatter(freq_unique, dnu_averages, s=Size, alpha=0.6, c='C0')
    # plt.scatter(freq_unique, dnu_est_averages, s=Size, alpha=0.6, c='C4')
    # plt.scatter(freq_unique, dnu_est2_averages, s=Size, alpha=0.6, c='C6')
    # xl = plt.xlim()
    # plt.plot(xl, (df[0], df[0]), color='C2')
    # plt.plot(freqs, theory_dnu, color='k', linewidth=4)
    # plt.xlabel('Frequency (MHz)')
    # plt.ylabel('Scintillation Bandwidth (MHz)')
    # plt.title(psrname + r', $\Delta\nu$')
    # plt.xlim(xl)
    # plt.ylim(0, 0.125)
    # plt.show()
    # plt.close()
###############################################################################
if compare:
    # Here we are going to combine the two datasets and attempt to fit the
    # powerlaw
    Font = 35
    Size = 80*np.pi  # Determines the size of the datapoints used
    font = {'size': 32}
    matplotlib.rc('font', **font)

    old_datafile = \
        wd0+'Datafiles/J0737-3039A_' + \
            'ScintillationResults_TimescaleVariance_total.txt'
    params = read_results(old_datafile)

    pars = read_par(str(par_dir) + str(psrname) + '.par')

    # Read in arrays
    old_mjd = float_array_from_dict(params, 'mjd')  # MJD for observation start
    old_df = float_array_from_dict(params, 'df')  # channel bandwidth
    old_dnu = float_array_from_dict(params, 'dnu')  # scint bandwidth
    old_dnu_est = float_array_from_dict(params, 'dnu_est')  # estimated
    old_dnuerr = float_array_from_dict(params, 'dnuerr')
    old_tau = float_array_from_dict(params, 'tau')
    old_tauerr = float_array_from_dict(params, 'tauerr')
    old_freq = float_array_from_dict(params, 'freq')
    old_bw = float_array_from_dict(params, 'bw')
    # old_scintle_num = float_array_from_dict(params, 'scintle_num')
    old_tobs = float_array_from_dict(params, 'tobs')  # tobs in second
    old_rcvrs = np.array([rcvr[0] for rcvr in params['name']])

    # Sort by MJD
    sort_ind = np.argsort(old_mjd)

    old_df = np.array(old_df[sort_ind]).squeeze()
    old_dnu = np.array(old_dnu[sort_ind]).squeeze()
    old_dnu_est = np.array(old_dnu_est[sort_ind]).squeeze()
    old_dnuerr = np.array(old_dnuerr[sort_ind]).squeeze()
    old_tau = np.array(old_tau[sort_ind]).squeeze()
    old_tauerr = np.array(old_tauerr[sort_ind]).squeeze()
    old_mjd = np.array(old_mjd[sort_ind]).squeeze()
    old_rcvrs = np.array(old_rcvrs[sort_ind]).squeeze()
    old_freq = np.array(old_freq[sort_ind]).squeeze()
    old_tobs = np.array(old_tobs[sort_ind]).squeeze()
    # old_scintle_num = np.array(old_scintle_num[sort_ind]).squeeze()
    old_bw = np.array(old_bw[sort_ind]).squeeze()

    # Used to filter the data
    indicies = np.argwhere((old_tauerr < 0.2*old_tau) * (old_dnuerr <
                                                         0.2*old_dnu)
                           * (old_freq > 1465) *
                           (old_dnu < 2))  # old_dnu > old_df

    old_df = old_df[indicies].squeeze()
    old_dnu = old_dnu[indicies].squeeze()
    old_dnu_est = old_dnu_est[indicies].squeeze()
    old_dnuerr = old_dnuerr[indicies].squeeze()
    old_tau = old_tau[indicies].squeeze()
    old_tauerr = old_tauerr[indicies].squeeze()
    old_mjd = old_mjd[indicies].squeeze()
    old_rcvrs = old_rcvrs[indicies].squeeze()
    old_freq = old_freq[indicies].squeeze()
    old_tobs = old_tobs[indicies].squeeze()
    # old_scintle_num = old_scintle_num[indicies].squeeze()
    old_bw = old_bw[indicies].squeeze()

    mjd_annual = old_mjd % 365.2425
    print('Getting SSB delays')
    ssb_delays = get_ssb_delay(old_mjd, pars['RAJ'], pars['DECJ'])
    old_mjd += np.divide(ssb_delays, 86400)  # add ssb delay

    """
    Model Viss
    """
    print('Getting Earth velocity')
    old_vearth_ra, old_vearth_dec = get_earth_velocity(old_mjd, pars['RAJ'],
                                                       pars['DECJ'])
    print('Getting true anomaly')
    pars['PBDOT'] *= 10**-12  # add in factor dropped by tempo
    old_U = get_true_anomaly(old_mjd, pars)

    old_true_anomaly = old_U.squeeze()
    old_vearth_ra = old_vearth_ra.squeeze()
    old_vearth_dec = old_vearth_dec.squeeze()

    old_om = pars['OM'] + pars['OMDOT']*(old_mjd - pars['T0'])/365.2425
    # compute orbital phase
    old_phase = old_U*180/np.pi + old_om
    old_phase[old_phase > 360] = old_phase[old_phase > 360] - 360
###############################################################################

    results_dir = outdir
    params = read_results(outfile)

    pars = read_par(str(par_dir) + str(psrname) + '.par')

    # Read in arrays
    mjd = float_array_from_dict(params, 'mjd')  # MJD for observation start
    df = float_array_from_dict(params, 'df')  # channel bandwidth
    dnu = float_array_from_dict(params, 'dnu')  # scint bandwidth
    dnu_est = float_array_from_dict(params, 'dnu_est')  # estimated
    dnuerr = float_array_from_dict(params, 'dnuerr')
    tau = float_array_from_dict(params, 'tau')
    tauerr = float_array_from_dict(params, 'tauerr')
    freq = float_array_from_dict(params, 'freq')
    bw = float_array_from_dict(params, 'bw')
    # scintle_num = float_array_from_dict(params, 'scintle_num')
    tobs = float_array_from_dict(params, 'tobs')  # tobs in second
    rcvrs = np.array([rcvr[0] for rcvr in params['name']])

    # Sort by MJD
    sort_ind = np.argsort(mjd)

    df = np.array(df[sort_ind]).squeeze()
    dnu = np.array(dnu[sort_ind]).squeeze()
    dnu_est = np.array(dnu_est[sort_ind]).squeeze()
    dnuerr = np.array(dnuerr[sort_ind]).squeeze()
    tau = np.array(tau[sort_ind]).squeeze()
    tauerr = np.array(tauerr[sort_ind]).squeeze()
    mjd = np.array(mjd[sort_ind]).squeeze()
    rcvrs = np.array(rcvrs[sort_ind]).squeeze()
    freq = np.array(freq[sort_ind]).squeeze()
    tobs = np.array(tobs[sort_ind]).squeeze()
    # scintle_num = np.array(scintle_num[sort_ind]).squeeze()
    bw = np.array(bw[sort_ind]).squeeze()

    # Used to filter the data
    indicies = np.argwhere((tauerr < 10*tau) * (dnuerr < 10*dnu))

    df = df[indicies].squeeze()
    dnu = dnu[indicies].squeeze()
    dnu_est = dnu_est[indicies].squeeze()
    dnuerr = dnuerr[indicies].squeeze()
    tau = tau[indicies].squeeze()
    tauerr = tauerr[indicies].squeeze()
    mjd = mjd[indicies].squeeze()
    rcvrs = rcvrs[indicies].squeeze()
    freq = freq[indicies].squeeze()
    tobs = tobs[indicies].squeeze()
    # scintle_num = scintle_num[indicies].squeeze()
    bw = bw[indicies].squeeze()

    mjd_annual = mjd % 365.2425
    print('Getting SSB delays')
    ssb_delays = get_ssb_delay(mjd, pars['RAJ'], pars['DECJ'])
    mjd += np.divide(ssb_delays, 86400)  # add ssb delay

    """
    Model Viss
    """
    print('Getting Earth velocity')
    vearth_ra, vearth_dec = get_earth_velocity(mjd, pars['RAJ'],
                                               pars['DECJ'])
    print('Getting true anomaly')
    pars['PBDOT'] *= 10**-12  # add in factor dropped by tempo
    U = get_true_anomaly(mjd, pars)

    true_anomaly = U.squeeze()
    vearth_ra = vearth_ra.squeeze()
    vearth_dec = vearth_dec.squeeze()

    om = pars['OM'] + pars['OMDOT']*(mjd - pars['T0'])/365.2425
    # compute orbital phase
    phase = U*180/np.pi + om
    phase[phase > 360] = phase[phase > 360] - 360
###############################################################################
    # Combine the data
    freq = np.concatenate((freq, old_freq))
    dnu = np.concatenate((dnu, old_dnu))
    dnuerr = np.concatenate((dnuerr, old_dnuerr))
    phase = np.concatenate((phase, old_phase))
###############################################################################
    # Here i want to explore all the possible fits to the entire data at diff
    # phase
    phase_list = []
    dnu_list = []
    freq_list = []
    dnuerr_list = []
    for i in range(0, 18):
        dnu_list.append(np.asarray(dnu[np.argwhere((phase > i*20) *
                                                   (phase < (i+1)*20))]))
        dnuerr_list.append(np.asarray(dnuerr[np.argwhere((phase > i*20) *
                                                         (phase < (i+1)*20))]))
        phase_list.append(np.asarray(phase[np.argwhere((phase > i*20) *
                                                       (phase < (i+1)*20))]))
        freq_list.append(np.asarray(freq[np.argwhere((phase > i*20) *
                                                     (phase < (i+1)*20))]))
    dnu_array = np.asarray(dnu_list)
    phase_array = np.asarray(phase_list)
    freq_array = np.asarray(freq_list)
    dnuerr_array = np.asarray(dnuerr_list)
    Slopes = []
    Slopeerrs = []
    Amps = []
    Amperrs = []
    for i in range(0, 18):
        reffreq_init = 1000
        amp_init = 0.06
        Slope, Slopeerr, Amp, Amperr = \
            powerlaw_fitter(xdata=freq_array[i], ydata=dnu_array[i],
                            weights=dnuerr_array[i], reffreq=reffreq_init,
                            amp_init=amp_init, steps=1000, return_amp=True)
        if freq_array[i][0] < 1200:
            Slopes.append(Slope)
            Slopeerrs.append(Slopeerr)
            Amps.append(Amp)
            Amperrs.append(Amp)
            # Reffreqs.append(Reffreq)
            # Reffreqerrs.append(Reffreqerr)
        freqs = np.arange(0, 2000)
        theory_dnu = Amp*(freqs/reffreq_init)**(4)
        powerlaw_dnu = Amp*(freqs/reffreq_init)**(Slope)
        powerlaw_dnu_upper = Amp*(freqs/reffreq_init)**(Slope+Slopeerr)
        powerlaw_dnu_lower = Amp*(freqs/reffreq_init)**(Slope-Slopeerr)
        #
        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(1, 1, 1)
        fig.subplots_adjust(hspace=0.5, wspace=0.5)
        plt.fill_between(freqs.flatten(),
                         powerlaw_dnu_upper.flatten(),
                         powerlaw_dnu_lower.flatten(),
                         alpha=0.2, color='C3')
        plt.plot(freqs, powerlaw_dnu, color='C3', linewidth=4,
                 label=r'$\alpha$ = '+str(round(Slope, 2))+r' $\pm$ ' +
                 str(round(Slopeerr, 2)))
        plt.scatter(freq_array[i], dnu_array[i])
        xl = plt.xlim()
        plt.plot(freqs, theory_dnu, color='k', linewidth=4)
        plt.plot(xl, (df[0], df[0]), color='C2')
        plt.plot(xl, (old_df[0], old_df[0]), color='C2')
        plt.xlim(500, 1800)
        plt.ylim(0, 2)
        ax.legend()
        plt.xlabel('Frequency (MHz)')
        plt.ylabel('Scintillation Bandwidth (MHz)')
        plt.savefig(wd+"PowerLaw/power_law_fit_phase"+str(i*20)+".pdf")
        plt.show()
        plt.close()

    Slope_avg = np.average(np.asarray(Slopes), weights=np.asarray(Slopeerrs))
    Slopeerr_median = np.median(np.asarray(Slopeerrs))
    Amp_avg = np.average(np.asarray(Amps), weights=np.asarray(Amperrs))
    Amperr_median = np.median(np.asarray(Amperrs))
    theory_dnu = Amp_avg*(freqs/1000)**(4)
    powerlaw_dnu = Amp_avg*(freqs/1000)**(Slope_avg)
    powerlaw_dnu_upper = \
        Amp_avg*(freqs/1000)**(Slope_avg+Slopeerr_median)
    powerlaw_dnu_lower = \
        Amp_avg*(freqs/1000)**(Slope_avg-Slopeerr_median)

    # Old and new at specific phase Full #
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(1, 1, 1)
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    plt.fill_between(freqs.flatten(),
                     powerlaw_dnu_upper.flatten(),
                     powerlaw_dnu_lower.flatten(),
                     alpha=0.2, color='C3')
    plt.plot(freqs, powerlaw_dnu, color='C3', linewidth=4,
             label=r'$\alpha$ = '+str(round(Slope, 2))+r' $\pm$ ' +
             str(round(Slopeerr, 2)))
    plt.scatter(freq, dnu, s=Size, c='C0')
    xl = plt.xlim()
    plt.plot(freqs, theory_dnu, color='k', linewidth=4)
    plt.plot(xl, (old_df[0], old_df[0]), color='C2', linestyle='dashed',
             label='L-Band df = '+str(round(old_df[0], 2))+'MHz')
    plt.plot(xl, (df[0], df[0]), color='C2',
             label='UHF df = '+str(round(df[0], 2))+'MHz')
    plt.xlim(500, 1800)
    plt.ylim(0, 2)
    ax.legend()
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Scintillation Bandwidth (MHz)')
    plt.savefig(wd+"PowerLaw/power_law_fit_phase_full.pdf")
    plt.show()
    plt.close()

    # Old and new at specific phase Zoomed in #
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(1, 1, 1)
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    plt.fill_between(freqs.flatten(),
                     powerlaw_dnu_upper.flatten(),
                     powerlaw_dnu_lower.flatten(),
                     alpha=0.2, color='C3')
    plt.plot(freqs, powerlaw_dnu, color='C3', linewidth=4,
             label=r'$\alpha$ = '+str(round(Slope, 2))+r' $\pm$ ' +
             str(round(Slopeerr, 2)))
    plt.scatter(freq, dnu, s=Size, c='C0')
    xl = plt.xlim()
    plt.plot(freqs, theory_dnu, color='k', linewidth=4)
    plt.plot(xl, (old_df[0], old_df[0]), color='C2', linestyle='dashed',
             label='L-Band df = '+str(round(old_df[0], 2))+'MHz')
    plt.plot(xl, (df[0], df[0]), color='C2',
             label='UHF df = '+str(round(df[0], 2))+'MHz')
    plt.xlim(500, 1100)
    plt.ylim(0, 0.2)
    ax.legend()
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Scintillation Bandwidth (MHz)')
    plt.savefig(wd+"PowerLaw/power_law_fit_phase_zoomed.pdf")
    plt.show()
    plt.close()

###############################################################################

    # Determine the averages
    dnu_average = []
    unique_freq = np.unique(freq)
    for i in range(0, len(unique_freq)):
        dnu_average.append(float(
            np.average(dnu[np.argwhere((unique_freq[i]+1 > freq) *
                                       (unique_freq[i]-1 < freq))],
                       weights=dnuerr[np.argwhere((unique_freq[i]+1
                                                   > freq) * (unique_freq[i]-1
                                                              < freq))])))
    dnu_average = np.asarray(dnu_average)[np.argwhere(unique_freq > 800)]
    unique_freq = unique_freq[unique_freq > 800]
    dnu_average1 = dnu_average[np.argwhere(unique_freq < 1200)]
    dnu_average2 = dnu_average
    dnu_average3 = dnu_average[np.argwhere(unique_freq > 1200)]
    unique_freq1 = unique_freq[unique_freq < 1200]
    unique_freq1 = unique_freq1[unique_freq1 > 800]
    unique_freq2 = unique_freq[unique_freq > 800]
    unique_freq3 = unique_freq[unique_freq > 1200]

    # Compute the powerlaw
    freqs = np.arange(0, 2000)
    # All of the new UHF data ... orange
    amp_init = 0.065
    Reffreq = 900
    Slope1, Slopeerr1, Amp1, Amperr1 = \
        powerlaw_fitter(xdata=unique_freq1, ydata=dnu_average1, weights=None,
                        reffreq=Reffreq, amp_init=amp_init, return_amp=True)
    powerlaw_dnu1 = Amp1*(freqs/900)**(Slope1)
    powerlaw_dnu_upper1 = Amp1*(freqs/900)**(Slope1+Slopeerr1)
    powerlaw_dnu_lower1 = Amp1*(freqs/900)**(Slope1-Slopeerr1)

    # All of the data new and old combined (averaged) ... green
    amp_init = 0.5
    Reffreq = 1300
    Slope2, Slopeerr2, Amp2, Amperr2 = \
        powerlaw_fitter(xdata=unique_freq2, ydata=dnu_average2, weights=None,
                        reffreq=Reffreq, amp_init=amp_init, return_amp=True)
    powerlaw_dnu2 = Amp2*(freqs/1300)**(Slope2)
    powerlaw_dnu_upper2 = Amp2*(freqs/1300)**(Slope2+Slopeerr2)
    powerlaw_dnu_lower2 = Amp2*(freqs/1300)**(Slope2-Slopeerr2)

    # All of the data old L-band data ... red
    amp_init = 1.2
    Reffreq = 1600
    Slope3, Slopeerr3, Amp3, Amperr3 = \
        powerlaw_fitter(xdata=unique_freq3, ydata=dnu_average3, weights=None,
                        reffreq=Reffreq, amp_init=amp_init, return_amp=True)
    powerlaw_dnu3 = Amp3*(freqs/1600)**(Slope3)
    powerlaw_dnu_upper3 = Amp3*(freqs/1600)**(Slope3+Slopeerr3)
    powerlaw_dnu_lower3 = Amp3*(freqs/1600)**(Slope3-Slopeerr3)

    # All of the data new and old combined (non-filtered) ... no colour yet
    amp_init = 0.5
    Reffreq = 1300
    Slope4, Slopeerr4, Amp4, Amperr4 = \
        powerlaw_fitter(xdata=freq, ydata=dnu, weights=None, reffreq=Reffreq,
                        amp_init=amp_init, return_amp=True)
    powerlaw_dnu4 = Amp4*(freqs/1300)**(Slope4)
    powerlaw_dnu_upper4 = Amp4*(freqs/1300)**(Slope4+Slopeerr4)
    powerlaw_dnu_lower4 = Amp4*(freqs/1300)**(Slope4-Slopeerr4)

    # Two data points ... no colour yet
    amp_init = 0.6325000000000001
    Reffreq = 1300
    Slope5, Slopeerr5, Amp5, Amperr5 = \
        powerlaw_fitter(xdata=np.asarray([1000, 1600]),
                        ydata=np.asarray([0.065, 1.2]),
                        weights=None, reffreq=1300,
                        amp_init=0.6325000000000001,
                        return_amp=True)

    powerlaw_dnu5 = Amp5*(freqs/1300)**(Slope5)
    powerlaw_dnu_upper5 = Amp5*(freqs/1300)**(Slope5+Slopeerr5)
    powerlaw_dnu_lower5 = Amp5*(freqs/1300)**(Slope5-Slopeerr5)

    theory_dnu = Amp1*(freqs/900)**(4)
    theory_dnu2 = Amp2*(freqs/1300)**(4)
    theory_dnu3 = Amp3*(freqs/1600)**(4)
    theory_dnu4 = Amp4*(freqs/1300)**(4)
    theory_dnu5 = Amp5*(freqs/1300)**(4)
###############################################################################
    # Plot the data

    # TEST #

    # fig = plt.figure(figsize=(20, 10))
    # fig.subplots_adjust(hspace=0.5, wspace=0.5)
    # plt.scatter(freq, dnu, s=Size, alpha=0.5)
    # xl = plt.xlim()
    # plt.plot(xl, (df[0], df[0]), color='C2')
    # plt.plot(xl, (old_df[0], old_df[0]), color='C2')
    # plt.plot(freqs, theory_dnu, color='k', linewidth=4, alpha=0.3)
    # plt.fill_between(freqs.flatten(),
    #                  powerlaw_dnu_upper1.flatten(),
    #                  powerlaw_dnu_lower1.flatten(),
    #                  alpha=0.2, color='C3')
    # plt.plot(freqs, powerlaw_dnu1, color='C3', linewidth=4, alpha=0.3)
    # plt.xlabel('Frequency (MHz)')
    # plt.ylabel('Scintillation Bandwidth (MHz)')
    # plt.title(psrname + r', $\Delta\nu$')
    # plt.xlim(xl)
    # plt.ylim(0, 0.2)
    # plt.show()
    # plt.close()

    # Slope 1

    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(1, 1, 1)
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    plt.scatter(unique_freq1, dnu_average1, s=Size, alpha=0.5)
    xl = plt.xlim()
    plt.plot(xl, (df[0], df[0]), color='C2')
    plt.plot(xl, (old_df[0], old_df[0]), color='C2')
    plt.plot(freqs, theory_dnu, color='k', linewidth=4, alpha=0.3)
    plt.fill_between(freqs.flatten(),
                     powerlaw_dnu_upper1.flatten(),
                     powerlaw_dnu_lower1.flatten(),
                     alpha=0.2, color='C1')
    plt.plot(freqs, powerlaw_dnu1, color='C1', linewidth=4, alpha=0.3)
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Scintillation Bandwidth (MHz)')
    plt.title(psrname + r', $\Delta\nu$')
    ax.legend()
    plt.xlim(xl)
    plt.ylim(0, 0.15)
    plt.show()
    plt.close()

    # Slope 2

    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(1, 1, 1)
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    plt.scatter(unique_freq2, dnu_average2, s=Size, alpha=0.5)
    xl = plt.xlim()
    plt.plot(xl, (df[0], df[0]), color='C2')
    plt.plot(xl, (old_df[0], old_df[0]), color='C2')
    plt.plot(freqs, theory_dnu2, color='k', linewidth=4, alpha=0.3)
    plt.fill_between(freqs.flatten(),
                     powerlaw_dnu_upper2.flatten(),
                     powerlaw_dnu_lower2.flatten(),
                     alpha=0.2, color='C3')
    plt.plot(freqs, powerlaw_dnu2, color='C3', linewidth=4, alpha=0.3)
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Scintillation Bandwidth (MHz)')
    plt.title(psrname + r', $\Delta\nu$')
    ax.legend()
    plt.xlim(xl)
    plt.ylim(0, 2)
    plt.show()
    plt.close()

    # Slope 3

    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(1, 1, 1)
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    plt.scatter(unique_freq3, dnu_average3, s=Size, alpha=0.5)
    xl = plt.xlim()
    plt.plot(xl, (df[0], df[0]), color='C2')
    plt.plot(xl, (old_df[0], old_df[0]), color='C2')
    plt.plot(freqs, theory_dnu3, color='k', linewidth=4, alpha=0.3)
    plt.fill_between(freqs.flatten(),
                     powerlaw_dnu_upper3.flatten(),
                     powerlaw_dnu_lower3.flatten(),
                     alpha=0.2, color='C4')
    plt.plot(freqs, powerlaw_dnu3, color='C4', linewidth=4, alpha=0.3)
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Scintillation Bandwidth (MHz)')
    plt.title(psrname + r', $\Delta\nu$')
    ax.legend()
    plt.xlim(xl)
    plt.ylim(0, 2)
    plt.show()
    plt.close()

    # Slope 4

    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(1, 1, 1)
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    plt.scatter(freq, dnu, s=Size, alpha=0.5)
    xl = plt.xlim()
    plt.plot(xl, (df[0], df[0]), color='C2')
    plt.plot(xl, (old_df[0], old_df[0]), color='C2')
    plt.plot(freqs, theory_dnu4, color='k', linewidth=4, alpha=0.3)
    plt.fill_between(freqs.flatten(),
                     powerlaw_dnu_upper4.flatten(),
                     powerlaw_dnu_lower4.flatten(),
                     alpha=0.2, color='C5')
    plt.plot(freqs, powerlaw_dnu4, color='C5', linewidth=4, alpha=0.3)
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Scintillation Bandwidth (MHz)')
    plt.title(psrname + r', $\Delta\nu$')
    ax.legend()
    plt.xlim(xl)
    plt.ylim(0, 2)
    plt.show()
    plt.close()

    # Slope 5

    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(1, 1, 1)
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    plt.scatter(np.asarray([1000, 1600]), np.asarray([0.065, 1.2]), s=Size,
                alpha=0.5)
    xl = plt.xlim()
    plt.plot(xl, (df[0], df[0]), color='C2')
    plt.plot(xl, (old_df[0], old_df[0]), color='C2')
    plt.plot(freqs, theory_dnu5, color='k', linewidth=4, alpha=0.3)
    plt.fill_between(freqs.flatten(),
                     powerlaw_dnu_upper5.flatten(),
                     powerlaw_dnu_lower5.flatten(),
                     alpha=0.2, color='C6')
    plt.plot(freqs, powerlaw_dnu5, color='C6', linewidth=4, alpha=0.3)
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Scintillation Bandwidth (MHz)')
    plt.title(psrname + r', $\Delta\nu$')
    ax.legend()
    plt.xlim(xl)
    plt.ylim(0, 2)
    plt.show()
    plt.close()

    # ALL #

    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(1, 1, 1)
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    plt.scatter(freq, dnu, s=Size, alpha=0.5)
    xl = plt.xlim()
    plt.plot(xl, (old_df[0], old_df[0]), color='C2', linestyle='dashed',
             label='L-Band df = '+str(round(old_df[0], 2))+'MHz')
    plt.plot(xl, (df[0], df[0]), color='C2',
             label='UHF df = '+str(round(df[0], 2))+'MHz')
    # plt.plot(freqs, theory_dnu, color='k', linewidth=4, alpha=1)
    # plt.plot(freqs, theory_dnu2, color='k', linewidth=4, alpha=1)
    # plt.plot(freqs, theory_dnu3, color='k', linewidth=4, alpha=1)
    # plt.plot(freqs, theory_dnu4, color='k', linewidth=4, alpha=1)
    # plt.plot(freqs, theory_dnu5, color='k', linewidth=4, alpha=1)
    # plt.fill_between(freqs.flatten(),
    #                  powerlaw_dnu_upper1.flatten(),
    #                  powerlaw_dnu_lower1.flatten(),
    #                  alpha=0.2, color='C3')
    plt.plot(freqs, powerlaw_dnu1, color='C1', linewidth=4, alpha=1,
             label=r'New UHF $\alpha$ = ' + str(round(Slope1, 2))+r' $\pm$ ' +
             str(round(Slopeerr1, 2)))

    # plt.fill_between(freqs.flatten(),
    #                  powerlaw_dnu_upper2.flatten(),
    #                  powerlaw_dnu_lower2.flatten(),
    #                  alpha=0.2, color='C4')
    plt.plot(freqs, powerlaw_dnu2, color='C3', linewidth=4, alpha=1,
             label=r'All Data (average) $\alpha$ = ' + str(round(Slope2, 2)) +
             r' $\pm$ ' + str(round(Slopeerr2, 2)))
    # plt.fill_between(freqs.flatten(),
    #                  powerlaw_dnu_upper3.flatten(),
    #                  powerlaw_dnu_lower3.flatten(),
    #                  alpha=0.2, color='C5')
    plt.plot(freqs, powerlaw_dnu3, color='C4', linewidth=4, alpha=1,
             label=r'Old L-Band $\alpha$ = ' + str(round(Slope3, 2)) +
             r' $\pm$ ' + str(round(Slopeerr3, 2)))
    # plt.fill_between(freqs.flatten(),
    #                  powerlaw_dnu_upper4.flatten(),
    #                  powerlaw_dnu_lower4.flatten(),
    #                  alpha=0.2, color='C6')
    plt.plot(freqs, powerlaw_dnu4, color='C5', linewidth=4, alpha=1,
             label=r'All Data (un-processed) $\alpha$ = ' + str(round(Slope4,
                                                                      2)) +
             r' $\pm$ ' + str(round(Slopeerr4, 2)))
    # plt.fill_between(freqs.flatten(),
    #                  powerlaw_dnu_upper5.flatten(),
    #                  powerlaw_dnu_lower5.flatten(),
    #                  alpha=0.2, color='C7')
    plt.plot(freqs, powerlaw_dnu5, color='C6', linewidth=4, alpha=1,
             label=r'Two points $\alpha$ = ' + str(round(Slope5, 2)) +
             r' $\pm$ ' + str(round(Slopeerr5, 2)))
    ax.legend(fontsize='small')
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Scintillation Bandwidth (MHz)')
    plt.title(psrname + r', $\Delta\nu$')
    plt.xlim(xl)
    plt.ylim(0, 2.25)
    plt.show()
    plt.close()

    # ALL ZOOMED #

    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(1, 1, 1)
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    plt.scatter(freq, dnu, s=Size, alpha=0.5)
    xl = plt.xlim()
    plt.plot(xl, (old_df[0], old_df[0]), color='C2', linestyle='dashed',
             label='L-Band df= '+str(round(old_df[0], 2))+'MHz')
    plt.plot(xl, (df[0], df[0]), color='C2',
             label='UHF df= '+str(round(df[0], 2))+'MHz')
    # plt.plot(freqs, theory_dnu, color='k', linewidth=4, alpha=1)
    # plt.plot(freqs, theory_dnu2, color='k', linewidth=4, alpha=1)
    # plt.plot(freqs, theory_dnu3, color='k', linewidth=4, alpha=1)
    # plt.plot(freqs, theory_dnu4, color='k', linewidth=4, alpha=1)
    # plt.plot(freqs, theory_dnu5, color='k', linewidth=4, alpha=1)
    # plt.fill_between(freqs.flatten(),
    #                  powerlaw_dnu_upper1.flatten(),
    #                  powerlaw_dnu_lower1.flatten(),
    #                  alpha=0.2, color='C3')
    plt.plot(freqs, powerlaw_dnu1, color='C1', linewidth=4, alpha=1,
             label=r'New UHF $\alpha$ = ' + str(round(Slope1, 2))+r' $\pm$ ' +
             str(round(Slopeerr1, 2)))

    # plt.fill_between(freqs.flatten(),
    #                  powerlaw_dnu_upper2.flatten(),
    #                  powerlaw_dnu_lower2.flatten(),
    #                  alpha=0.2, color='C4')
    plt.plot(freqs, powerlaw_dnu2, color='C3', linewidth=4, alpha=1,
             label=r'All Data (average) $\alpha$ = ' + str(round(Slope2, 2)) +
             r' $\pm$ ' + str(round(Slopeerr2, 2)))
    # plt.fill_between(freqs.flatten(),
    #                  powerlaw_dnu_upper3.flatten(),
    #                  powerlaw_dnu_lower3.flatten(),
    #                  alpha=0.2, color='C5')
    plt.plot(freqs, powerlaw_dnu3, color='C4', linewidth=4, alpha=1,
             label=r'Old L-Band $\alpha$ = ' + str(round(Slope3, 2)) +
             r' $\pm$ ' + str(round(Slopeerr3, 2)))
    # plt.fill_between(freqs.flatten(),
    #                  powerlaw_dnu_upper4.flatten(),
    #                  powerlaw_dnu_lower4.flatten(),
    #                  alpha=0.2, color='C6')
    plt.plot(freqs, powerlaw_dnu4, color='C5', linewidth=4, alpha=1,
             label=r'All Data (un-processed) $\alpha$ = ' + str(round(Slope4,
                                                                      2)) +
             r' $\pm$ ' + str(round(Slopeerr4, 2)))
    # plt.fill_between(freqs.flatten(),
    #                   powerlaw_dnu_upper5.flatten(),
    #                   powerlaw_dnu_lower5.flatten(),
    #                   alpha=0.2, color='C7')
    plt.plot(freqs, powerlaw_dnu5, color='C6', linewidth=4, alpha=1,
             label=r'Two points $\alpha$ = ' + str(round(Slope5, 2)) +
             r' $\pm$ ' + str(round(Slopeerr5, 2)))
    ax.legend(fontsize='small')
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Scintillation Bandwidth (MHz)')
    plt.title(psrname + r', $\Delta\nu$')
    plt.xlim(600, 1100)
    plt.ylim(0, 0.2)
    plt.show()
    plt.close()

    # fig = plt.figure(figsize=(20, 10))
    # fig.subplots_adjust(hspace=0.5, wspace=0.5)
    # plt.scatter([1000, 1600], [0.065, 1.2], s=Size, alpha=0.5)
    # plt.scatter(unique_freq, dnu_average, s=Size, alpha=0.5)
    # xl = plt.xlim()
    # plt.plot(xl, (df[0], df[0]), color='C2')
    # plt.plot(xl, (old_df[0], old_df[0]), color='C2')
    # plt.plot(freqs, theory_dnu2, color='k', linewidth=4, alpha=0.3)
    # # plt.fill_between(freqs.flatten(),
    # #                  powerlaw_dnu_upper5.flatten(),
    # #                  powerlaw_dnu_lower5.flatten(),
    # #                  alpha=0.2, color='C5')
    # # plt.plot(freqs, powerlaw_dnu5, color='C5', linewidth=4, alpha=0.3)
    # plt.xlabel('Frequency (MHz)')
    # plt.ylabel('Scintillation Bandwidth (MHz)')
    # plt.title(psrname + r', $\Delta\nu$')
    # plt.xlim(xl)
    # # plt.xlim(600, 1100)
    # # plt.ylim(0, 0.2)
    # # plt.savefig(str(plotdir) + "Dnu_Orbital_Freq.pdf")
    # plt.show()
    # plt.close()
###############################################################################
    Slope_Theory = 5
    frequency_data = np.linspace(0, 1000, 1001)
    scintillation_data = np.random.rand((1001)) * \
        (frequency_data/500)**(Slope_Theory)
    indicies2 = np.argwhere((scintillation_data > 0.5 *
                             (frequency_data/500)**(Slope_Theory)))
    scintillation_data = scintillation_data[indicies2].squeeze()
    frequency_data = frequency_data[indicies2].squeeze()

    Slope, Slopeerr, Amp, Amperr = \
        powerlaw_fitter(xdata=frequency_data, ydata=scintillation_data,
                        weights=None, reffreq=500, amp_init=1, amp_min=-np.inf,
                        amp_max=np.inf, alpha_init=4, alpha_min=0,
                        alpha_max=np.inf, steps=10000, burn=0.20,
                        return_amp=True)
    model = Amp*(frequency_data/Reffreq)**(Slope)
    model_positive = Amp*(frequency_data/Reffreq)**(Slope+Slopeerr)
    model_negative = Amp*(frequency_data/Reffreq)**(Slope-Slopeerr)
    plt.scatter(frequency_data, scintillation_data, c='C0', alpha=0.7)
    plt.plot(frequency_data, Amp*(frequency_data/Reffreq)**(Slope_Theory),
             c='k')
    plt.plot(frequency_data, model, c='C1')
    plt.fill_between(frequency_data.flatten(),
                     model_positive.flatten(),
                     model_negative.flatten(),
                     alpha=0.2, color='C1')
###############################################################################
if model2:

    dnu_UHF = dnu[np.argwhere(freq < 1200)].flatten()
    dnuerr_UHF = dnuerr[np.argwhere(freq < 1200)].flatten()
    freq_UHF = freq[np.argwhere(freq < 1200)].flatten()

    dnu_Lband = dnu[np.argwhere(freq > 1200)].flatten()
    dnuerr_Lband = dnuerr[np.argwhere(freq > 1200)].flatten()
    freq_Lband = freq[np.argwhere(freq > 1200)].flatten()

###############################################################################

    xdata = freq_UHF[np.argsort(freq_UHF)]
    ydata = dnu_UHF[np.argsort(freq_UHF)]
    ydataerr = dnuerr_UHF[np.argsort(freq_UHF)]
    xdata2 = freq_Lband[np.argsort(freq_Lband)]
    ydata2 = dnu_Lband[np.argsort(freq_Lband)]
    ydataerr2 = dnuerr_Lband[np.argsort(freq_Lband)]
    xdata3 = freq[np.argsort(freq)]
    ydata3 = dnu[np.argsort(freq)]
    ydataerr3 = dnuerr[np.argsort(freq)]
    
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(1, 1, 1)
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    plt.scatter(xdata, ydata, c='C0', s=Size/4)
    plt.errorbar(xdata, ydata, yerr=ydataerr, fmt=' ', ecolor='k',
                 elinewidth=2, capsize=3, alpha=0.55)
    plt.scatter(xdata2, ydata2, c='C0', s=Size/4)
    plt.errorbar(xdata2, ydata2, yerr=ydataerr2, fmt=' ', ecolor='k',
                 elinewidth=2, capsize=3, alpha=0.55)
    xl = plt.xlim()
    xdata4 = np.linspace(xl[0], xl[1], 1000)
    plt.plot(xl, (old_df[0], old_df[0]), color='C2', linestyle='dashed',
             label='L-Band df='+str(round(old_df[0], 2))+'MHz')
    plt.plot(xl, (df[0], df[0]), color='C2',
             label='UHF df='+str(round(df[0], 2))+'MHz')
    popt, pcov = curve_fit(func1, xdata, ydata)
    plt.plot(xdata4, func1(xdata4, *popt), 'C1', label=r'UHF $\alpha$=' +
             str(round(popt[0], 2)))
    perr = np.sqrt(np.diag(pcov))
    plt.fill_between(xdata4.flatten(),
                     func1(xdata4, *[popt[0]+perr[0], popt[1]]).flatten(),
                     func1(xdata4, *[popt[0]-perr[0], popt[1]]).flatten(),
                     alpha=0.5, color='C1')
    popt2, pcov2 = curve_fit(func2, xdata2, ydata2)
    plt.plot(xdata4, func2(xdata4, *popt2), 'C3', label=r'L-band $\alpha$=' +
             str(round(popt2[0], 2)))
    perr2 = np.sqrt(np.diag(pcov2))
    plt.fill_between(xdata4,
                     func2(xdata4, *[popt2[0]+perr2[0], popt2[1]]).flatten(),
                     func2(xdata4, *[popt2[0]-perr2[0], popt2[1]]).flatten(),
                     alpha=0.5, color='C3')
    popt3, pcov3 = curve_fit(func3, xdata3, ydata3)
    perr3 = np.sqrt(np.diag(pcov3))
    plt.plot(xdata4, func3(xdata4, *popt3), 'C4', label=r'ALL $\alpha$=' +
             str(round(popt3[0], 2)))
    plt.fill_between(xdata4.flatten(),
                     func3(xdata4, *[popt3[0]+perr3[0], popt3[1]]).flatten(),
                     func3(xdata4, *[popt3[0]-perr3[0], popt3[1]]).flatten(),
                     alpha=0.5, color='C4')
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Scintillation Bandwidth (MHz)')
    ax.legend()
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.xlim(xl)
    plt.ylim(0, np.max(dnu)*1.05)
    plt.show()

###############################################################################
# Here I have copied code and attempted to do the same thing with a different
# method

if compare:
    # Here we are going to combine the two datasets and attempt to fit the
    # powerlaw
    Font = 35
    Size = 80*np.pi  # Determines the size of the datapoints used
    font = {'size': 32}
    matplotlib.rc('font', **font)

    old_datafile = \
        wd0+'Datafiles/J0737-3039A_' + \
            'ScintillationResults_TimescaleVariance_total.txt'
    params = read_results(old_datafile)

    pars = read_par(str(par_dir) + str(psrname) + '.par')

    # Read in arrays
    old_mjd = float_array_from_dict(params, 'mjd')  # MJD for observation start
    old_df = float_array_from_dict(params, 'df')  # channel bandwidth
    old_dnu = float_array_from_dict(params, 'dnu')  # scint bandwidth
    old_dnu_est = float_array_from_dict(params, 'dnu_est')  # estimated
    old_dnuerr = float_array_from_dict(params, 'dnuerr')
    old_tau = float_array_from_dict(params, 'tau')
    old_tauerr = float_array_from_dict(params, 'tauerr')
    old_freq = float_array_from_dict(params, 'freq')
    old_bw = float_array_from_dict(params, 'bw')
    # old_scintle_num = float_array_from_dict(params, 'scintle_num')
    old_tobs = float_array_from_dict(params, 'tobs')  # tobs in second
    old_rcvrs = np.array([rcvr[0] for rcvr in params['name']])

    # Sort by MJD
    sort_ind = np.argsort(old_mjd)

    old_df = np.array(old_df[sort_ind]).squeeze()
    old_dnu = np.array(old_dnu[sort_ind]).squeeze()
    old_dnu_est = np.array(old_dnu_est[sort_ind]).squeeze()
    old_dnuerr = np.array(old_dnuerr[sort_ind]).squeeze()
    old_tau = np.array(old_tau[sort_ind]).squeeze()
    old_tauerr = np.array(old_tauerr[sort_ind]).squeeze()
    old_mjd = np.array(old_mjd[sort_ind]).squeeze()
    old_rcvrs = np.array(old_rcvrs[sort_ind]).squeeze()
    old_freq = np.array(old_freq[sort_ind]).squeeze()
    old_tobs = np.array(old_tobs[sort_ind]).squeeze()
    # old_scintle_num = np.array(old_scintle_num[sort_ind]).squeeze()
    old_bw = np.array(old_bw[sort_ind]).squeeze()

    # Used to filter the data
    indicies = np.argwhere((old_tauerr < 0.2*old_tau) * (old_dnuerr <
                                                         0.2*old_dnu)
                           * (old_freq > 1465) *
                           (old_dnu < 2))  # old_dnu > old_df

    old_df = old_df[indicies].squeeze()
    old_dnu = old_dnu[indicies].squeeze()
    old_dnu_est = old_dnu_est[indicies].squeeze()
    old_dnuerr = old_dnuerr[indicies].squeeze()
    old_tau = old_tau[indicies].squeeze()
    old_tauerr = old_tauerr[indicies].squeeze()
    old_mjd = old_mjd[indicies].squeeze()
    old_rcvrs = old_rcvrs[indicies].squeeze()
    old_freq = old_freq[indicies].squeeze()
    old_tobs = old_tobs[indicies].squeeze()
    # old_scintle_num = old_scintle_num[indicies].squeeze()
    old_bw = old_bw[indicies].squeeze()

    mjd_annual = old_mjd % 365.2425
    print('Getting SSB delays')
    ssb_delays = get_ssb_delay(old_mjd, pars['RAJ'], pars['DECJ'])
    old_mjd += np.divide(ssb_delays, 86400)  # add ssb delay

    """
    Model Viss
    """
    print('Getting Earth velocity')
    old_vearth_ra, old_vearth_dec = get_earth_velocity(old_mjd, pars['RAJ'],
                                                       pars['DECJ'])
    print('Getting true anomaly')
    pars['PBDOT'] *= 10**-12  # add in factor dropped by tempo
    old_U = get_true_anomaly(old_mjd, pars)

    old_true_anomaly = old_U.squeeze()
    old_vearth_ra = old_vearth_ra.squeeze()
    old_vearth_dec = old_vearth_dec.squeeze()

    old_om = pars['OM'] + pars['OMDOT']*(old_mjd - pars['T0'])/365.2425
    # compute orbital phase
    old_phase = old_U*180/np.pi + old_om
    old_phase[old_phase > 360] = old_phase[old_phase > 360] - 360
###############################################################################

    results_dir = outdir
    params = read_results(outfile_total)

    pars = read_par(str(par_dir) + str(psrname) + '.par')

    # Read in arrays
    mjd = float_array_from_dict(params, 'mjd')  # MJD for observation start
    df = float_array_from_dict(params, 'df')  # channel bandwidth
    dnu = float_array_from_dict(params, 'dnu')  # scint bandwidth
    dnu_est = float_array_from_dict(params, 'dnu_est')  # estimated
    dnuerr = float_array_from_dict(params, 'dnuerr')
    tau = float_array_from_dict(params, 'tau')
    tauerr = float_array_from_dict(params, 'tauerr')
    freq = float_array_from_dict(params, 'freq')
    bw = float_array_from_dict(params, 'bw')
    # scintle_num = float_array_from_dict(params, 'scintle_num')
    tobs = float_array_from_dict(params, 'tobs')  # tobs in second
    rcvrs = np.array([rcvr[0] for rcvr in params['name']])

    # Sort by MJD
    sort_ind = np.argsort(mjd)

    df = np.array(df[sort_ind]).squeeze()
    dnu = np.array(dnu[sort_ind]).squeeze()
    dnu_est = np.array(dnu_est[sort_ind]).squeeze()
    dnuerr = np.array(dnuerr[sort_ind]).squeeze()
    tau = np.array(tau[sort_ind]).squeeze()
    tauerr = np.array(tauerr[sort_ind]).squeeze()
    mjd = np.array(mjd[sort_ind]).squeeze()
    rcvrs = np.array(rcvrs[sort_ind]).squeeze()
    freq = np.array(freq[sort_ind]).squeeze()
    tobs = np.array(tobs[sort_ind]).squeeze()
    # scintle_num = np.array(scintle_num[sort_ind]).squeeze()
    bw = np.array(bw[sort_ind]).squeeze()

    # Used to filter the data
    indicies = np.argwhere((tauerr < 10*tau) * (dnuerr < 10*dnu))

    df = df[indicies].squeeze()
    dnu = dnu[indicies].squeeze()
    dnu_est = dnu_est[indicies].squeeze()
    dnuerr = dnuerr[indicies].squeeze()
    tau = tau[indicies].squeeze()
    tauerr = tauerr[indicies].squeeze()
    mjd = mjd[indicies].squeeze()
    rcvrs = rcvrs[indicies].squeeze()
    freq = freq[indicies].squeeze()
    tobs = tobs[indicies].squeeze()
    # scintle_num = scintle_num[indicies].squeeze()
    bw = bw[indicies].squeeze()

    mjd_annual = mjd % 365.2425
    print('Getting SSB delays')
    ssb_delays = get_ssb_delay(mjd, pars['RAJ'], pars['DECJ'])
    mjd += np.divide(ssb_delays, 86400)  # add ssb delay

    """
    Model Viss
    """
    print('Getting Earth velocity')
    vearth_ra, vearth_dec = get_earth_velocity(mjd, pars['RAJ'],
                                               pars['DECJ'])
    print('Getting true anomaly')
    pars['PBDOT'] *= 10**-12  # add in factor dropped by tempo
    U = get_true_anomaly(mjd, pars)

    true_anomaly = U.squeeze()
    vearth_ra = vearth_ra.squeeze()
    vearth_dec = vearth_dec.squeeze()

    om = pars['OM'] + pars['OMDOT']*(mjd - pars['T0'])/365.2425
    # compute orbital phase
    phase = U*180/np.pi + om
    phase[phase > 360] = phase[phase > 360] - 360
###############################################################################
    # Combine the data
    freq = np.concatenate((freq, old_freq))
    dnu = np.concatenate((dnu, old_dnu))
    dnuerr = np.concatenate((dnuerr, old_dnuerr))
    phase = np.concatenate((phase, old_phase))
###############################################################################
    # Here i want to explore all the possible fits to the entire data at diff
    # phase
    phase_list = []
    dnu_list = []
    freq_list = []
    dnuerr_list = []
    for i in range(0, 18):
        dnu_list.append(np.asarray(dnu[np.argwhere((phase > i*20) *
                                                   (phase < (i+1)*20))]))
        dnuerr_list.append(np.asarray(dnuerr[np.argwhere((phase > i*20) *
                                                         (phase < (i+1)*20))]))
        phase_list.append(np.asarray(phase[np.argwhere((phase > i*20) *
                                                       (phase < (i+1)*20))]))
        freq_list.append(np.asarray(freq[np.argwhere((phase > i*20) *
                                                     (phase < (i+1)*20))]))
    dnu_array = np.asarray(dnu_list)
    phase_array = np.asarray(phase_list)
    freq_array = np.asarray(freq_list)
    dnuerr_array = np.asarray(dnuerr_list)
    Slopes1 = []
    Slopeerrs1 = []
    Slopes2 = []
    Slopeerrs2 = []
    Slopes3 = []
    Slopeerrs3 = []
    for i in range(0, 18):

        xdataA = freq_array[i][np.argwhere(freq_array[i] < 1200)].flatten()
        ydataA = dnu_array[i][np.argwhere(freq_array[i] < 1200)].flatten()
        ydataerrA = dnuerr_array[i][np.argwhere(freq_array[i] < 1200)].flatten()
        xdata = freq_array[i][np.argwhere((freq_array[i] > 800) *
                                          (freq_array[i] < 1200))].flatten()
        ydata = dnu_array[i][np.argwhere((freq_array[i] > 800) *
                                         (freq_array[i] < 1200))].flatten()
        ydataerr = dnuerr_array[i][np.argwhere((freq_array[i] > 800) *
                                               (freq_array[i] < 1200))].flatten()

        xdata2 = freq_array[i][np.argwhere((freq_array[i] > 800) *
                                           (freq_array[i] > 1200))].flatten()
        ydata2 = dnu_array[i][np.argwhere((freq_array[i] > 800) *
                                          (freq_array[i] > 1200))].flatten()
        ydataerr2 = dnuerr_array[i][np.argwhere((freq_array[i] > 800) *
                                                (freq_array[i] > 1200))].flatten()

        xdata3 = freq_array[i][np.argwhere(freq_array[i] > 800)].flatten()
        ydata3 = dnu_array[i][np.argwhere(freq_array[i] > 800)].flatten()
        ydataerr3 = dnuerr_array[i][np.argwhere(freq_array[i] > 800)].flatten()

        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(1, 1, 1)
        fig.subplots_adjust(hspace=0.5, wspace=0.5)
        xdata4 = np.linspace(np.min(freq_array[1])*0.75,
                             np.max(freq_array[1])*1.25,
                             1000)
# xdata
        if xdata.size > 0:
            plt.scatter(xdataA, ydataA, c='C0', s=Size/4)
            plt.errorbar(xdataA, ydataA, yerr=ydataerrA,
                         fmt=' ', ecolor='k',
                         elinewidth=2, capsize=3, alpha=0.55)
            popt, pcov = curve_fit(func1, xdata, ydata)
            perr = np.sqrt(np.diag(pcov))
            plt.plot(xdata4, func1(xdata4, *popt), 'C1', label=r'UHF $\alpha$=' +
                     str(round(popt[0], 2))+r'$\pm$'+str(round(perr[0], 2)))
            plt.fill_between(xdata4.flatten(),
                             func1(xdata4, *[popt[0]+perr[0], popt[1]]).flatten(),
                             func1(xdata4, *[popt[0]-perr[0], popt[1]]).flatten(),
                             alpha=0.5, color='C1')
            Slopes1.append(popt)
            Slopeerrs1.append(perr[0])

# xdata2
        if xdata2.size > 0:
            plt.scatter(xdata2, ydata2, c='C0', s=Size/4)
            plt.errorbar(xdata2, ydata2, yerr=ydataerr2, fmt=' ', ecolor='k',
                         elinewidth=2, capsize=3, alpha=0.55)
            popt2, pcov2 = curve_fit(func2, xdata2, ydata2)
            perr2 = np.sqrt(np.diag(pcov2))
            plt.plot(xdata4, func2(xdata4, *popt2), 'C3', label=r'L-band $\alpha$=' +
                     str(round(popt2[0], 2))+r'$\pm$'+str(round(perr2[0], 2)))
            plt.fill_between(xdata4,
                             func2(xdata4, *[popt2[0]+perr2[0], popt2[1]]).flatten(),
                             func2(xdata4, *[popt2[0]-perr2[0], popt2[1]]).flatten(),
                             alpha=0.5, color='C3')
            Slopes2.append(popt2)
            Slopeerrs2.append(perr2[0])
# xdata3
        popt3, pcov3 = curve_fit(func3, xdata3, ydata3)
        perr3 = np.sqrt(np.diag(pcov3))
        plt.plot(xdata4, func3(xdata4, *popt3), 'C4', label=r'ALL $\alpha$=' +
                 str(round(popt3[0], 2))+r'$\pm$'+str(round(perr3[0], 2)))
        plt.fill_between(xdata4.flatten(),
                         func3(xdata4, *[popt3[0]+perr3[0], popt3[1]]).flatten(),
                         func3(xdata4, *[popt3[0]-perr3[0], popt3[1]]).flatten(),
                         alpha=0.5, color='C4')
        Slopes3.append(popt3)
        Slopeerrs3.append(perr3[0])

# Rest of plot

        plt.plot(xl, (old_df[0], old_df[0]), color='C2', linestyle='dashed',
                 label='L-Band df='+str(round(old_df[0], 2))+'MHz')
        plt.plot(xl, (df[0], df[0]), color='C2',
                 label='UHF df='+str(round(df[0], 2))+'MHz')
        plt.xlabel('Frequency (MHz)')
        plt.ylabel('Scintillation Bandwidth (MHz)')
        ax.legend()
        ax.set_xscale('log')
        ax.set_yscale('log')
        plt.xlim(xl)
        plt.ylim(0, np.max(dnu)*1.05)
        plt.show()

    Slopes_average1 = np.mean(Slopes1, axis=0)
    Slopeerrs_median1 = np.median(Slopeerrs1)
    Slopes_average2 = np.mean(Slopes2, axis=0)
    Slopeerrs_median2 = np.median(Slopeerrs2)
    Slopes_average3 = np.mean(Slopes3, axis=0)
    Slopeerrs_median3 = np.median(Slopeerrs3)

    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(1, 1, 1)
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    xdata4 = np.linspace(np.min(freq)*0.75,
                         np.max(freq)*1.25,
                         1000)
    plt.scatter(freq, dnu, c='C0', s=Size/4)
    plt.errorbar(freq, dnu, yerr=dnuerr, fmt=' ', ecolor='k',
                 elinewidth=2, capsize=3, alpha=0.55)
    plt.plot(xdata4, func1(xdata4, *Slopes_average1), 'C1', label=r'UHF $\alpha$=' +
             str(round(Slopes_average1[0], 2))+r'$\pm$'+str(round(Slopeerrs_median1, 2)))
    xl = plt.xlim()
    plt.fill_between(xdata4.flatten(),
                     func1(xdata4, *[Slopes_average1[0]+Slopeerrs_median1,
                                     Slopes_average1[1]]).flatten(),
                     func1(xdata4, *[Slopes_average1[0]-Slopeerrs_median1,
                                     Slopes_average1[1]]).flatten(),
                     alpha=0.5, color='C1')
    plt.plot(xdata4, func2(xdata4, *Slopes_average2), 'C3', label=r'L-band $\alpha$=' +
             str(round(Slopes_average2[0], 2))+r'$\pm$'+str(round(Slopeerrs_median2, 2)))
    plt.fill_between(xdata4.flatten(),
                     func2(xdata4, *[Slopes_average2[0]+Slopeerrs_median2,
                                     Slopes_average2[1]]).flatten(),
                     func2(xdata4, *[Slopes_average2[0]-Slopeerrs_median2,
                                     Slopes_average2[1]]).flatten(),
                     alpha=0.5, color='C3')
    plt.plot(xdata4, func3(xdata4, *Slopes_average3), 'C4', label=r'ALL $\alpha$=' +
             str(round(Slopes_average3[0], 2))+r'$\pm$'+str(round(Slopeerrs_median3, 2)))
    plt.fill_between(xdata4.flatten(),
                     func3(xdata4, *[Slopes_average3[0]+Slopeerrs_median3,
                                     Slopes_average3[1]]).flatten(),
                     func3(xdata4, *[Slopes_average3[0]-Slopeerrs_median3,
                                     Slopes_average3[1]]).flatten(),
                     alpha=0.5, color='C4')
    plt.plot(xl, (old_df[0], old_df[0]), color='C2', linestyle='dashed',
             label='L-Band df='+str(round(old_df[0], 2))+'MHz')
    plt.plot(xl, (df[0], df[0]), color='C2',
             label='UHF df='+str(round(df[0], 2))+'MHz')
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Scintillation Bandwidth (MHz)')
    ax.legend()
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.xlim(xl)
    plt.ylim(0, np.max(dnu)*1.05)
    plt.show()
    plt.close()
