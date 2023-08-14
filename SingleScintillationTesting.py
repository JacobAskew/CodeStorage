#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 09:38:56 2021

@author: jaskew
"""
##############################################################################
# Common #
from scintools.dynspec import Dynspec
from scintools.scint_utils import write_results, read_results, read_par, \
        float_array_from_dict, get_ssb_delay, get_earth_velocity, \
        get_true_anomaly, scint_velocity
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import glob
from copy import deepcopy as cp

psrname = 'J0737-3039A'
pulsar = '0737-3039A'
##############################################################################
# OzStar #
# datadir = '/fred/oz002/jaskew/Data/' + str(psrname) + '/'
# filedir = str(datadir) + '/Datafiles/'
# spectradir = str(datadir) + '/Spectra/'
# par_dir = '/fred/oz002/jaskew/Data/ParFiles/'
# eclipsefile = '/fred/oz002/jaskew/Eclipse_mjd.txt'
# plotdir = str(datadir) + '/Plots/'
# outdir = filedir
##############################################################################
# Local #
wd = '/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/'
datadir = '/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/Data/'
spectradir = '/Users/jacobaskew/Desktop/DoublePulsar_Project/' + str(pulsar)\
    + '/Spectra/'
par_dir = '/Users/jacobaskew/Desktop/Swinburne_Daniel/ParFiles/'
plotdir = '/Users/jacobaskew/Desktop/DoublePulsar_Project/' + str(pulsar) +\
    '/Plots/'
eclipsefile = str(wd) + 'Datafiles/Eclipse_mjd.txt'
outdir = '/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/Datafiles/'
##############################################################################


def SearchEclipse(start_mjd, tobs):
    end_mjd = start_mjd + tobs/86400
    Eclipse_mjd = np.genfromtxt(eclipsefile, delimiter=',',
                                encoding=None, dtype=float)
    Eclipse_events = np.array(np.where((Eclipse_mjd > start_mjd) *
                              (Eclipse_mjd < end_mjd)))
    if Eclipse_events.size == 0:
        Eclipse_index = None
        print("No Eclispe in dynspec")
    else:
        Eclipse_events_mjd = Eclipse_mjd[Eclipse_events]
        mjds = start_mjd + dyn.times/86400
        Eclipse_index = np.argmin(abs(mjds - Eclipse_events_mjd))
    return Eclipse_index


##############################################################################
# Manual Inputs #
measure = False
model = True

zap = True
linear = True

outfile = str(outdir) + 'J0737-3039A_SingleScintillationResults'
if zap:
    outfile += '_zap'
if linear:
    outfile += '_linear'
outfile += '.txt'
##############################################################################
# Single analysis on this file.

dynspec = str(datadir) + \
    'J0737-3039A_2020-12-22-03:15:28_ch5.0_sub5.0.ar.dynspec'

Filename = 'J0737-3039A_2020-12-22-03:15:28_ch5'
if measure:

    try:
        dyn = Dynspec(filename=dynspec, process=False)
        dyn.trim_edges()
        start_mjd = dyn.mjd
        tobs = dyn.tobs
        Eclipse_index = SearchEclipse(start_mjd, tobs)
        if Eclipse_index is not True:
            dyn.dyn[:, Eclipse_index-3:Eclipse_index+3] = 0
        dyn.plot_dyn(filename=str(spectradir) + str(Filename) +
                     '_Spectra.png')
    except Exception as e:
        print(e)

    if dyn.freq > 1000:
        dyn_crop = cp(dyn)
        dyn_crop.crop_dyn(fmin=1650)
        f_high = 1660
        bw_high = 50
        for istart_f in range(1650, int(1670), int(10)):
            for istart_t in range(0, int(dyn.tobs/60), 10):
                try:
                    FreqRange = 'High'
                    dyn_new = cp(dyn_crop)
                    dyn_new.crop_dyn(fmin=istart_f, fmax=istart_f+10,
                                     tmin=istart_t, tmax=istart_t + 10)
                    dyn_new.trim_edges()
                    if dyn_new.tobs <= 5:
                        continue
                    if zap:
                        dyn_new.zap()
                    if linear:
                        dyn_new.refill(linear=True)
                    else:
                        dyn_new.refill(linear=False)
                    dyn_new.get_acf_tilt()
                    dyn_new.get_scint_params(method='acf2d_approx',
                                             flux_estimate=True)
                    dyn_new.plot_dyn()
                    dyn_new.plot_acf(fit=True)
                    write_results(outfile, dyn=dyn_new)
                except Exception as e:
                    print(e)
                    continue

        dyn_crop = cp(dyn)
        dyn_crop.crop_dyn(fmin=1300, fmax=1500)
        f_mid = dyn_crop.freq
        bw_mid = (f_mid / f_high)**2 * bw_high
        for istart_f in range(1300, 1500, int(bw_mid)):
            for istart_t in range(0, int(dyn.tobs/60), 10):
                try:
                    FreqRange = 'Mid'
                    dyn_new = cp(dyn_crop)
                    dyn_new.crop_dyn(fmin=istart_f, fmax=istart_f +
                                     int(bw_mid), tmin=istart_t,
                                     tmax=istart_t + 10)
                    dyn_new.trim_edges()
                    if dyn_new.tobs <= 5 or dyn_new.bw < bw_mid*0.9:
                        continue
                    if zap:
                        dyn_new.zap()
                    if linear:
                        dyn_new.refill(linear=True)
                    else:
                        dyn_new.refill(linear=False)
                    dyn_new.get_acf_tilt()
                    dyn_new.get_scint_params(method='acf2d_approx',
                                             flux_estimate=True)
                    dyn_new.plot_dyn()
                    dyn_new.plot_acf(fit=True)
                    write_results(outfile, dyn=dyn_new)
                except Exception as e:
                    print(e)
                    continue

        dyn_crop = cp(dyn)
        dyn_crop.crop_dyn(fmin=975, fmax=1075)
        f_low = dyn_crop.freq
        bw_low = (f_low / f_high)**2 * bw_high
        for istart_f in range(975, 1075, int(bw_low)):
            for istart_t in range(0, int(dyn.tobs/60), 10):
                try:
                    FreqRange = 'Low'
                    dyn_new = cp(dyn_crop)
                    dyn_new.crop_dyn(fmin=istart_f, fmax=istart_f +
                                     int(bw_low), tmin=istart_t,
                                     tmax=istart_t + 10)
                    dyn_new.trim_edges()
                    if dyn_new.tobs <= 5 or dyn_new.bw < bw_low*0.9:
                        continue
                    if zap:
                        dyn_new.zap()
                    if linear:
                        dyn_new.refill(linear=True)
                    else:
                        dyn_new.refill(linear=False)
                    dyn_new.get_acf_tilt()
                    dyn_new.get_scint_params(method='acf2d_approx',
                                             flux_estimate=True)
                    dyn_new.plot_dyn()
                    dyn_new.plot_acf(fit=True)
                    write_results(outfile, dyn=dyn_new)
                except Exception as e:
                    print(e)
                    continue

##############################################################################
# Plotting the data for the single analysis
if model:

    results_dir = outdir
    params = read_results(outfile)

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
    bw = np.array(bw[sort_ind]).squeeze()

    total_length = len(mjd)

    """
    Do corrections!
    """

    indicies = np.argwhere((tauerr < 0.3*tau) * (dnuerr < 0.3*dnu))

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
    bw = bw[indicies].squeeze()

    new_length = len(mjd)
    percentage_cut = round((total_length - new_length)/total_length * 100, 1)
    print()
    print("Filtered data: " + str(percentage_cut) + '%')
    print()

    # Make MJD centre of observation, instead of start
    mjd = mjd + tobs/86400/2

    # Form Viss from the data
    Aiss = 2.78*10**4  # thin screen, table 2 of Cordes & Rickett (1998)
    D = 1  # kpc
    ind_low = np.argwhere((freq < 1100))

    viss, visserr = scint_velocity(None, dnu, tau, freq, dnuerr,
                                   tauerr, a=Aiss)

    # Plotting Begins here #

    # fig = plt.figure(figsize=(15, 10))
    # plt.errorbar(dnu_est, dnu, yerr=dnuerr, fmt='o', alpha=0.8,
    #              label='dnu>df')
    # ax = fig.add_subplot(1, 1, 1)
    # inds = np.argwhere((dnu < df))
    # plt.errorbar(dnu_est[inds], dnu[inds], yerr=dnuerr[inds].squeeze(),
    #              fmt='o', alpha=0.8, label='dnu<df')
    # plt.grid()
    # plt.xlim([0, 1])
    # plt.ylim([0, 1.5])
    # plt.plot([0, 1], [0, 1], 'k', zorder=3)
    # plt.ylabel('Measured scint bandwidth (MHz)')
    # plt.xlabel('Estimated scint bandwidth (MHz)')
    # ax.legend(fontsize='xx-small')
    # plt.savefig(plotdir + str(psrname) + "_EsitmatedScintBandwidth.png")
    # plt.show()

    # plt.figure(figsize=(12, 6))
    # plt.errorbar(freq, dnu, yerr=dnuerr, fmt='o', alpha=0.8)
    # inds = np.argwhere((dnu < df))
    # plt.errorbar(freq[inds], dnu[inds], yerr=dnuerr[inds].squeeze(), fmt='o',
    #              alpha=0.8)
    # plt.yscale('log')
    # plt.xscale('log')
    # plt.grid()
    # plt.ylim([0, 1.5])
    # plt.ylabel('Measured scint bandwidth (MHz)')
    # plt.xlabel('Observing frequency (MHz)')
    # plt.savefig(plotdir + str(psrname) + "_ScintBandwidth.png")
    # plt.show()

    # plt.figure(figsize=(12, 6))
    # inds = np.argwhere((freq > 1300))
    # plt.errorbar(mjd[inds], viss[inds], yerr=visserr[inds].squeeze()*15,
    #              fmt='o')
    # inds = np.argwhere((freq < 1300))
    # plt.errorbar(mjd[inds], viss[inds], yerr=visserr[inds].squeeze()*15,
    #              fmt='o')
    # inds = np.argwhere((dnu < 0.5*df))
    # plt.xlabel('MJD')
    # plt.ylabel('Viss (km/s)')
    # plt.title(psrname)
    # plt.savefig(plotdir + str(psrname) + "_Viss_MJD.png")
    # plt.show()

    # mjd_annual = mjd % 365.2425
    # plt.errorbar(mjd_annual, viss, yerr=visserr, fmt='o', )
    # plt.xlabel('Annual phase (arb)')
    # plt.ylabel('Viss')
    # plt.title(psrname)
    # plt.savefig(plotdir + str(psrname) + "_Viss_AnnualPhase.png")
    # plt.show()

    print('Getting SSB delays')
    ssb_delays = get_ssb_delay(mjd, pars['RAJ'], pars['DECJ'])
    mjd += np.divide(ssb_delays, 86400)  # add ssb delay

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

    ind_high = np.argwhere(freq > 1600)
    ind_mid = np.argwhere((freq > 1100) * (freq < 1600))
    ind_low = np.argwhere((freq < 1100) * (freq > 950))

    Font = 35
    Size = 80*np.pi  # Determines the size of the datapoints used
    font = {'size': 32}
    matplotlib.rc('font', **font)

    # fig = plt.figure(figsize=(15, 10))
    # ax = fig.add_subplot(1, 1, 1)
    # plt.scatter(phase[ind_high].flatten(), viss[ind_high].flatten(),
    #             c='red', alpha=0.6, label=r'$f_{c} =$ 1675', s=Size)
    # plt.errorbar(phase[ind_high].flatten(), viss[ind_high].flatten(),
    #              yerr=visserr[ind_high].flatten(), fmt=' ', ecolor='red',
    #              alpha=0.4, elinewidth=5)
    # plt.scatter(phase[ind_mid].flatten(), viss[ind_mid].flatten(),
    #             c='blue', alpha=0.6, label=r'$f_{c} =$ 1400', s=Size)
    # plt.errorbar(phase[ind_mid].flatten(), viss[ind_mid].flatten(),
    #              yerr=visserr[ind_mid].flatten(), fmt=' ', ecolor='blue',
    #              alpha=0.4, elinewidth=5)
    # plt.scatter(phase[ind_low].flatten(), viss[ind_low].flatten(),
    #             c='green', alpha=0.6, label=r'$f_{c} =$ 1025', s=Size)
    # plt.errorbar(phase[ind_low].flatten(), viss[ind_low].flatten(),
    #              yerr=visserr[ind_low].flatten(), fmt=' ', ecolor='green',
    #              alpha=0.4, elinewidth=5)
    # plt.xlabel('Orbital phase (degrees)')
    # plt.ylabel('Viss (km/s)')
    # plt.xlim(0, 360)
    # plt.title(psrname + ' Scintillation velocity')
    # ax.legend(fontsize='xx-small')
    # plt.savefig(plotdir + str(psrname) + "_Viss_OrbitalPhase.png")
    # plt.show()
    # plt.close()

    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(1, 1, 1)
    plt.scatter(freq[ind_high].flatten(), dnu[ind_high].flatten(),
                c='red', alpha=0.6, label=r'$f_{c} =$ 1675', s=Size)
    plt.errorbar(freq[ind_high].flatten(), dnu[ind_high].flatten(),
                 yerr=dnuerr[ind_high].flatten(), fmt=' ', ecolor='red',
                 alpha=0.4, elinewidth=5)
    plt.scatter(freq[ind_mid].flatten(), dnu[ind_mid].flatten(),
                c='blue', alpha=0.6, label=r'$f_{c} =$ 1400', s=Size)
    plt.errorbar(freq[ind_mid].flatten(), dnu[ind_mid].flatten(),
                 yerr=dnuerr[ind_mid].flatten(), fmt=' ', ecolor='blue',
                 alpha=0.4, elinewidth=5)
    plt.scatter(freq[ind_low].flatten(), dnu[ind_low].flatten(),
                c='green', alpha=0.6, label=r'$f_{c} =$ 1025', s=Size)
    plt.errorbar(freq[ind_low].flatten(), dnu[ind_low].flatten(),
                 yerr=dnuerr[ind_low].flatten(), fmt=' ', ecolor='green',
                 alpha=0.4, elinewidth=5)
    plt.plot([np.min(freq), np.max(freq)], [df[0], df[0]], c='C0')
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Scintillation Bandwidth (MHz)')
    # plt.xlim(0, 360)
    plt.title(psrname + ' Scintillation Bandwidth')
    ax.legend(fontsize='xx-small')
    plt.savefig(plotdir + str(psrname) + "Dnu_Frequency_SingleAnalysis.png")
    plt.show()
    plt.close()
