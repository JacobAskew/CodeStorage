#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 22:52:35 2020

@author: jacobaskew
"""
from scintools.dynspec import Dynspec
from scintools.dynspec import BasicDyn
from scintools.scint_sim import Simulation
from scintools.scint_utils import write_results, read_results, read_par, \
        float_array_from_dict, get_ssb_delay, get_earth_velocity, \
        get_true_anomaly, is_valid, svd_model, interp_nan_2d, \
        centres_to_edges

import glob
import numpy as np
# import random

import math
# from astropy.time import Time
import matplotlib.pyplot as plt
import matplotlib
# import scipy.signal as sig
from copy import deepcopy as cp
from lmfit import Parameters, minimize
import os
from scipy.optimize import curve_fit
from scipy.interpolate import griddata
from scipy import interpolate
from scipy.interpolate import interp2d


##############################################################################


def powerlaw(params, xdata, ydata, weights, reffreq, amp):

    if weights is None:
        weights = np.ones(np.shape(xdata))

    if ydata is None:
        ydata = np.zeros(np.shape(xdata))

    parvals = params.valuesdict()
    if amp is None:
        amp = parvals['amp']
    alpha = parvals['alpha']

    func = amp*(xdata/reffreq)**(alpha)

    return (ydata - func) * weights


def func1(xdata, c, a):
    return a*(xdata/1000)**c


def func2(xdata, c, a):
    return a*(xdata/1300)**c


def func3(xdata, c, a):
    return a*(xdata/1600)**c


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
f = 1100*10**6
wavelength = c/f
k = (2*np.pi)/wavelength
rf = np.sqrt(z/k)
# mb2 = 0.773*(rf) * 100
dnu_c = 0.2
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
if sim:
    # Creating simulated data and a dynamic spectrum from the data
    sim = Simulation(mb2=mb2, rf=10, ds=0.01, alpha=5/3, ar=1, psi=0,
                     inner=0.001, ns=1360, nf=NF,
                     dlam=0.9565217391304348, lamsteps=False, seed=64, nx=None,
                     ny=None, dx=None, dy=None, plot=False, verbose=False,
                     freq=1150, dt=8, mjd=50000, nsub=None, efield=False,
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
###############################################################################
psrname = 'J0737-3039A'
pulsar = '0737-3039A'
resolutiondir = str(resolution) + '/'
wd0 = '/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/'
eclipsefile = wd0+'Datafiles/Eclipse_mjd.txt'
wd = wd0+'Simulation/'
datadir = wd + 'Dynspec/'
outdir = wd + 'DataFiles/' + resolutiondir
par_dir = '/Users/jacobaskew/Desktop/Swinburne_Daniel/ParFiles/'
spectradir = wd + 'SpectraPlots/'
spectrabindir = wd + 'SpectraPlotsBin/'
ACFdir = wd + 'ACFPlots/'
ACFbindir = wd + 'ACFPlotsBin/'
plotdir = wd + 'Plots/'
try:
    os.mkdir(outdir)
except OSError as error:
    print(error)

freq_bin = 30
time_bin = 30

outfile = outdir+'freq'+str(freq_bin)+'_time'+str(time_bin) + \
    '_ScintillationResults_Sim'+str(resolution)+'.txt'
if os.path.exists(outfile) and measure:
    os.remove(outfile)

zap = False
median = False
linear = False

if measure:
    dyn_crop = cp(dyn_initial)

    # Here is another attempt at copying the current method for sim data
    Fmax = np.max(dyn_crop.freqs)
    Fmin = np.min(dyn_crop.freqs)
    f_min_init = Fmax - freq_bin
    f_max = Fmax
    f_init = int((f_max + f_min_init)/2)
    bw_init = int(f_max - f_min_init)

    # I want to create a for loop that looks at each chunk of frequency
    # until it hits the bottom.
    fail_counter = 0
    good_counter = 0
    counter_nodynspec = 0
    for istart_t in range(0, int(dyn_crop.tobs/60), int(time_bin)):
        try:
            os.mkdir(spectradir+str(round(istart_t, 1))+'/')
        except OSError as error:
            print(error)
        try:
            os.mkdir(spectrabindir+str(round(istart_t, 1))+'/')
        except OSError as error:
            print(error)
        try:
            os.mkdir(ACFdir+str(round(istart_t, 1))+'/')
        except OSError as error:
            print(error)
        try:
            os.mkdir(ACFbindir+str(round(istart_t, 1))+'/')
        except OSError as error:
            print(error)
        t_max_new = istart_t + time_bin
        dyn_new_time = cp(dyn_crop)
        dyn_new_time.crop_dyn(tmin=istart_t, tmax=t_max_new)
        freq_bins = freq_bin
        for istart_f in range(int(np.max(dyn_crop.freqs)), 0,
                              -int(freq_bins)):
            try:
                f_min_new = istart_f - freq_bin
                f_max_new = istart_f
                if f_min_new < Fmin:
                    continue
                dyn_new_freq = cp(dyn_new_time)
                dyn_new_freq.crop_dyn(fmin=f_min_new, fmax=f_max_new)
                if zap:
                    dyn_new_freq.zap()
                if linear:
                    dyn_new_freq.refill(method='linear')
                elif median:
                    dyn_new_freq.refill(method='median')
                else:
                    dyn_new_freq.refill()
                dyn_new_freq.get_acf_tilt(plot=False, display=False)
                dyn_new_freq.get_scint_params(filename=str(ACFdir) +
                                              str(round(istart_t, 1)) +
                                              '/ACF_chunk_' +
                                              str(round(dyn_new_freq.freq,
                                                2))+'.pdf',
                                              method='acf2d_approx',
                                              plot=True, display=True)
                write_results(outfile, dyn=dyn_new_freq)
                good_counter += 1
                dyn_new_freq.plot_dyn(filename=str(spectradir) +
                                      str(round(istart_t, 1)) +
                                      '/dynspec_chunk_' +
                                      str(round(dyn_new_freq.freq,
                                                2))+'.pdf',
                                      dpi=400)
                # dyn_new_freq.plot_acf(, crop=True,
                #                       dpi=400)
                if dyn_new_freq.tauerr < 10*dyn_new_freq.tau or \
                        dyn_new_freq.dnuerr < 10*dyn_new_freq.dnu:
                    fail_counter += 1
                    try:
                        dyn_new_freq.plot_dyn(filename=str(spectrabindir) +
                                              str(round(istart_t, 1)) +
                                              '/dynspec_chunk_' +
                                              str(round(dyn_new_freq.freq,
                                                        2)) +
                                              '.pdf', dpi=400)
                        dyn_new_freq.get_scint_params(filename=str(ACFdir) +
                                                      str(round(istart_t, 1)) +
                                                      '/ACF_chunk_' +
                                                      str(round(dyn_new_freq.freq,
                                                        2))+'.pdf',
                                                      method='acf2d_approx',
                                                      plot=True, display=True)
                    except Exception as e:
                        print(e)
                        counter_nodynspec += 1
    
            except Exception as e:
                print(e)
                fail_counter += 1
                try:
                    dyn_new_freq.plot_dyn(filename=str(spectrabindir) +
                                          str(round(istart_t, 1)) +
                                          '/dynspec_chunk_' +
                                          str(round(dyn_new_freq.freq, 2))
                                          + '.pdf', dpi=400)
                    dyn_new_freq.get_scint_params(filename=str(ACFdir) +
                                                  str(round(istart_t, 1)) +
                                                  '/ACF_chunk_' +
                                                  str(round(dyn_new_freq.freq,
                                                    2))+'.pdf',
                                                  method='acf2d_approx',
                                                  plot=True, display=True)
                except Exception as e:
                    print(e)
                    counter_nodynspec += 1
                continue

###############################################################################
# Here I simply want to load the data and prepare basic plotting
if load_data:
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

if compare:
    outfile='/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/New/DataFiles/J0737-3039A_freq30_time30_ScintillationResults_UHF.txt'
    outdir ='/Users/jacobaskew/Desktop/'
    results_dir = outdir
    params = read_results(outfile)

    pars = read_par(str(par_dir) + str(psrname) + '.par')

    # Read in arrays
    mjd_UHF = float_array_from_dict(params, 'mjd')  # MJD for observation start
    df_UHF = float_array_from_dict(params, 'df')  # channel bandwidth
    dnu_UHF = float_array_from_dict(params, 'dnu')  # scint bandwidth
    dnu_est_UHF = float_array_from_dict(params, 'dnu_est')  # estimated bandwidth
    dnuerr_UHF = float_array_from_dict(params, 'dnuerr')
    tau_UHF = float_array_from_dict(params, 'tau')
    tauerr_UHF = float_array_from_dict(params, 'tauerr')
    freq_UHF = float_array_from_dict(params, 'freq')
    bw_UHF = float_array_from_dict(params, 'bw')
    # scintle_num = float_array_from_dict(params, 'scintle_num')
    tobs_UHF = float_array_from_dict(params, 'tobs')  # tobs in second
    rcvrs_UHF = np.array([rcvr[0] for rcvr in params['name']])

    # Sort by MJD
    sort_ind = np.argsort(mjd_UHF)

    df_UHF = np.array(df_UHF[sort_ind]).squeeze()
    dnu_UHF = np.array(dnu_UHF[sort_ind]).squeeze()
    dnu_est_UHF = np.array(dnu_est_UHF[sort_ind]).squeeze()
    dnuerr_UHF = np.array(dnuerr_UHF[sort_ind]).squeeze()
    tau_UHF = np.array(tau_UHF[sort_ind]).squeeze()
    tauerr_UHF = np.array(tauerr_UHF[sort_ind]).squeeze()
    mjd_UHF = np.array(mjd_UHF[sort_ind]).squeeze()
    rcvrs_UHF = np.array(rcvrs_UHF[sort_ind]).squeeze()
    freq_UHF = np.array(freq_UHF[sort_ind]).squeeze()
    tobs_UHF = np.array(tobs_UHF[sort_ind]).squeeze()
    # scintle_num = np.array(scintle_num[sort_ind]).squeeze()
    bw_UHF = np.array(bw_UHF[sort_ind]).squeeze()

    # Used to filter the data
    indicies = np.argwhere((tauerr_UHF < 10*tau_UHF) * (dnuerr_UHF < 10*dnu_UHF))

    df_UHF = df_UHF[indicies].squeeze()
    dnu_UHF = dnu_UHF[indicies].squeeze()
    dnu_est_UHF = dnu_est_UHF[indicies].squeeze()
    dnuerr_UHF = dnuerr_UHF[indicies].squeeze()
    tau_UHF = tau_UHF[indicies].squeeze()
    tauerr_UHF = tauerr_UHF[indicies].squeeze()
    mjd_UHF = mjd_UHF[indicies].squeeze()
    rcvrs_UHF = rcvrs_UHF[indicies].squeeze()
    freq_UHF = freq_UHF[indicies].squeeze()
    tobs_UHF = tobs_UHF[indicies].squeeze()
    # scintle_num = scintle_num[indicies].squeeze()
    bw_UHF = bw_UHF[indicies].squeeze()

    mjd_annual = mjd_UHF % 365.2425
    print('Getting SSB delays')
    ssb_delays = get_ssb_delay(mjd_UHF, pars['RAJ'], pars['DECJ'])
    mjd_UHF += np.divide(ssb_delays, 86400)  # add ssb delay

    dnu_std = np.std(dnu)

    """
    Model Viss
    """
    print('Getting Earth velocity')
    vearth_ra, vearth_dec = get_earth_velocity(mjd_UHF, pars['RAJ'], pars['DECJ'])
    print('Getting true anomaly')
    pars['PBDOT'] *= 10**-12  # add in factor dropped by tempo
    U = get_true_anomaly(mjd_UHF, pars)

    true_anomaly = U.squeeze()
    vearth_ra = vearth_ra.squeeze()
    vearth_dec = vearth_dec.squeeze()

    om = pars['OM'] + pars['OMDOT']*(mjd_UHF - pars['T0'])/365.2425
    # compute orbital phase
    phase = U*180/np.pi + om
    phase[phase > 360] = phase[phase > 360] - 360

if plotting:

    Font = 30
    Size = 80*np.pi  # Determines the size of the datapoints used
    font = {'size': 28}
    matplotlib.rc('font', **font)

    fig = plt.figure(figsize=(20, 10))
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    ax = fig.add_subplot(1, 1, 1)
    #
    plt.scatter(freq, dnu, s=Size, alpha=0.6, c='C0', label='Simulated Data')
    plt.errorbar(freq, dnu, yerr=dnuerr, fmt=' ', ecolor='k',
                 elinewidth=2, capsize=3, alpha=0.4)
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
    #
    plt.scatter(freq_UHF, dnu_UHF, s=Size, alpha=0.6, c='C0', marker='s',
                label='UHF Data')
    plt.errorbar(freq_UHF, dnu_UHF, yerr=dnuerr_UHF, fmt=' ', ecolor='k',
                 elinewidth=2, capsize=3, alpha=0.4)
    popt, pcov = curve_fit(func1, freq_UHF, dnu_UHF)
    perr = np.sqrt(np.diag(pcov))
    plt.plot(xdata, func1(xdata, *popt),
             'C3', label=r'$f_c=1000$, $\alpha$='+str(round(popt[0], 2)) +
             r'$\pm$'+str(round(perr[0], 2)))
    plt.fill_between(xdata.flatten(),
                     func1(xdata, *[popt[0]+perr[0],
                                                     popt[1]]).flatten(),
                     func1(xdata, *[popt[0]-perr[0],
                                                     popt[1]]).flatten(),
                     alpha=0.5, color='C3')
    #
    theory_dnu = popt[1]*(xdata/1000)**(4)
    plt.plot(xdata, theory_dnu, c='k', linewidth=2)
    #
    # popt, pcov = curve_fit(func2, freq, dnu)    # perr = np.sqrt(np.diag(pcov))
    # plt.plot(freq[np.argsort(freq)], func2(freq[np.argsort(freq)], *popt),
    #          'C3', label=r'$f_c=1300$, $\alpha$='+str(round(popt[0], 2)) +
    #          r'$\pm$'+str(round(perr[0], 2)))
    # plt.fill_between(freq[np.argsort(freq)].flatten(),
    #                  func2(freq[np.argsort(freq)], *[popt[0]+perr[0],
    #                                                  popt[1]]).flatten(),
    #                  func2(freq[np.argsort(freq)], *[popt[0]-perr[0],
    #                                                  popt[1]]).flatten(),
    #                  alpha=0.5, color='C3')
    # #
    # popt, pcov = curve_fit(func3, freq, dnu)
    # perr = np.sqrt(np.diag(pcov))
    # plt.plot(freq[np.argsort(freq)], func3(freq[np.argsort(freq)], *popt),
    #          'C4', label=r'$f_c=1600$, $\alpha$='+str(round(popt[0], 2)) +
    #          r'$\pm$'+str(round(perr[0], 2)))
    # plt.fill_between(freq[np.argsort(freq)].flatten(),
    #                  func3(freq[np.argsort(freq)], *[popt[0]+perr[0],
    #                                                  popt[1]]).flatten(),
    #                  func3(freq[np.argsort(freq)], *[popt[0]-perr[0],
    #                                                  popt[1]]).flatten(),
    #                  alpha=0.5, color='C4')
    # #
    plt.plot(xl, (df[0], df[0]), color='C2')
    plt.plot(xl, (df_UHF[0], df_UHF[0]), color='C2', linestyle='dashed')
    # plt.plot(freqs, theory_dnu, color='k', linewidth=4)
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Scintillation Bandwidth (MHz)')
    plt.title('Simulated Resolution '+str(resolution))
    plt.xlim(xl)
    ax.legend()
    # plt.ylim(0, 0.2)
    plt.savefig("/Users/jacobaskew/Desktop/SimulatedBandwidth_res" +
                str(resolution)+".pdf")
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.savefig("/Users/jacobaskew/Desktop/SimulatedBandwidth_res_log" +
                str(resolution)+".pdf")
    plt.show()
    plt.close()


###############################################################################
# Here is my attempt at measuring the scintillation bandwidth at
# different frequencies
# if resolution == 128:
#     if measure:
#         dyn_128 = cp(dyn_initial)
#     dyn = cp(dyn_128)
# if resolution == 1280:
#     if measure:
#         dyn_1280 = cp(dyn_initial)
#     dyn = cp(dyn_1280)
# if resolution == 12800:
#     if measure:
#         dyn_12800 = cp(dyn_initial)
#     dyn = cp(dyn_12800)

# dyn = cp(dyn_initial)
# freq_step_size = 40
# freq_bins = int(round(dyn_initial.bw, -1) / freq_step_size)
# dyn_freq_max = np.max(dyn.freqs)
# start_freq = dyn_freq_max - freq_step_size
# dnu_acf2d_approx_list_pre = []
# dnuerr_acf2d_approx_list_pre = []
# dnu_acf2d_list_pre = []
# dnuerr_acf2d_list_pre = []
# dnu_acf1d_list_pre = []
# dnuerr_acf1d_list_pre = []
# mjd_acf2d_approx_list_pre = []
# freq_acf2d_approx_list_pre = []
# mjd_acf2d_list_pre = []
# freq_acf2d_list_pre = []
# mjd_acf1d_list_pre = []
# freq_acf1d_list_pre = []
# dyn_freq_max = np.max(dyn.freqs)
# for i in range(0, freq_bins):
#     Fmin = round(math.floor(start_freq - (i * freq_step_size)), -1)
#     Fmax = round(math.floor(dyn_freq_max - (i * freq_step_size)), -1)
#     try:
#         dyn = cp(dyn_initial)
#         dyn.crop_dyn(fmin=Fmin, fmax=Fmax)
#         dyn.refill(linear=False)
#         dyn.get_acf_tilt(plot=False, display=False)
#         dyn.get_scint_params(method='acf2d_approx',
#                              flux_estimate=True,
#                              plot=False, display=False)
#         write_results(outfile, dyn=dyn)
#         freq_acf2d_approx_list_pre.append(dyn.freq)
#         dnu_acf2d_approx_list_pre.append(dyn.dnu)
#         dnuerr_acf2d_approx_list_pre.append(dyn.dnuerr)
#         mjd_acf2d_approx_list_pre.append(dyn.mjd)
#     except Exception as e:
#         print("BAD", e)
# freq_acf2d_approx_list_pre = np.asarray(freq_acf2d_approx_list_pre)
# dnu_acf2d_approx_list_pre = np.asarray(dnu_acf2d_approx_list_pre)
# dnuerr_acf2d_approx_list_pre = np.asarray(dnuerr_acf2d_approx_list_pre)
# mjd_acf2d_approx_list_pre = np.asarray(mjd_acf2d_approx_list_pre)
# for i in range(0, freq_bins):
#     Fmin = round(math.floor(start_freq - (i * freq_step_size)), -1)
#     Fmax = round(math.floor(dyn_freq_max - (i * freq_step_size)), -1)
#     try:
#         dyn = cp(dyn_initial)
#         dyn.crop_dyn(fmin=Fmin, fmax=Fmax)
#         dyn.refill(linear=False)
#         dyn.get_acf_tilt(plot=False, display=False)
#         dyn.get_scint_params(method='acf2d',
#                              flux_estimate=True,
#                              plot=False, display=False)
#         write_results(outfile, dyn=dyn)
#         freq_acf2d_list_pre.append(dyn.freq)
#         dnu_acf2d_list_pre.append(dyn.dnu)
#         dnuerr_acf2d_list_pre.append(dyn.dnuerr)
#         mjd_acf2d_list_pre.append(dyn.mjd)
#     except Exception as e:
#         print("BAD", e)
# freq_acf2d_list_pre = np.asarray(freq_acf2d_list_pre)
# dnu_acf2d_list_pre = np.asarray(dnu_acf2d_list_pre)
# dnuerr_acf2d_list_pre = np.asarray(dnuerr_acf2d_list_pre)
# mjd_acf2d_list_pre = np.asarray(mjd_acf2d_list_pre)
# for i in range(0, freq_bins):
#     Fmin = round(math.floor(start_freq - (i * freq_step_size)), -1)
#     Fmax = round(math.floor(dyn_freq_max - (i * freq_step_size)), -1)
#     try:
#         dyn = cp(dyn_initial)
#         dyn.crop_dyn(fmin=Fmin, fmax=Fmax)
#         dyn.refill(linear=False)
#         dyn.get_acf_tilt(plot=False, display=False)
#         dyn.get_scint_params(method='acf1d',
#                              flux_estimate=True,
#                              plot=False, display=False)
#         write_results(outfile, dyn=dyn)
#         freq_acf1d_list_pre.append(dyn.freq)
#         dnu_acf1d_list_pre.append(dyn.dnu)
#         dnuerr_acf1d_list_pre.append(dyn.dnuerr)
#         mjd_acf1d_list_pre.append(dyn.mjd)
#     except Exception as e:
#         print("BAD", e)
# freq_acf1d_list_pre = np.asarray(freq_acf1d_list_pre)
# dnu_acf1d_list_pre = np.asarray(dnu_acf1d_list_pre)
# dnuerr_acf1d_list_pre = np.asarray(dnuerr_acf1d_list_pre)
# mjd_acf1d_list_pre = np.asarray(mjd_acf1d_list_pre)
# dnu_acf2d_approx_list_pre
# indicies = np.argwhere((dnuerr_list_pre < 0.5*dnu_list_pre) * (dnu_list_pre < 100))

# dnu_list = dnu_list_pre[indicies].squeeze()
# dnuerr_list = dnuerr_list_pre[indicies].squeeze()
# freq_list = freq_list_pre[indicies].squeeze()
# mjd_list = mjd_list_pre[indicies].squeeze()

# indicies = np.argwhere((dnuerr_list_pre < 0.5*dnu_list_pre) * (dnu_list_pre < 100))

# dnu_list = dnu_list_pre[indicies].squeeze()
# dnuerr_list = dnuerr_list_pre[indicies].squeeze()
# freq_list = freq_list_pre[indicies].squeeze()
# mjd_list = mjd_list_pre[indicies].squeeze()
# indicies = np.argwhere((dnuerr_list_pre < 0.5*dnu_list_pre) * (dnu_list_pre < 100))

# dnu_list = dnu_list_pre[indicies].squeeze()
# dnuerr_list = dnuerr_list_pre[indicies].squeeze()
# freq_list = freq_list_pre[indicies].squeeze()
# mjd_list = mjd_list_pre[indicies].squeeze()

# desktopdir = '/Users/jacobaskew/Desktop/'
# HighFreqSpectradir = desktopdir
# par_dir = '/Users/jacobaskew/Desktop/Swinburne_Daniel/ParFiles/'
# Dnudir = '/Users/jacobaskew/Desktop/'
# psrname = 'J0737-3039A'
# pulsar = '0737-3039A'
# zap = False
# linear = False

# Font = 30
# Size = 80*np.pi  # Determines the size of the datapoints used
# font = {'size': 28}
# matplotlib.rc('font', **font)

# fig = plt.figure(figsize=(20, 10))
# fig.subplots_adjust(hspace=0.5, wspace=0.5)
# ax = fig.add_subplot(1, 1, 1)
# plt.scatter(freq_acf2d_approx_list_pre, dnu_acf2d_approx_list_pre, c='C0',
#             s=Size, alpha=0.6, label='acf2d_approx')
# plt.errorbar(freq_acf2d_approx_list_pre, dnu_acf2d_approx_list_pre,
#              yerr=dnuerr_acf2d_approx_list_pre, fmt=' ', ecolor='k',
#              elinewidth=2, capsize=3, alpha=0.55)
# # plt.scatter(freq_acf2d_approx_list_pre, dnu_acf2d_list_pre, c='C2', s=Size,
# #             alpha=0.6)
# # plt.errorbar(freq_acf2d_list_pre, dnu_acf2d_list_pre,
# #              yerr=dnuerr_acf2d_list_pre, fmt=' ', ecolor='k',
# #              elinewidth=2, capsize=3, alpha=0.55)
# plt.scatter(freq_acf1d_list_pre, dnu_acf1d_list_pre, c='C3', s=Size,
#             alpha=0.6, label='acf1d')
# plt.errorbar(freq_acf1d_list_pre, dnu_acf1d_list_pre,
#              yerr=dnuerr_acf1d_list_pre, fmt=' ', ecolor='k',
#              elinewidth=2, capsize=3, alpha=0.55)
# ax.legend(fontsize='xx-small')
# xl = plt.xlim()
# freq_range = np.linspace(xl[0], xl[1], 100)
# dnu_estimated_individual = (freq_range/1400)**4 * dnu_c
# mean_dnu_acf2d_approx = \
#     np.mean(dnu_acf2d_approx_list_pre[np.argwhere(freq_acf2d_approx_list_pre >
#                                                   1500)])
# mean_freq_acf2d_approx = \
#     np.mean(freq_acf2d_approx_list_pre[
#         np.argwhere(freq_acf2d_approx_list_pre > 1500)])
# parameters = Parameters()
# parameters.add('amp', value=dnu_c, vary=True, min=0, max=2)
# parameters.add('alpha', value=4, vary=True, min=1, max=5)
# results = minimize(powerlaw, parameters,
#                    args=(freq_acf2d_approx_list_pre,
#                          dnu_acf2d_approx_list_pre,
#                          dnuerr_acf2d_approx_list_pre,
#                          mean_freq_acf2d_approx,
#                          mean_dnu_acf2d_approx),
#                    method='emcee', steps=10000, burn=1000)
# Slope = results.params['alpha'].value
# Slopeerr = results.params['alpha'].stderr
# dnu_estimated_average = (freq_range/mean_freq_acf2d_approx)**Slope * \
#     mean_dnu_acf2d_approx
# dnu_estimated_poserr = \
#     (freq_range/mean_freq_acf2d_approx)**(Slope + Slopeerr) * \
#     mean_dnu_acf2d_approx
# dnu_estimated_negerr = \
#     (freq_range/mean_freq_acf2d_approx)**(Slope - Slopeerr) * \
#     mean_dnu_acf2d_approx
# plt.plot(freq_range, dnu_estimated_average, linewidth=4,
#          c='C3', alpha=0.4, label='Data Fit')
# plt.fill_between(freq_range, dnu_estimated_negerr,
#                  dnu_estimated_poserr, alpha=0.2, color='C3')
# plt.plot(freq_range, dnu_estimated_individual, c='k',
#          linewidth=4, alpha=0.4, label='Model')
# plt.plot(xl, (dyn.df, dyn.df), linewidth=4, color='C2')
# plt.xlim(xl)
# # plt.ylim(-1, 10)
# plt.xlabel('Frequency (MHz)', fontsize=Font, ha='center')
# plt.ylabel('Scintillation Bandwidth (MHz)')
# plt.title(psrname + ' Scintillation Bandwidth')
# plt.savefig("/Users/jacobaskew/Desktop/Simulated_Dnu_freq_phase_" +
#             str(resolution) + ".png")
# plt.show()
# plt.close()


##############################################################################
# This is the same method as with the real observations where you take chunks
# of frequency and time. Above is the method with only chunks of frequency.

# dyn = cp(dyn_initial)
# freq_step_size = 80
# time_step_size = 10
# freq_bins = int(round(dyn_initial.bw, -1) / freq_step_size)
# time_bins = int(round(np.max(dyn_initial.times / 60), -1) / time_step_size)
# start_freq = round(np.max(dyn_initial.freqs), -1) - freq_step_size
# start_time = 0
# # dyn_initial.crop_dyn(fmin=1840, fmax=1880, tmin=0, tmax=10)

# # dyn_initial.plot_dyn(filename='/Users/jacobaskew/Desktop/SimSpectra.png')

# # dyn_initial.get_acf_tilt(plot=False, display=False)

# # dyn_initial.get_scint_params(method='acf2d_approx',
# #                              flux_estimate=True,
# #                              plot=False, display=False)
# # write_results(outfile, dyn=dyn_initial)
# freq_list_pre = []
# dnu_list_pre = []
# dnuerr_list_pre = []
# mjd_list_pre = []
# test = 0
# dyn_freq_max = np.max(dyn.freqs)
# for i in range(0, freq_bins):
#     freq_step = start_freq - (i * freq_step_size)
#     for ii in range(0, time_bins):
#         time_step = start_time + (ii * time_step_size)
#         try:
#             dyn = cp(dyn_initial)
#             if i == 0:
#                 dyn.crop_dyn(fmin=start_freq,
#                              fmax=dyn_freq_max,
#                              tmin=time_step,
#                              tmax=time_step+time_step_size)
#             else:
#                 dyn.crop_dyn(fmin=freq_step,
#                              fmax=freq_step+freq_step_size,
#                              tmin=time_step,
#                              tmax=time_step+time_step_size)
#                 dyn.refill(linear=False)
#                 dyn.get_acf_tilt(plot=False, display=False)
#                 dyn.get_scint_params(method='acf2d_approx',
#                                      flux_estimate=True,
#                                      plot=False, display=False)
#                 write_results(outfile, dyn=dyn)
#                 freq_list_pre.append(dyn.freq)
#                 dnu_list_pre.append(dyn.dnu)
#                 dnuerr_list_pre.append(dyn.dnuerr)
#                 mjd_list_pre.append(dyn.mjd)
#                 test += 1
#         except Exception as e:
#             print("BAD", e)
# print(test)
# freq_list_pre = np.asarray(freq_list_pre)
# dnu_list_pre = np.asarray(dnu_list_pre)
# dnuerr_list_pre = np.asarray(dnuerr_list_pre)
# mjd_list_pre = np.asarray(mjd_list_pre)


# indicies = np.argwhere((dnuerr_list_pre < 0.5*dnu_list_pre) *
#                        (dnu_list_pre < 100))

# dnu_list = dnu_list_pre[indicies].squeeze()
# dnuerr_list = dnuerr_list_pre[indicies].squeeze()
# freq_list = freq_list_pre[indicies].squeeze()
# mjd_list = mjd_list_pre[indicies].squeeze()

# unique_time_list = (np.unique(mjd_list) - 50000) * 24 * 60
# time_list = (mjd_list - 50000) * 24 * 60

# # dyn.crop_dyn(tmin=0, tmax=np.inf, fmin=1300, fmax=1670)
# # dyn.plot_dyn(filename='/Users/jacobaskew/Desktop/SimSpectra.png')

# # #Remove noise
# # # Take away the median value ?
# # #Reduce signal
# # # Take away 75% of the maximum
# # #Add bonus noise on top
# # # Add the gaussian noise

# # # V = ds / dt # velocity in units of Fresnel scales per second (ds is a fraction of a Fesnel scale)
# # # lambda = 1/freq_sim  # because c=1 in the simulator, so wavelength = 1/frequency
# # # k = 2*pi/lambda  # wavenumber 
# # # eta = rf**2 * k / (2 * freq**2 * V**2)  # eta = curvature. rf = Fresnel scale

# # plotdir = '/Users/jacobaskew/Desktop/1909_Project/SimulatedArcs/'

# # mu, sigma = 0, 0.5

# # sim = Simulation(seed=64,dt=1,freq=1)
# # dyn = Dynspec(dyn=sim, process=False)
# # dyn.trim_edges()
# # # Median = np.median(dyn.dyn)
# # # Maximum = np.max(dyn.dyn)
# # # GaussianNoise = np.random.normal(mu, sigma, (256, 256))
# # # dyn.dyn[:,:] -= Median
# # # dyn.dyn[:,:] -= 0.75 * Maximum
# # # dyn.dyn[:,:] += GaussianNoise
# # dyn.trim_edges()
# # dyn.refill(linear=True)
# # dyn.calc_acf()
# # # dyn.correct_dyn()
# # dyn.calc_sspec(window_frac=0.1, prewhite=True, lamsteps=True)
# # #
# # dyn.plot_dyn(filename='/Users/jacobaskew/Desktop/BigSpectra.png')
# # dyn.plot_acf(filename='/Users/jacobaskew/Desktop/BigACF.png', fit=True)
# # dyn.fit_arc(filename='/Users/jacobaskew/Desktop/BigPlots.png', cutmid=2, startbin=5, plot=True, lamsteps=True)
# # dyn.plot_sspec(filename='/Users/jacobaskew/Desktop/CleanSecSpectra.png', maxfdop=3, plotarc=True, lamsteps=True)
# # dyn.plot_sspec(filename='/Users/jacobaskew/Desktop/RawSecSpectra.png', plotarc=False, lamsteps=True)
# # #
# # print()
# # print('\u03B7 = ' + str(dyn.betaeta) + ' +/- ' + str(dyn.betaetaerr))
# # print()
# # # print("dnu = " + str(dyn.dnu) + ' +/- ' + str(dyn.dnuerr))
# # # print("tau = " + str(dyn.tau) + ' +/- ' + str(dyn.tauerr))
# # # print()

# desktopdir = '/Users/jacobaskew/Desktop/'
# HighFreqSpectradir = desktopdir
# par_dir = '/Users/jacobaskew/Desktop/Swinburne_Daniel/ParFiles/'
# Dnudir = '/Users/jacobaskew/Desktop/'
# psrname = 'J0737-3039A'
# pulsar = '0737-3039A'
# zap = False
# linear = False

# Font = 30
# Size = 80*np.pi  # Determines the size of the datapoints used
# font = {'size': 28}
# matplotlib.rc('font', **font)
# # Dnu v Frequency v Time
# fig = plt.figure(figsize=(20, 10))
# fig.subplots_adjust(hspace=0.5, wspace=0.5)
# ax = fig.add_subplot(1, 1, 1)
# plt.scatter(freq_list, dnu_list, c='C0', s=Size, alpha=0.6)
# plt.errorbar(freq_list, dnu_list, yerr=dnuerr_list, fmt=' ', ecolor='k',
#              elinewidth=2, capsize=3, alpha=0.55)
# xl = plt.xlim()
# plt.plot(xl, (dyn.df, dyn.df), linewidth=4, color='C2')
# freq_range = np.linspace(xl[0], xl[1], 100)
# freq_upper_average = \
#     np.average(freq_list[np.argwhere(freq_list > 1600)].flatten())
# dnu_upper_average = \
#     np.average(dnu_list[np.argwhere(freq_list > 1600)].flatten(),
#                weights=dnuerr_list[np.argwhere(freq_list >
#                                                1600)].flatten())
# # Here we are calculated for the residuals of the powerlaw
# # parameters = Parameters()
# # parameters.add('amp', value=dnu_upper_average, vary=True,
# #                 min=0, max=2)
# # parameters.add('alpha', value=4, vary=True, min=1, max=5)
# # results = minimize(powerlaw, parameters,
# #                     args=(freq_list,
# #                           dnu_list,
# #                           dnuerr_list,
# #                           1400,
# #                           dnu_upper_average),
# #                     method='emcee', steps=1000, burn=100)
# # Slope = results.params['alpha'].value
# # Slopeerr = results.params['alpha'].stderr
# dnu_estimated_individual = (freq_range /
#                             freq_upper_average)**4 * dnu_upper_average
# # dnu_estimated_average = (freq_range /
# #                           freq_upper_average)**Slope * \
# #     dnu_upper_average
# # dnu_estimated_poserr = (freq_range /
# #                         freq_upper_average)**(Slope +
# #                                               Slopeerr) * \
# #     dnu_upper_average
# # dnu_estimated_negerr = (freq_range /
# #                         freq_upper_average)**(Slope -
# #                                               Slopeerr) * \
# #     dnu_upper_average
# plt.plot(freq_range, dnu_estimated_individual, c='k',
#          linewidth=4, alpha=0.4, label='Model')
# # plt.plot(freq_range, dnu_estimated_average, linewidth=4,
# #           c='C3', alpha=0.4, label='Data Fit')
# # plt.fill_between(freq_range, dnu_estimated_negerr,
# #                   dnu_estimated_poserr, alpha=0.2, color='C3')
# ax.legend(fontsize="xx-small")
# plt.grid(True, which="both", ls="-", color='0.65')
# plt.xlim(xl[0], 1710)
# plt.ylim(0, 100)
# plt.xlabel('Frequency (MHz)', fontsize=Font, ha='center')
# plt.ylabel('Scintillation Bandwidth (MHz)')
# plt.title(psrname + ' Scintillation Bandwidth')
# plt.savefig("/Users/jacobaskew/Desktop/Simulated_Dnu_freq_phase_" +
#             str(resolution) + ".png")
# plt.show()
# plt.close()

# for i in range(0, len(unique_time_list)):
#     dnu_time = dnu_list[np.argwhere(unique_time_list[i] ==
#                                     time_list)].flatten()
#     dnuerr_time = dnuerr_list[np.argwhere(unique_time_list[i] ==
#                                           time_list)].flatten()
#     freq_time = freq_list[np.argwhere(unique_time_list[i] ==
#                                       time_list)].flatten()
#     fig = plt.figure(figsize=(20, 10))
#     fig.subplots_adjust(hspace=0.5, wspace=0.5)
#     ax = fig.add_subplot(1, 1, 1)
#     plt.scatter(freq_time, dnu_time, c='C0', s=Size, alpha=0.6)
#     plt.errorbar(freq_time, dnu_time, yerr=dnuerr_time, fmt=' ', ecolor='C0',
#                  elinewidth=2, capsize=3, alpha=0.55)
#     xl = plt.xlim()
#     plt.plot(xl, (dyn.df, dyn.df), linewidth=4, color='C2')
#     freq_range = np.linspace(xl[0], xl[1], 100)
#     freq_upper_average = \
#         np.average(freq_time[np.argwhere(freq_time > 1600)].flatten())
#     dnu_upper_average = \
#         np.average(dnu_time[np.argwhere(freq_time > 1600)].flatten(),
#                    weights=1/dnuerr_time[np.argwhere(freq_time >
#                                                      1600)].flatten())
#     dnu_estimated_individual = (freq_range /
#                                 freq_upper_average)**4 * dnu_upper_average
#     plt.plot(freq_range, dnu_estimated_individual, c='k',
#              linewidth=4, alpha=0.4, label='Model')
#     ax.legend(fontsize="xx-small")
#     plt.grid(True, which="both", ls="-", color='0.65')
#     plt.xlim(xl[0], 1710)
#     plt.ylim(0, 25)
#     plt.xlabel('Frequency (MHz)', fontsize=Font, ha='center')
#     plt.ylabel('Scintillation Bandwidth (MHz)')
#     plt.title(psrname + ' Scintillation Bandwidth Chunk: ' + str(i+1))
#     plt.savefig("/Users/jacobaskew/Desktop/Simulated_Dnu_freq_phase" +
#                 str(i+1) + "_" + str(resolution) + ".png")
#     plt.show()
#     plt.close()


# # freqmin = 0
# # freqmax = round(np.max(dyn.freqs), -1)
# # time_bin_length = 10
# # time_len = int((round(dyn.tobs/60, 0) - time_bin_length))
# # # dyn.crop_dyn(fmin=freqmin, fmax=np.inf)
# # freq_bin_length = 40
# # freq_bins = math.ceil((freqmax-freqmin)/freq_bin_length)
# # if freq_bins == 0:
# #     freq_bins = 1
# # time_bins = math.floor(time_len/time_bin_length)
# # # Plotting the cropped dynspec
# # # dyn.plot_dyn(filename=str(desktopdir) +
# # #              str(dyn.name.split('.')[0]) +
# # #              '_CroppedDynspec')
# # # dyn.plot_dyn(filename=str(HighFreqSpectradir) +
# # #              str(dyn.name.split('.')[0]) +
# # #              '_CroppedDynspec')

# # for i in range(0, time_bins):
# #     time = i * time_bin_length
# #     for ii in range(0, freq_bins):
# #         freq0 = ii * freq_bin_length
# #         freq1 = (ii - 1) * freq_bin_length
# #         if ii == 0:
# #             freq1 = 0
# #         if 1630-freq0 < 0:
# #             continue
# #         elif 1630-freq1 < np.min(dyn.freqs):
# #             continue
# #         try:
# #             dyn = cp(dyn)
# #             if ii == 0:
# #                 dyn.crop_dyn(fmin=freqmax-freq0,
# #                              fmax=np.max(dyn.freqs),
# #                              tmin=time,
# #                              tmax=time_bin_length+time)

# #             else:
# #                 dyn.crop_dyn(fmin=freqmax-freq0,
# #                              fmax=freqmax-freq1,
# #                              tmin=time,
# #                              tmax=time_bin_length+time)
# #             if zap:
# #                 dyn.zap()
# #             if linear:
# #                 dyn.refill(linear=True)
# #             else:
# #                 dyn.refill(linear=False)
# #             dyn.get_acf_tilt(plot=False, display=False)
# #             dyn.get_scint_params(method='acf2d_approx',
# #                                  flux_estimate=True,
# #                                  plot=False, display=False)
# #             write_results(outfile, dyn=dyn)
# #             # write_results(outfile_group, dyn=dyn)
# #             print("Got a live one: " + str(i))
# #         except Exception as e:
# #             print(e)
# #             print("THIS FILE DIDN'T WORK")
# #             print("Tmin: " + str(time))
# #             print("Tmax: " + str(time_bin_length+time))
# #             print("Fmin: " + str(freqmax-freq0))
# #             if ii == 0:
# #                 print("Fmax: " + str(np.max(dyn.freqs)))
# #             else:
# #                 print("Fmax: " + str(freqmax-freq1))
# #             continue

# # params = read_results(outfile)
# # pars = read_par(str(par_dir) + str(psrname) + '.par')

# # # Read in arrays
# # mjd = float_array_from_dict(params, 'mjd')  # MJD for observation
# # df = float_array_from_dict(params, 'df')  # channel bandwidth
# # dnu = float_array_from_dict(params, 'dnu')  # scint bandwidth
# # dnu_est = float_array_from_dict(params, 'dnu_est')  # est bandwidth
# # dnuerr = float_array_from_dict(params, 'dnuerr')
# # tau = float_array_from_dict(params, 'tau')
# # tauerr = float_array_from_dict(params, 'tauerr')
# # freq = float_array_from_dict(params, 'freq')
# # bw = float_array_from_dict(params, 'bw')
# # scintle_num = float_array_from_dict(params, 'scintle_num')
# # tobs = float_array_from_dict(params, 'tobs')  # tobs in second
# # rcvrs = np.array([rcvr[0] for rcvr in params['name']])
# # acf_tilt = float_array_from_dict(params, 'acf_tilt')
# # acf_tilt_err = float_array_from_dict(params, 'acf_tilt_err')
# # phasegrad = float_array_from_dict(params, 'phasegrad')
# # phasegraderr = float_array_from_dict(params, 'phasegraderr')

# # # Sort by MJD
# # sort_ind = np.argsort(mjd)

# # mjd = np.array(mjd[sort_ind]).squeeze()
# # df = np.array(df[sort_ind]).squeeze()
# # dnu = np.array(dnu[sort_ind]).squeeze()
# # dnu_est = np.array(dnu_est[sort_ind]).squeeze()
# # dnuerr = np.array(dnuerr[sort_ind]).squeeze()
# # tau = np.array(tau[sort_ind]).squeeze()
# # tauerr = np.array(tauerr[sort_ind]).squeeze()
# # rcvrs = np.array(rcvrs[sort_ind]).squeeze()
# # freq = np.array(freq[sort_ind]).squeeze()
# # tobs = np.array(tobs[sort_ind]).squeeze()
# # scintle_num = np.array(scintle_num[sort_ind]).squeeze()
# # bw = np.array(bw[sort_ind]).squeeze()
# # acf_tilt = np.array(acf_tilt[sort_ind]).squeeze()
# # acf_tilt_err = np.array(acf_tilt_err[sort_ind]).squeeze()
# # phasegrad = np.array(phasegrad[sort_ind]).squeeze()
# # phasegraderr = np.array(phasegraderr[sort_ind]).squeeze()

# # # Do corrections!

# # # indicies = np.argwhere((tauerr < 0.5*tau) * (dnuerr < 0.5*dnu))

# # # df = df[indicies].squeeze()
# # # dnu = dnu[indicies].squeeze()
# # # dnu_est = dnu_est[indicies].squeeze()
# # # dnuerr = dnuerr[indicies].squeeze()
# # # tau = tau[indicies].squeeze()
# # # tauerr = tauerr[indicies].squeeze()
# # # mjd = mjd[indicies].squeeze()
# # # rcvrs = rcvrs[indicies].squeeze()
# # # freq = freq[indicies].squeeze()
# # # tobs = tobs[indicies].squeeze()
# # # scintle_num = scintle_num[indicies].squeeze()
# # # bw = bw[indicies].squeeze()
# # # acf_tilt = np.array(acf_tilt[indicies]).squeeze()
# # # acf_tilt_err = np.array(acf_tilt_err[indicies]).squeeze()
# # # phasegrad = np.array(phasegrad[indicies]).squeeze()
# # # phasegraderr = np.array(phasegraderr[indicies]).squeeze()

# # # Make MJD centre of observation, instead of start
# # mjd = mjd + tobs/86400/2
# # mjd_annual = mjd % 365.2425
# # print('Getting Earth velocity')
# # vearth_ra, vearth_dec = get_earth_velocity(mjd, pars['RAJ'],
# #                                            pars['DECJ'])
# # print('Getting true anomaly')
# # pars['PBDOT'] *= 10**-12  # add in factor dropped by tempo
# # U = get_true_anomaly(mjd, pars)
# # om = pars['OM'] + pars['OMDOT']*(mjd - pars['T0'])/365.2425
# # # compute orbital phase
# # phase = U*180/np.pi + om
# # phase[phase > 360] = phase[phase > 360] - 360

# # Font = 30
# # Size = 80*np.pi  # Determines the size of the datapoints used
# # font = {'size': 28}
# # matplotlib.rc('font', **font)
# # # Dnu v Frequency v Time
# # fig = plt.figure(figsize=(20, 10))
# # fig.subplots_adjust(hspace=0.5, wspace=0.5)
# # ax = fig.add_subplot(1, 1, 1)
# # cm = plt.cm.get_cmap('viridis')
# # z = phase
# # sc = plt.scatter(freq, dnu, c=z, cmap=cm, s=Size, alpha=0.6)
# # plt.colorbar(sc)
# # plt.errorbar(freq, dnu, yerr=dnuerr, fmt=' ', ecolor='k',
# #              elinewidth=2, capsize=3, alpha=0.55)
# # xl = plt.xlim()
# # plt.plot(xl, (df[0], df[0]), linewidth=4, color='C2')
# # freq_range = np.linspace(xl[0], xl[1], 100)
# # freq_upper_average = \
# #     np.average(freq[np.argwhere(freq > 1500)].flatten())
# # dnu_upper_average = \
# #     np.average(dnu[np.argwhere(freq > 1500)].flatten(),
# #                weights=dnuerr[np.argwhere(freq >
# #                                           1500)].flatten())
# # # Here we are calculated for the residuals of the powerlaw
# # parameters = Parameters()
# # parameters.add('amp', value=dnu_upper_average, vary=True,
# #                min=0, max=2)
# # parameters.add('alpha', value=4, vary=True, min=1, max=5)
# # results = minimize(powerlaw, parameters,
# #                    args=(freq,
# #                          dnu,
# #                          dnuerr,
# #                          1400,
# #                          dnu_upper_average),
# #                    method='emcee', steps=1000, burn=100)
# # Slope = results.params['alpha'].value
# # Slopeerr = results.params['alpha'].stderr
# # dnu_estimated_individual = (freq_range /
# #                             freq_upper_average)**4 * \
# #     dnu_upper_average
# # dnu_estimated_average = (freq_range /
# #                          freq_upper_average)**Slope * \
# #     dnu_upper_average
# # dnu_estimated_poserr = (freq_range /
# #                         freq_upper_average)**(Slope +
# #                                               Slopeerr) * \
# #     dnu_upper_average
# # dnu_estimated_negerr = (freq_range /
# #                         freq_upper_average)**(Slope -
# #                                               Slopeerr) * \
# #     dnu_upper_average
# # plt.plot(freq_range, dnu_estimated_individual, c='k',
# #          linewidth=4, alpha=0.4, label='Model')
# # plt.plot(freq_range, dnu_estimated_average, linewidth=4,
# #          c='C3', alpha=0.4, label='Data Fit')
# # plt.fill_between(freq_range, dnu_estimated_negerr,
# #                  dnu_estimated_poserr, alpha=0.2, color='C3')
# # ax.legend(fontsize="xx-small")
# # plt.grid(True, which="both", ls="-", color='0.65')
# # plt.xlim(xl)
# # plt.xlabel('Frequency (MHz)', fontsize=Font, ha='center')
# # plt.ylabel('Scintillation Bandwidth (MHz)')
# # plt.title(psrname + ' Scintillation Bandwidth')
# # plt.savefig(str(Dnudir) + str(dyn.name.split('.')[0]) +
# #             "_Dnu_freq_phase.png")
# # plt.show()
# # plt.close()
# ###############################################################################
# And now for something completely different ...
# Noise adding parameters
mu, sigma = 0, 1
s = 0.7
Dpsr = 900
z = Dpsr * (1-s) * 3.1e16
c = 3*10**8
centre_frequency = 1150
f = centre_frequency*10**6
wavelength = c/f
k = (2*np.pi)/wavelength
rf = np.sqrt(z/k) / 10**8
rf = 2
dnu_c = 1
mb2 = 0.773*(centre_frequency/dnu_c)**(5/6)
outfile = '/Users/jacobaskew/Desktop/test.txt'
num = 100
resolution_grid = np.linspace(0.1, 5, num)
NF_grid = np.linspace(20, 2000, num)

# mb2 = 100  # Born variance of simulation
frac_bw = (0.773/mb2)**(6/5)
s0 = np.sqrt(frac_bw)*rf
estimated_scint_bandwidth_1150 = (1150)*(s0/rf)**(2)
estimated_scint_bandwidth_1350 = \
    estimated_scint_bandwidth_1150*(1.35/1.15)**(4.4)
estimated_scint_bandwidth_950 = estimated_scint_bandwidth_1150*(0.95/1.15)**(4.4)

# estimated_scint_bandwidth_1350 = (1350)*(0.773/300)**(6/5)
# estimated_scint_bandwidth_950 = (950)*(0.773/300)**(6/5)

Dnu = []
Dnuerr = []
Nscint = []
Resolution = []
Channelbandwidth = []
ACFresidual = []
# An example sim spec
# sim = Simulation(mb2=mb2, rf=2, ds=0.01, alpha=5/3, ar=1, psi=0,
#                   inner=0.001, ns=230, nf=NF,
#                   dlam=0.435, lamsteps=False, seed=64, nx=None,
#                   ny=None, dx=None, dy=None, plot=False, verbose=False,
#                   freq=centre_frequency, dt=8, mjd=50000, nsub=None,
#                   efield=False, noise=None)
# dyn = Dynspec(dyn=sim, process=False)
# dyn.trim_edges()
# dyn.plot_dyn(dpi=400)

for i in range(0, num):
    resolution = resolution_grid[i]
    # NF = int(round(centre_frequency/(resolution*dnu_c), -1))
    # NF = int(round(30/resolution, -1))
    if i > 0:
        if NF == int(round(NF_grid[i], -1)):
            continue
        else:
            NF = int(round(NF_grid[i], -1))
    else:
        NF = int(round(NF_grid[i], -1))
    # dlam for bw=30 is 0.02608695652173913
    # Creating simulated data and a dynamic spectrum from the data
    sim = Simulation(mb2=mb2, rf=2, ds=0.01, alpha=5/3, ar=1, psi=0,
                     inner=0.001, ns=230, nf=NF,
                     dlam=0.435, lamsteps=False, seed=64, nx=None,
                     ny=None, dx=None, dy=None, plot=False, verbose=False,
                     freq=centre_frequency, dt=8, mjd=50000, nsub=None,
                     efield=False, noise=None)
    dyn = Dynspec(dyn=sim, process=False)
    dyn.trim_edges()
    # ADDING NOISE
    Median = np.median(dyn.dyn)
    Maximum = np.max(dyn.dyn)
    std_dyn = np.std(dyn.dyn)
    GaussianNoise = np.random.normal(mu, sigma, dyn.dyn.shape)
    dyn.dyn[:, :] -= Median
    dyn.dyn[:, :] -= 0.75 * Maximum
    dyn.dyn[:, :] += GaussianNoise
    # ADDING NOISE
    dyn.crop_dyn(fmin=0, fmax=1000)
    dyn.plot_dyn(filename='/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/New/Simulations/Spectra//SimulatedDynspec_'+str(round(dyn.df, 2))+'.pdf', dpi=400)
    # dyn.get_acf_tilt(filename='/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/New/Simulations/ACF/SimulatedACF_'+str(round(dyn.df, 2))+'.pdf', dpi=400, plot=True, display=True)
    dyn.get_scint_params(filename='/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/New/Simulations/ACF/SimulatedACF_'+str(round(dyn.df, 2))+'.pdf', method='acf2d_approx', plot=True, display=True)
    try:
        print("ACF residual mean: ", np.mean(dyn.acf_residuals))
    except AttributeError as Error:
        print("Error: ", Error)
        continue
    # if dyn.nscint > 2000:
    #     continue
    Dnu.append(dyn.dnu)
    Dnuerr.append(dyn.dnuerr)
    Nscint.append(dyn.nscint)
    Resolution.append(resolution)
    Channelbandwidth.append(round(dyn.df, 2))
    ACFresidual.append(np.std(dyn.acf_residuals))
    # dyn.get_scint_params(method='acf2d', plot=True, display=True)
    # Dnu_acf2d.append(dyn.dnu)
    # Dnuerr_acf2d.append(dyn.dnuerr)
    # Nscint_acf2d.append(dyn.nscint)
    # Resolution_acf2d.append(resolution)
    # Channelbandwidth_acf2d.append(round(dyn.df, 2))
    # dyn.get_scint_params(method='acf1d', plot=True, display=True)
    # Dnu_acf1d.append(dyn.dnu)
    # Dnuerr_acf1d.append(dyn.dnuerr)
    # Nscint_acf1d.append(dyn.nscint)
    # Resolution_acf1d.append(resolution)
    # Channelbandwidth_acf1d.append(round(dyn.df, 2))

Font = 30
Size = 80*np.pi  # Determines the size of the datapoints used
font = {'size': 28}
matplotlib.rc('font', **font)

Dnu = np.asarray(Dnu)
Dnuerr = np.asarray(Dnuerr)
Nscint = np.asarray(Nscint)
Resolution = np.asarray(Resolution)
Freq = np.ones(Dnu.shape)*1150
Channelbandwidth = np.asarray(Channelbandwidth)

# Dnu = Dnu[np.argsort(Dnu)]
# Dnuerr = Dnuerr[np.argsort(Dnuerr)]
# Nscint = Nscint[np.argsort(Nscint)]
# Resolution = Resolution[np.argsort(Resolution)]
# Channelbandwidth = Channelbandwidth[np.argsort(Channelbandwidth)]

fig = plt.figure(figsize=(20, 10))
fig.subplots_adjust(hspace=0.5, wspace=0.5)
ax = fig.add_subplot(1, 1, 1)
cm = plt.cm.get_cmap('viridis')
z = Nscint
sc = plt.scatter(Channelbandwidth, Dnu,
                 marker='o', s=Size, alpha=0.6, c=z, cmap=cm)
plt.errorbar(Channelbandwidth, Dnu,
             yerr=Dnuerr, fmt=' ', ecolor='k',
             elinewidth=2, capsize=3, alpha=0.5)
xl = plt.xlim()
# sc = plt.scatter(Channelbandwidth_acf1d, Dnu_acf1d, marker='v', s=Size,
#                  alpha=0.6, c=z, cmap=cm)
# plt.errorbar(Channelbandwidth_acf1d, Dnu_acf1d, yerr=Dnuerr_acf1d, fmt=' ',
# ecolor='k', elinewidth=2, capsize=3, alpha=0.4)
# sc = plt.scatter(Channelbandwidth_acf2d, Dnu_acf2d, marker='s', s=Size,
#                  alpha=0.6, c=z, cmap=cm)
# plt.errorbar(Channelbandwidth_acf2d, Dnu_acf2d, yerr=Dnuerr_acf2d, fmt=' ',
#              ecolor='k', elinewidth=2, capsize=3, alpha=0.4)
# plt.hlines(estimated_scint_bandwidth_950, xl[0], xl[1], colors='C2',
#            linestyles='dashed')
plt.colorbar(sc)
plt.xlabel('Channel Bandwidth (MHz)')
plt.ylabel('Scintillation Bandwidth (MHz)')
# plt.xlim(xl)
# plt.xlim(0, 3.5)
# ax.legend()
# plt.ylim(0.4, 1)
plt.savefig("/Users/jacobaskew/Desktop/SimulatedBandwidth_res" +
            str(resolution)+".pdf")
# ax.set_xscale('log')
# ax.set_yscale('log')
plt.savefig("/Users/jacobaskew/Desktop/SimulatedBandwidth_res_log" +
            str(resolution)+".pdf")
plt.show()
plt.close()

ACFresidual_norm = (ACFresidual - np.min(ACFresidual)) / \
    (np.max(ACFresidual) - np.min(ACFresidual))
Dnu_norm = (Dnu - np.min(Dnu)) / (np.max(Dnu) - np.min(Dnu))
Dnu_alternate = Dnu*0.01

fig = plt.figure(figsize=(20, 10))
fig.subplots_adjust(hspace=0.5, wspace=0.5)
ax = fig.add_subplot(1, 1, 1)
plt.plot(Channelbandwidth, ACFresidual_norm, color='C1', alpha=0.7)
plt.scatter(Channelbandwidth, Dnu_norm, c='C0', marker='o', s=Size, alpha=0.6)
xl = plt.xlim()
plt.xlabel('Channel Bandwidth (MHz)')
plt.ylabel(r'Normalised $\Delta\nu$ and ACF $\sigma$ (arb)')
# plt.savefig("/Users/jacobaskew/Desktop/SimulatedBandwidth_res_log" +
#             str(resolution)+".pdf")
plt.show()
plt.close()

# dyn_initial.plot_dyn(filename='/Users/jacobaskew/Desktop/SimSpectraTest_' +
#                      str(resolution) + '.png', dpi=400)
###############################################################################

# x = np.linspace(np.min(dyn.times/60), np.max(dyn.times/60), 256)
# y = np.linspace(np.min(dyn.freqs), np.max(dyn.freqs), 4096)
# grid_x, grid_y = np.meshgrid(x, y)

# points_y = np.linspace(0, 4096, 4096)
# points_x = np.linspace(0, 256, 255)
# values = dyn.dyn
# grid_test = griddata((points_y, points_x), values, (grid_x, grid_y), method='linear')

# plt.imshow(grid_test.T, extent=(np.min(dyn.times/60), np.max(dyn.times/60),
#                                 np.min(dyn.freqs), np.max(dyn.freqs)))

# ###############################################################################


# def func(x, y):
#     return x*(1-x)*np.cos(4*np.pi*x) * np.sin(4*np.pi*y**2)**2


# grid_x, grid_y = np.mgrid[0:1:100j, 0:1:200j]
# rng = np.random.default_rng()
# points = rng.random((1000, 2))
# values = func(points[:, 0], points[:, 1])

# grid_test = griddata(points, values, (grid_x, grid_y), method='linear')
# plt.imshow(grid_test.T, extent=(0, 1, 0, 1))

# ###############################################################################

# Font = 10
# Size = 80*np.pi  # Determines the size of the datapoints used
# font = {'size': 10}
# matplotlib.rc('font', **font)

# x = np.linspace(0, 34, 256)
# y = np.linspace(900, 1400, 4096)
# X, Y = np.meshgrid(x, y)


# # def f(x, y):
# #     s = np.hypot(x, y)
# #     phi = np.arctan2(y, x)
# #     tau = s + s*(1-s)/5 * np.sin(6*phi)
# #     return 5*(1-tau) + tau


# T = f(X, Y)
# T = dyn.dyn
# # Choose npts random point from the discrete domain of our model function
# # npts = 1000
# # px, py = np.random.choice(x, npts), np.random.choice(y, npts)

# # values = np.ones(dyn.dyn.shape)
# # for i in range(0, dyn.dyn.shape[0]):
# #     for ii in range(0, dyn.dyn.shape[1]):
# #         values[i, ii] = dyn.dyn[int(py[i-1]), int(px[ii-1])]
# # values = values.flatten()
# fig = plt.figure(figsize=(20, 15))
# fig, ax = plt.subplots(nrows=2, ncols=2)
# fig.subplots_adjust(hspace=0.5, wspace=0.5)
# # Plot the model function and the randomly selected sample points
# ax[0, 0].contourf(X, Y, T)
# ax[0, 0].scatter(x, y, c='k', alpha=0.2, marker='.')
# ax[0, 0].set_title('Sample points on f(X,Y)')

# # Interpolate using three different methods and plot
# for i, method in enumerate(('nearest', 'linear', 'cubic')):
#     Ti = griddata((x, y), dyn.dyn, (X, Y), method=method)
#     r, c = (i+1) // 2, (i+1) % 2
#     ax[r, c].contourf(X, Y, Ti)
#     ax[r, c].set_title("method = '{}'".format(method))

# # plt.tight_layout()
# plt.show()
# ###############################################################################
# fedges = fedges[0:len(fedges)-1]
# tedges = tedges[0:len(tedges)-1]
# X, Y = np.meshgrid(tedges, fedges)
# grid_test = griddata((fedges, tedges), dyn.dyn, (X, Y), method='nearest')


###############################################################################
# Here I want to simulate the dynspec and change the number of channels using
# a 2D interpolator??!

# An example sim spec
sim = Simulation(mb2=mb2, rf=2, ds=0.01, alpha=5/3, ar=1, psi=0,
                 inner=0.001, ns=256, nf=4096,
                 dlam=0.435, lamsteps=False, seed=64, nx=None,
                 ny=None, dx=None, dy=None, plot=False, verbose=False,
                 freq=centre_frequency, dt=8, mjd=50000, nsub=None,
                 efield=False, noise=None)
dyn = Dynspec(dyn=sim, process=False)
dyn.trim_edges()
dyn.plot_dyn(dpi=400)
dyn.get_scint_params(method='acf2d_approx', plot=True, display=True)

tedges = centres_to_edges(dyn.times/60)
fedges = centres_to_edges(dyn.freqs)
medval = np.median(dyn.dyn[is_valid(dyn.dyn)*np.array(np.abs(
                                              is_valid(dyn.dyn)) > 0)])
minval = np.min(dyn.dyn[is_valid(dyn.dyn)*np.array(np.abs(
                                            is_valid(dyn.dyn)) > 0)])
# standard deviation
std = np.std(dyn.dyn[is_valid(dyn.dyn)*np.array(np.abs(
                                        is_valid(dyn.dyn)) > 0)])
vmin = minval + std
vmax = medval + 4*std
plt.figure(figsize=(20, 15))
plt.pcolormesh(tedges, fedges, dyn.dyn, vmin=vmin, vmax=vmax, linewidth=0,
               rasterized=True, shading='auto')
plt.ylabel('Frequency (MHz)')
plt.xlabel('Time (mins)')
plt.savefig('/Users/jacobaskew/Desktop/test.pdf', dpi=400,
            bbox_inches='tight', pad_inches=0.1)
plt.show()
plt.close()

wd = '/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/Simulation/'
datadir = wd + 'Dynspec/'

grid_num_array = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
Nscint = []
Channelbandwidth = []
Dnu = []
Dnuerr = []

for grid_num in range(1, 1024, 2):
    # for x in range(1, 11):
    # grid_num = grid_num_array[x]
    grid_num += 1
    dynspecfile = \
        datadir+'DynspecPlotFiles/CompleteDynspec'+str(grid_num)+'.dynspec'
    Fedges = []
    new_freq_len = int(dyn.dyn.shape[0]/grid_num)
    new_dyn = np.zeros((new_freq_len, dyn.dyn.shape[1]))
    for i in range(0, len(dyn.freqs)):  # new_freq_len
        iii = i
        iv = i*grid_num
        ii = iv + grid_num - 1
        if ii >= int(dyn.dyn.shape[0]):
            continue
        if iv % grid_num:
            continue
        Fedges.append(dyn.freqs[i])
        new_dyn[iii, :] = (dyn.dyn[iv, :] + dyn.dyn[ii, :]) / 2
    print("THIS IS VERY IMPORTANT", len(Fedges))
    medval = np.median(new_dyn[is_valid(new_dyn)*np.array(np.abs(
                                                  is_valid(new_dyn)) > 0)])
    minval = np.min(new_dyn[is_valid(new_dyn)*np.array(np.abs(
                                                is_valid(new_dyn)) > 0)])
    # standard deviation
    std = np.std(new_dyn[is_valid(new_dyn)*np.array(np.abs(
                                            is_valid(new_dyn)) > 0)])
    vmin = minval + std
    vmax = medval + 4*std
    plt.figure(figsize=(20, 15))
    plt.pcolormesh(tedges, Fedges, new_dyn, vmin=vmin, vmax=vmax,
                   linewidth=0,
                   rasterized=True, shading='auto')
    plt.ylabel('Frequency (MHz)')
    plt.xlabel('Time (mins)')
    plt.savefig('/Users/jacobaskew/Desktop/test.pdf', dpi=400,
                bbox_inches='tight', pad_inches=0.1)
    plt.show()
    plt.close()

    # A_dyn = Basicdyn(dyn=new_dyn, name="BasicDyn", header=["BasicDyn"],
    #                  times=[], freqs=[], nchan=None, nsub=None, bw=None,
    #                  df=None, freq=None, tobs=None, dt=None, mjd=None)
    array_to_dynspec(flux=new_dyn, times=tedges*60, freqs=Fedges,
                     filename=str(dynspecfile))
    sim = Simulation()
    dyn_sim = Dynspec(dyn=sim, process=False)
    dyn_sim.load_file(filename=dynspecfile)
    dyn_sim.plot_dyn(dpi=400)
    dyn_sim.get_scint_params(method='acf2d_approx', plot=True, display=True)
    Nscint.append(dyn_sim.nscint)
    Channelbandwidth.append(dyn_sim.df)
    Dnu.append(dyn_sim.dnu)
    Dnuerr.append(dyn_sim.dnuerr)
Font = 30
Size = 80*np.pi  # Determines the size of the datapoints used
font = {'size': 28}
matplotlib.rc('font', **font)

fig = plt.figure(figsize=(20, 10))
fig.subplots_adjust(hspace=0.5, wspace=0.5)
ax = fig.add_subplot(1, 1, 1)
cm = plt.cm.get_cmap('viridis')
z = Nscint
sc = plt.scatter(Channelbandwidth, Dnu,
                 marker='o', s=Size, alpha=0.6, c=z, cmap=cm)
plt.errorbar(Channelbandwidth, Dnu,
             yerr=Dnuerr, fmt=' ', ecolor='k',
             elinewidth=2, capsize=3, alpha=0.5)
xl = plt.xlim()
plt.hlines(estimated_scint_bandwidth_1150, xl[0], xl[1], colors='C2',
           linestyles='dashed')
plt.colorbar(sc)
plt.xlabel('Channel Bandwidth (MHz)')
plt.ylabel('Scintillation Bandwidth (MHz)')
plt.savefig("/Users/jacobaskew/Desktop/SimulatedBandwidth_res.pdf")
plt.show()
plt.close()
ax.set_xscale('log')
ax.set_yscale('log')
plt.savefig("/Users/jacobaskew/Desktop/SimulatedBandwidth_res_log.pdf")
plt.show()
plt.close()


# nf, nt = np.shape(dyn.dyn)
# scalefrac = 1/(max(dyn.freqs)/min(dyn.freqs))
# # timestep = max(dyn.times)*(1 - scalefrac)/(nf + 1)  # time step
# trapdyn = np.empty(shape=np.shape(dyn.dyn))

# for i in range(0, 4096):
#     idyn = dyn.dyn[i, :]
#     # maxtime = max(dyn.times)-(nf-(ii+1))*timestep
#     maxtime = np.max(dyn.times)
#     # How many times to resample to, for a given frequency
#     inddata = np.argwhere(dyn.times <= maxtime)
#     # How many trailing zeros to add
#     indzeros = np.argwhere(dyn.times > maxtime)
#     # Interpolate line
#     newline = np.interp(
#               np.linspace(min(dyn.times), max(dyn.times),
#                           len(inddata)), dyn.times, idyn)
#     newline = list(newline) + list(np.zeros(np.shape(indzeros)))
#     trapdyn[i, :] = newline

# medval = np.median(trapdyn[is_valid(trapdyn)*np.array(np.abs(
#                                               is_valid(trapdyn)) > 0)])
# minval = np.min(trapdyn[is_valid(trapdyn)*np.array(np.abs(
#                                             is_valid(trapdyn)) > 0)])
# # standard deviation
# std = np.std(trapdyn[is_valid(trapdyn)*np.array(np.abs(
#                                         is_valid(trapdyn)) > 0)])
# vmin = minval + std
# vmax = medval + 4*std
# plt.figure(figsize=(20, 15))
# plt.pcolormesh(tedges, fedges, trapdyn, vmin=vmin, vmax=vmax, linewidth=0,
#                rasterized=True, shading='auto')
# plt.ylabel('Frequency (MHz)')
# plt.xlabel('Time (mins)')
# plt.savefig('/Users/jacobaskew/Desktop/test.pdf', dpi=400,
#             bbox_inches='tight', pad_inches=0.1)
# plt.show()
# plt.close()


###############################################################################
# i_list = []
# ii_list = []
# iii_list = []

grid_num = 4

new_freq_len = int(dyn.dyn.shape[0]/grid_num)
new_dyn = np.zeros(dyn.dyn.shape)
# new_dyn = np.zeros((new_freq_len, dyn.dyn.shape[1]))

for i in range(0, len(dyn.freqs)):  # new_freq_len
    iii = i*grid_num
    ii = iii + grid_num - 1
    # i_list.append(i)
    # ii_list.append(ii)
    # iii_list.append(iii)
    if ii >= int(dyn.dyn.shape[0]):
        Fedges = np.linspace(start=fedges[0], stop=fedges[len(fedges)-1],
                             num=i)
        break
findex = []
for i in range(0, len(Fedges)):
    if np.argmin(abs(fedges - Fedges[i])) >= int(dyn.dyn.shape[0]):
        continue
    findex.append(np.argmin(abs(fedges - Fedges[i])))
findex = np.asarray(findex)
# for i in range(0, len(findex)):
#     ii = i + grid_num - 1
#     if ii >= len(findex):
#         continue
#     new_dyn[i, :] = (dyn.dyn[findex[i], :] + dyn.dyn[findex[ii], :]) / 2
#     # new_dyn[i:ii, :] = dyn.dyn[findex[i:ii], :]
#     # new_dyn[i:ii, :] = (dyn.dyn[findex[i], :] + dyn.dyn[findex[ii], :]) / 2
#
for i in range(0, len(findex)):
    if i == 0:
        continue
    findex_prev = findex[i-1]
    findex_now = findex[i]
    iii = ii + grid_num - 1
    if iii >= len(findex):
        continue
    new_dyn[findex_prev:findex_now, :] = (dyn.dyn[findex_prev, :] +
                                          dyn.dyn[findex_now, :]) / 2
#
medval = np.median(new_dyn[is_valid(new_dyn)*np.array(np.abs(
                                              is_valid(new_dyn)) > 0)])
minval = np.min(new_dyn[is_valid(new_dyn)*np.array(np.abs(
                                            is_valid(new_dyn)) > 0)])
# standard deviation
std = np.std(new_dyn[is_valid(new_dyn)*np.array(np.abs(
                                        is_valid(new_dyn)) > 0)])
vmin = minval + std
vmax = medval + 4*std

plt.figure(figsize=(20, 15))
cm = plt.cm.get_cmap('viridis')
z = new_dyn
sc = plt.pcolormesh(tedges, fedges, new_dyn, vmin=vmin, vmax=vmax,
                    linewidth=0,
                    rasterized=True, shading='auto')
plt.colorbar(sc)
plt.ylabel('Frequency (MHz)')
plt.xlabel('Time (mins)')
plt.savefig('/Users/jacobaskew/Desktop/test.pdf', dpi=400,
            bbox_inches='tight', pad_inches=0.1)
plt.show()
plt.close()
###############################################################################











