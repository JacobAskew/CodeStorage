#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 17:34:31 2021

@author: jacobaskew
"""

# Modelling the scintillation bandwidth and timescale variations across
# a single observation in time

##############################################################################
# Common #
from scintools.dynspec import Dynspec
from scintools.scint_utils import write_results, read_results, \
    float_array_from_dict, interp_nan_2d, read_par, get_earth_velocity, \
    get_true_anomaly, scint_velocity
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import glob
from copy import deepcopy as cp
import math
import os
from lmfit import Parameters, minimize

psrname = 'J0737-3039A'
pulsar = '0737-3039A'
##############################################################################
# OzStar #
# datadir = '/fred/oz002/jaskew/0737_Project/RawData/'
# outdir = '/fred/oz002/jaskew/0737_Project/Datafiles/'
# spectradir = '/fred/oz002/jaskew/0737_Project/RawDynspec/'
# par_dir = '/fred/oz002/jaskew/Data/ParFiles/'
# eclipsefile = '/fred/oz002/jaskew/Eclipse_mjd.txt'
# plotdir = '/fred/oz002/jaskew/0737_Project/Plots/'
##############################################################################
# Local #
datadir = '/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/RawData/'
desktopdir = '/Users/jacobaskew/Desktop/'
spectradir = '/Users/jacobaskew/Desktop/DoublePulsar_Project/' + str(pulsar) + '/Spectra/'
par_dir = '/Users/jacobaskew/Desktop/Swinburne_Daniel/ParFiles/'
plotdir = '/Users/jacobaskew/Desktop/DoublePulsar_Project/' + str(pulsar) + '/Plots/'
eclipsefile = '/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/Datafiles/Eclipse_mjd.txt'
outdir = '/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/Datafiles/' 
HighFreqDir = '/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/HighFreq/Plots/'
Spectradir = "/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/LongTrack/Spectra/"
HighFreqSpectradir = "/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/LongTrack/HighFreqSpectra/"
ACFtiltdir = "/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/LongTrack/ACFtilt/"
Dnudir = "/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/LongTrack/Dnu/"
Vissdir = "/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/LongTrack/Viss/"
NormTimeSeriesdir = "/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/LongTrack/NormTimeSeries/"
Phasegraddir = "/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/LongTrack/Phasegrad/"
Taudir = "/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/LongTrack/Tau/"
##############################################################################
# Also Common #
dynspecs = sorted(glob.glob(datadir + '/*ar.dynspec'))
outfiles = str(outdir) + str(psrname) + \
    '_ScintillationResults_TimescaleVariance_'
##############################################################################
# Manual Inputs #
plotting = True
zap = False
linear = False
group = False
individual = True
measurement = True
time_bin_length_initial = 10
freq_bin_length_initial = 40
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


##############################################################################

def SearchEclipse(start_mjd, tobs, times):
    end_mjd = start_mjd + tobs/86400
    Eclipse_mjd = np.genfromtxt(eclipsefile, delimiter=',', encoding=None,
                                dtype=float)
    Eclipse_events = np.array(np.where((Eclipse_mjd > start_mjd) *
                              (Eclipse_mjd < end_mjd)))
    print("Searching for eclipse ...")
    if Eclipse_events.size == 0:
        Eclipse_index = None
        print("No eclispe in dynspec ...")
        return Eclipse_index
    elif Eclipse_events.shape[1] == 1:
        Eclipse_events_mjd = Eclipse_mjd[Eclipse_events]
        mjds = start_mjd + times/86400
        Eclipse_index = np.argmin(abs(mjds - Eclipse_events_mjd))
        if Eclipse_index is not None:
            dyn_tot.dyn[:, Eclipse_index-3:Eclipse_index+3] = np.nan
            dyn_tot.dyn[:, Eclipse_index-4:Eclipse_index+4] = \
                interp_nan_2d(dyn_tot.dyn[:, Eclipse_index-4:Eclipse_index+4])

    elif Eclipse_events.shape[1] > 1:
        print("Interpolating across eclipse ...")
        Eclipse_index = []
        for i in range(0, Eclipse_events.size):
            Eclipse_events_mjd = Eclipse_mjd[Eclipse_events]
            mjds = start_mjd + times/86400
            Eclipse_index.append(np.argmin(abs(mjds -
                                               Eclipse_events_mjd[:, i])))
        if Eclipse_index is not None and len(Eclipse_index) > 1:
            for i in range(0, len(Eclipse_index)):
                dyn_tot.dyn[:, Eclipse_index[i]-3:Eclipse_index[i]+3] = np.nan
                dyn_tot.dyn[:, Eclipse_index[i]-4:Eclipse_index[i]+4] = \
                    interp_nan_2d(dyn_tot.dyn[:,
                                              Eclipse_index[i] -
                                              4:Eclipse_index[i]+4])
    return Eclipse_index


##############################################################################


def modelling_results(plot=False, analysis=False, freqmin=1630):
    if analysis:
        if group:
            # Cropping the spectra to what we actually measure
            outfile = outfiles + str(dyn_tot.name) + ".txt"
            time_len = int((round(dyn_tot.tobs, 0) - 10))
            dyn_tot.crop_dyn(fmin=freqmin, fmax=np.inf)
            dyn_tot.plot_dyn(filename=str(desktopdir) +
                             str(dyn_tot.name.split('.')[0]) +
                             '_CroppedDynspec')
            dyn_tot.plot_dyn(filename=str(HighFreqSpectradir) +
                             str(dyn_tot.name.split('.')[0]) +
                             '_CroppedDynspec')
            for i in range(0, time_len):
                try:
                    dyn = cp(dyn_tot)
                    dyn.crop_dyn(fmin=freqmin, fmax=np.inf, tmin=0+i,
                                 tmax=10+i)
                    if zap:
                        dyn.zap()
                    if linear:
                        dyn.refill(linear=True)
                    else:
                        dyn.refill(linear=False)
                    if dyn.tobs/60 < 5:
                        continue
                    dyn.get_acf_tilt(plot=False, display=False)
                    dyn.get_scint_params(method='acf2d_approx',
                                         flux_estimate=True, plot=False,
                                         display=False)
                    write_results(outfile, dyn=dyn)
                except Exception as e:
                    print(e)
                    print("THIS FILE DIDN'T WORK")
                    continue

            print("Creating models for this group ... ")
            params = read_results(outfile)
            pars = read_par(str(par_dir) + str(psrname) + '.par')

            # Read in arrays
            mjd = float_array_from_dict(params, 'mjd')  # MJD for observation
            df = float_array_from_dict(params, 'df')  # channel bandwidth
            dnu = float_array_from_dict(params, 'dnu')  # scint bandwidth
            dnu_est = float_array_from_dict(params, 'dnu_est')  # est bandwidth
            dnuerr = float_array_from_dict(params, 'dnuerr')
            tau = float_array_from_dict(params, 'tau')
            tauerr = float_array_from_dict(params, 'tauerr')
            freq = float_array_from_dict(params, 'freq')
            bw = float_array_from_dict(params, 'bw')
            scintle_num = float_array_from_dict(params, 'scintle_num')
            tobs = float_array_from_dict(params, 'tobs')  # tobs in second
            rcvrs = np.array([rcvr[0] for rcvr in params['name']])
            acf_tilt = float_array_from_dict(params, 'acf_tilt')
            acf_tilt_err = float_array_from_dict(params, 'acf_tilt_err')
            phasegrad = float_array_from_dict(params, 'phasegrad')
            phasegraderr = float_array_from_dict(params, 'phasegraderr')

            # Sort by MJD
            sort_ind = np.argsort(mjd)

            mjd = np.array(mjd[sort_ind]).squeeze()
            df = np.array(df[sort_ind]).squeeze()
            dnu = np.array(dnu[sort_ind]).squeeze()
            dnu_est = np.array(dnu_est[sort_ind]).squeeze()
            dnuerr = np.array(dnuerr[sort_ind]).squeeze()
            tau = np.array(tau[sort_ind]).squeeze()
            tauerr = np.array(tauerr[sort_ind]).squeeze()
            rcvrs = np.array(rcvrs[sort_ind]).squeeze()
            freq = np.array(freq[sort_ind]).squeeze()
            tobs = np.array(tobs[sort_ind]).squeeze()
            scintle_num = np.array(scintle_num[sort_ind]).squeeze()
            bw = np.array(bw[sort_ind]).squeeze()
            acf_tilt = np.array(acf_tilt[sort_ind]).squeeze()
            acf_tilt_err = np.array(acf_tilt_err[sort_ind]).squeeze()
            phasegrad = np.array(phasegrad[sort_ind]).squeeze()
            phasegraderr = np.array(phasegraderr[sort_ind]).squeeze()

            # Do corrections!

            # indicies = np.argwhere((tauerr < 0.5*tau) * (dnuerr < 0.5*dnu))

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
            # scintle_num = scintle_num[indicies].squeeze()
            # bw = bw[indicies].squeeze()
            # acf_tilt = np.array(acf_tilt[indicies]).squeeze()
            # acf_tilt_err = np.array(acf_tilt_err[indicies]).squeeze()
            # phasegrad = np.array(phasegrad[indicies]).squeeze()
            # phasegraderr = np.array(phasegraderr[indicies]).squeeze()

            # Make MJD centre of observation, instead of start
            mjd = mjd + tobs/86400/2
            mjd_min = (mjd*(60*24))
            mjd_min = mjd_min - mjd_min[0]

            Font = 30
            Size = 80*np.pi  # Determines the size of the datapoints used
            font = {'size': 28}
            matplotlib.rc('font', **font)

            # Tau #
            fig = plt.figure(figsize=(15, 10))
            ax = fig.add_subplot(1, 1, 1)
            plt.scatter(mjd_min, tau, c='C3', alpha=0.6, s=Size)
            plt.errorbar(mjd_min, tau, yerr=tauerr, fmt=' ', ecolor='red',
                         alpha=0.4, elinewidth=5)
            plt.xlabel('Time (mins)')
            plt.ylabel('Scintillation Timescale (seconds)')
            plt.title(psrname + r', $\tau_d$')
            plt.grid()
            plt.ylim(0, 800)
            plt.savefig(str(Taudir) + str(dyn_tot.name.split('.')[0]) +
                        "_Tau_TimeSeries.png")
            plt.show()
            plt.close()

            # Dnu #
            fig = plt.figure(figsize=(15, 10))
            ax = fig.add_subplot(1, 1, 1)
            plt.scatter(mjd_min, dnu, c='C0', alpha=0.6, s=Size)
            plt.errorbar(mjd_min, dnu, yerr=dnuerr, fmt=' ', ecolor='C0',
                         alpha=0.4, elinewidth=5)
            xl = plt.xlim()
            plt.plot(xl, (df[0], df[0]), color='C2')
            plt.xlabel('Time (mins)')
            plt.ylabel('Scintillation Bandwidth (MHz)')
            plt.xlim(xl)
            plt.grid()
            plt.ylim(0, 8)
            plt.title(psrname + r', $\Delta\nu$')
            plt.savefig(str(Dnudir) + str(dyn_tot.name.split('.')[0]) +
                        "_Dnu_TimeSeries.png")
            plt.show()
            plt.close()

            # ACF Tilt #
            fig = plt.figure(figsize=(15, 10))
            ax = fig.add_subplot(1, 1, 1)
            plt.scatter(mjd_min, acf_tilt, c='C4', alpha=0.6, s=Size)
            plt.errorbar(mjd_min, acf_tilt, yerr=acf_tilt_err, fmt=' ',
                         ecolor='C4', alpha=0.4, elinewidth=5)
            plt.xlabel('Time (mins)')
            plt.ylabel('ACF Tilt (?)')
            plt.ylim(-10, 10)
            plt.grid()
            plt.title(psrname + ', ACF Tilt')
            plt.savefig(str(ACFtiltdir) + str(dyn_tot.name.split('.')[0]) +
                        "_Tilt_Timeseries.png")
            plt.show()
            plt.close()

            # Phase Gradient #
            fig = plt.figure(figsize=(15, 10))
            ax = fig.add_subplot(1, 1, 1)
            plt.scatter(mjd_min, phasegrad, c='C7', alpha=0.6, s=Size)
            plt.errorbar(mjd_min, phasegrad, yerr=phasegraderr, fmt=' ',
                         ecolor='C7', alpha=0.4, elinewidth=5)
            plt.xlabel('Time (mins)')
            plt.ylabel('Phase Gradient (?)')
            plt.ylim(-2, 5)
            plt.title(psrname + ', Phase Gradient')
            plt.savefig(str(Phasegraddir) + str(dyn_tot.name.split('.')[0]) +
                        "_Tilt_Timeseries.png")
            plt.show()
            plt.grid()
            plt.close()

            # Normalised Results Plot #
            norm_dnu = (dnu-np.min(dnu))/(np.max(dnu) - np.min(dnu))
            norm_tau = (tau-np.min(tau))/(np.max(tau) - np.min(tau))
            # norm_tilt = \
            #     (acf_tilt-np.min(acf_tilt))/(np.max(acf_tilt) -
            # np.min(acf_tilt))
            norm_phase = (phasegrad-np.min(phasegrad))/(np.max(phasegrad) -
                                                        np.min(phasegrad))

            fig = plt.figure(figsize=(15, 10))
            ax = fig.add_subplot(1, 1, 1)
            plt.scatter(mjd_min, norm_dnu, c='C0', alpha=0.6, s=Size,
                        label='dnu')
            plt.scatter(mjd_min, norm_tau, c='C3', alpha=0.6, s=Size,
                        label='tau')
            # plt.scatter(mjd_min, norm_tilt, c='C4', alpha=0.6, s=Size,
            #             label='ACFtilt')
            plt.scatter(mjd_min, norm_phase, c='C7', alpha=0.6, s=Size,
                        label='phasegrad')
            plt.xlabel('Time (mins)')
            plt.ylabel('Normalised Values')
            plt.grid()
            plt.title(psrname + ' Normalised Scintillation Results')
            ax.legend(fontsize='xx-small')
            plt.savefig(str(NormTimeSeriesdir) +
                        str(dyn_tot.name.split('.')[0]) +
                        "_Norm_TimeSeries_Comparison.png")
            plt.show()
            plt.close()

        if individual:
            # Cropping the spectra to what we actually measure
            outfile = outfiles + "total.txt"
            outfile_group = outfiles + str(dyn_tot.name) + ".txt"
            if measurement:
                time_bin_length = time_bin_length_initial
                time_len = int((round(dyn_tot.tobs/60, 0) - time_bin_length))
                dyn_tot.crop_dyn(fmin=freqmin, fmax=np.inf)
                freq_bin_length = freq_bin_length_initial
                freq_bins = math.ceil((1630-freqmin)/freq_bin_length)
                if freq_bins == 0:
                    freq_bins = 1
                time_bins = math.floor(time_len/time_bin_length)
                # Plotting the cropped dynspec
                dyn_tot.plot_dyn(filename=str(desktopdir) +
                                 str(dyn_tot.name.split('.')[0]) +
                                 '_CroppedDynspec')
                dyn_tot.plot_dyn(filename=str(HighFreqSpectradir) +
                                 str(dyn_tot.name.split('.')[0]) +
                                 '_CroppedDynspec')

                for i in range(0, time_bins):
                    time = i * time_bin_length
                    for ii in range(0, freq_bins):
                        freq0 = ii * freq_bin_length
                        freq1 = (ii - 1) * freq_bin_length
                        if ii == 0:
                            freq1 = 0
                        if 1630-freq0 < 0:
                            continue
                        elif 1630-freq1 < np.min(dyn_tot.freqs):
                            continue
                        try:
                            dyn = cp(dyn_tot)
                            if ii == 0:
                                dyn.crop_dyn(fmin=1630-freq0,
                                             fmax=np.max(dyn_tot.freqs),
                                             tmin=time,
                                             tmax=time_bin_length+time)

                            else:
                                dyn.crop_dyn(fmin=1630-freq0,
                                             fmax=1630-freq1,
                                             tmin=time,
                                             tmax=time_bin_length+time)
                            if zap:
                                dyn.zap()
                            if linear:
                                dyn.refill(linear=True)
                            else:
                                dyn.refill(linear=False)
                            dyn.get_acf_tilt(plot=False, display=False)
                            dyn.get_scint_params(method='acf2d_approx',
                                                 flux_estimate=True,
                                                 plot=False, display=False)
                            write_results(outfile, dyn=dyn)
                            write_results(outfile_group, dyn=dyn)
                            print("Got a live one: " + str(i))
                        except Exception as e:
                            print(e)
                            print("THIS FILE DIDN'T WORK")
                            print("Tmin: " + str(time))
                            print("Tmax: " + str(time_bin_length+time))
                            print("Fmin: " + str(1630-freq0))
                            if ii == 0:
                                print("Fmax: " + str(np.max(dyn_tot.freqs)))
                            else:
                                print("Fmax: " + str(1630-freq1))
                            continue

            if plotting:
                dnuplotdir = str(Dnudir) + dyn_tot.name
                try:
                    os.mkdir(dnuplotdir)
                except OSError as error:
                    print(error)

                print("Creating models for this group ... ")
                params = read_results(outfile_group)
                pars = read_par(str(par_dir) + str(psrname) + '.par')

                # Read in arrays
                mjd = float_array_from_dict(params, 'mjd')
                df = float_array_from_dict(params, 'df')  # channel bandwidth
                dnu = float_array_from_dict(params, 'dnu')  # scint bandwidth
                dnu_est = float_array_from_dict(params, 'dnu_est')
                dnuerr = float_array_from_dict(params, 'dnuerr')
                tau = float_array_from_dict(params, 'tau')
                tauerr = float_array_from_dict(params, 'tauerr')
                freq = float_array_from_dict(params, 'freq')
                bw = float_array_from_dict(params, 'bw')
                scintle_num = float_array_from_dict(params, 'scintle_num')
                tobs = float_array_from_dict(params, 'tobs')  # tobs in second
                rcvrs = np.array([rcvr[0] for rcvr in params['name']])
                acf_tilt = float_array_from_dict(params, 'acf_tilt')
                acf_tilt_err = float_array_from_dict(params, 'acf_tilt_err')
                phasegrad = float_array_from_dict(params, 'phasegrad')
                phasegraderr = float_array_from_dict(params, 'phasegraderr')

                # Sort by MJD
                sort_ind = np.argsort(mjd)

                mjd = np.array(mjd[sort_ind]).squeeze()
                df = np.array(df[sort_ind]).squeeze()
                dnu = np.array(dnu[sort_ind]).squeeze()
                dnu_est = np.array(dnu_est[sort_ind]).squeeze()
                dnuerr = np.array(dnuerr[sort_ind]).squeeze()
                tau = np.array(tau[sort_ind]).squeeze()
                tauerr = np.array(tauerr[sort_ind]).squeeze()
                rcvrs = np.array(rcvrs[sort_ind]).squeeze()
                freq = np.array(freq[sort_ind]).squeeze()
                tobs = np.array(tobs[sort_ind]).squeeze()
                scintle_num = np.array(scintle_num[sort_ind]).squeeze()
                bw = np.array(bw[sort_ind]).squeeze()
                acf_tilt = np.array(acf_tilt[sort_ind]).squeeze()
                acf_tilt_err = np.array(acf_tilt_err[sort_ind]).squeeze()
                phasegrad = np.array(phasegrad[sort_ind]).squeeze()
                phasegraderr = np.array(phasegraderr[sort_ind]).squeeze()

                # Do corrections!

                indicies = np.argwhere((tauerr < 1*tau) * (dnuerr < 1*dnu))

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
                scintle_num = scintle_num[indicies].squeeze()
                bw = bw[indicies].squeeze()
                acf_tilt = np.array(acf_tilt[indicies]).squeeze()
                acf_tilt_err = np.array(acf_tilt_err[indicies]).squeeze()
                phasegrad = np.array(phasegrad[indicies]).squeeze()
                phasegraderr = np.array(phasegraderr[indicies]).squeeze()

                # Make MJD centre of observation, instead of start
                mjd = mjd + tobs/86400/2
                mjd_annual = mjd % 365.2425
                print('Getting Earth velocity')
                vearth_ra, vearth_dec = get_earth_velocity(mjd, pars['RAJ'],
                                                           pars['DECJ'])
                print('Getting true anomaly')
                pars['PBDOT'] *= 10**-12  # add in factor dropped by tempo
                U = get_true_anomaly(mjd, pars)
                om = pars['OM'] + pars['OMDOT']*(mjd - pars['T0'])/365.2425
                # compute orbital phase
                phase = U*180/np.pi + om
                phase[phase > 360] = phase[phase > 360] - 360

                Aiss = 2.78*10**4  # thin screen,
                # table 2 of Cordes & Rickett (1998)
                # D = 1  # kpc

                viss, visserr = scint_velocity(None, dnu, tau, freq, dnuerr,
                                               tauerr, a=Aiss)

                Font = 30
                Size = 80*np.pi  # Determines the size of the datapoints used
                font = {'size': 28}
                matplotlib.rc('font', **font)

                # Tau Phase #
                fig = plt.figure(figsize=(15, 10))
                ax = fig.add_subplot(1, 1, 1)
                plt.scatter(phase, tau, c='C3', alpha=0.6, s=Size)
                plt.errorbar(phase, tau, yerr=tauerr, fmt=' ', ecolor='C3',
                             alpha=0.4, elinewidth=5)
                xl = plt.xlim()
                plt.xlabel('Orbital Phase (degrees)', fontsize=Font,
                           ha='center')
                plt.ylabel('Scintillation Timescale (seconds)')
                plt.xlim(xl)
                plt.grid()
                plt.ylim(0, 800)
                plt.title(psrname + r', $\tau_d$')
                plt.savefig(str(Taudir) + str(dyn_tot.name.split('.')[0]) +
                            "_Tau_Phase.png")
                plt.show()
                plt.close()

                # Dnu v Time v Frequency
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
                plt.xlabel('Orbital Phase (degrees)', fontsize=Font,
                           ha='center')
                plt.ylabel('Scintillation Bandwidth (MHz)')
                plt.title(psrname + r', $\Delta\nu$')
                plt.grid(True, which="both", ls="-", color='0.65')
                plt.savefig(str(Dnudir) + str(dyn_tot.name.split('.')[0]) +
                            "_Dnu_phase.png")
                plt.show()
                plt.close()

                # Viss v Phase
                fig = plt.figure(figsize=(20, 10))
                fig.subplots_adjust(hspace=0.5, wspace=0.5)
                plt.scatter(phase, viss, c='k', s=Size, alpha=0.6)
                plt.errorbar(phase, viss, yerr=visserr, fmt=' ', ecolor='k',
                             elinewidth=2, capsize=3, alpha=0.55)
                plt.xlabel('Orbital Phase (degrees)', fontsize=Font,
                           ha='center')
                plt.ylabel(r'Scintillation Velocity ($ms^{-1}$)')
                plt.title(psrname + r', $V_{iss}$')
                plt.grid(True, which="both", ls="-", color='0.65')
                plt.savefig(str(Vissdir) + str(dyn_tot.name.split('.')[0]) +
                            "_viss_phase.png")
                plt.show()
                plt.close()

                # Dnu v Frequency v Time
                fig = plt.figure(figsize=(20, 10))
                fig.subplots_adjust(hspace=0.5, wspace=0.5)
                ax = fig.add_subplot(1, 1, 1)
                cm = plt.cm.get_cmap('viridis')
                z = phase
                sc = plt.scatter(freq, dnu, c=z, cmap=cm, s=Size, alpha=0.6)
                plt.colorbar(sc)
                plt.errorbar(freq, dnu, yerr=dnuerr, fmt=' ', ecolor='k',
                             elinewidth=2, capsize=3, alpha=0.55)
                xl = plt.xlim()
                plt.plot(xl, (df[0], df[0]), linewidth=4, color='C2')
                freq_range = np.linspace(xl[0], xl[1], 100)
                freq_upper_average = \
                    np.average(freq[np.argwhere(freq > 1500)].flatten())
                dnu_upper_average = \
                    np.average(dnu[np.argwhere(freq > 1500)].flatten(),
                               weights=1/dnuerr[np.argwhere(freq >
                                                            1500)].flatten())
                # Here we are calculated for the residuals of the powerlaw
                parameters = Parameters()
                parameters.add('amp', value=dnu_upper_average, vary=True,
                               min=0, max=2)
                parameters.add('alpha', value=4, vary=True, min=1, max=5)
                results = minimize(powerlaw, parameters,
                                   args=(freq,
                                         dnu,
                                         dnuerr,
                                         1400,
                                         dnu_upper_average),
                                   method='emcee', steps=1000, burn=100)
                Slope = results.params['alpha'].value
                Slopeerr = results.params['alpha'].stderr
                dnu_estimated_individual = (freq_range /
                                            freq_upper_average)**4 * \
                    dnu_upper_average
                dnu_estimated_average = (freq_range /
                                         freq_upper_average)**Slope * \
                    dnu_upper_average
                dnu_estimated_poserr = (freq_range /
                                        freq_upper_average)**(Slope +
                                                              Slopeerr) * \
                    dnu_upper_average
                dnu_estimated_negerr = (freq_range /
                                        freq_upper_average)**(Slope -
                                                              Slopeerr) * \
                    dnu_upper_average
                plt.plot(freq_range, dnu_estimated_individual, c='k',
                         linewidth=4, alpha=0.4, label='Model')
                plt.plot(freq_range, dnu_estimated_average, linewidth=4,
                         c='C3', alpha=0.4, label='Data Fit')
                plt.fill_between(freq_range, dnu_estimated_negerr,
                                 dnu_estimated_poserr, alpha=0.2, color='C3')
                ax.legend(fontsize="xx-small")
                plt.grid(True, which="both", ls="-", color='0.65')
                plt.xlim(xl)
                plt.xlabel('Frequency (MHz)', fontsize=Font, ha='center')
                plt.ylabel('Scintillation Bandwidth (MHz)')
                plt.title(psrname + r', $\Delta\nu$')
                plt.savefig(str(Dnudir) + str(dyn_tot.name.split('.')[0]) +
                            "_Dnu_freq_phase.png")
                plt.show()
                plt.close()

                unique_mjd = np.unique(mjd)
                for i in range(0, len(unique_mjd)):
                    freq_individual = \
                        freq[np.argwhere(unique_mjd[i] == mjd)].flatten()
                    dnu_individual = \
                        dnu[np.argwhere(unique_mjd[i] == mjd)].flatten()
                    dnuerr_individual = \
                        dnuerr[np.argwhere(unique_mjd[i] == mjd)].flatten()
                    if np.argwhere(freq_individual > 1500).size == 0:
                        fit_freq_min = np.max(freq_individual) - 100
                    elif np.argwhere(freq_individual > 1500).size < 3:
                        fit_freq_min = 1400
                    else:
                        fit_freq_min = 1500
                    freq_upper = \
                        freq_individual[np.argwhere(freq_individual >
                                                    fit_freq_min)].flatten()
                    dnu_upper = \
                        dnu_individual[np.argwhere(freq_individual >
                                                   fit_freq_min)].flatten()
                    dnuerr_upper = \
                        dnuerr_individual[np.argwhere(freq_individual >
                                                      fit_freq_min)].flatten()
                    freq_upper_average = np.average(freq_upper)
                    dnu_upper_average = np.average(dnu_upper,
                                                   weights=1/dnuerr_upper)

                    # freq_individual_argsrted = np.argsort(freq_individual)
                    # Dnu v Frequency v Unique Times
                    fig = plt.figure(figsize=(20, 10))
                    fig.subplots_adjust(hspace=0.5, wspace=0.5)
                    ax = fig.add_subplot(1, 1, 1)
                    plt.scatter(freq_individual, dnu_individual, c='C0',
                                s=Size, alpha=0.6)
                    plt.errorbar(freq_individual, dnu_individual,
                                 yerr=dnuerr_individual, fmt=' ', ecolor='C0',
                                 elinewidth=3, capsize=2, alpha=0.55)
                    xl = plt.xlim()
                    freq_range = np.linspace(xl[0], xl[1], 100)
                    dnu_estimated_individual = (freq_range /
                                                freq_upper_average)**4 * \
                        dnu_upper_average
                    # Here we are calculated for the residuals of the powerlaw
                    parameters = Parameters()
                    parameters.add('amp', value=dnu_upper_average, vary=True,
                                   min=0, max=2)
                    parameters.add('alpha', value=4, vary=True, min=1, max=5)
                    results = \
                        minimize(powerlaw, parameters,
                                 args=(freq,
                                       dnu,
                                       dnuerr,
                                       1400,
                                       dnu_upper_average),
                                 method='emcee', steps=1000, burn=100)
                    Slope = results.params['alpha'].value
                    Slopeerr = results.params['alpha'].stderr
                    dnu_estimated_individual = (freq_range /
                                                freq_upper_average)**4 * \
                        dnu_upper_average
                    dnu_estimated_average = (freq_range /
                                             freq_upper_average)**Slope * \
                        dnu_upper_average
                    dnu_estimated_poserr = (freq_range /
                                            freq_upper_average)**(Slope +
                                                                  Slopeerr) * \
                        dnu_upper_average
                    dnu_estimated_negerr = (freq_range /
                                            freq_upper_average)**(Slope -
                                                                  Slopeerr) * \
                        dnu_upper_average
                    plt.plot(freq_range, dnu_estimated_individual, c='k',
                             linewidth=4, alpha=0.4, label='Model')
                    plt.plot(freq_range, dnu_estimated_average, linewidth=4,
                             c='C3', alpha=0.4, label='Data Fit')
                    plt.fill_between(freq_range, dnu_estimated_negerr,
                                     dnu_estimated_poserr, alpha=0.2,
                                     color='C3')
                    plt.grid(True, which="both", ls="-", color='0.65')
                    plt.plot(xl, (df[0], df[0]), linewidth=4, color='C2')
                    plt.xlim(xl)
                    ax.legend(fontsize="xx-small")
                    plt.xlabel('Frequency (MHz)', fontsize=Font, ha='center')
                    plt.ylabel('Scintillation Bandwidth (MHz)')
                    plt.title(psrname +
                              r', $\Delta\nu$: Chunk ' + str(i+1))
                    plt.savefig(str(dnuplotdir) + "/Dnu_freq_time" + str(i+1) +
                                ".png")
                    plt.show()
                    plt.close()

    if plot:
        outfile = outfiles + "total.txt"

        print("Creating models for this dataset ... ")
        params = read_results(outfile)
        pars = read_par(str(par_dir) + str(psrname) + '.par')

        # Read in arrays
        mjd = float_array_from_dict(params, 'mjd')  # MJD for observation start
        df = float_array_from_dict(params, 'df')  # channel bandwidth
        dnu = float_array_from_dict(params, 'dnu')  # scint bandwidth
        dnu_est = float_array_from_dict(params, 'dnu_est')  # est bandwidth
        dnuerr = float_array_from_dict(params, 'dnuerr')
        tau = float_array_from_dict(params, 'tau')
        tauerr = float_array_from_dict(params, 'tauerr')
        freq = float_array_from_dict(params, 'freq')
        bw = float_array_from_dict(params, 'bw')
        scintle_num = float_array_from_dict(params, 'scintle_num')
        tobs = float_array_from_dict(params, 'tobs')  # tobs in second
        rcvrs = np.array([rcvr[0] for rcvr in params['name']])
        acf_tilt = float_array_from_dict(params, 'acf_tilt')
        acf_tilt_err = float_array_from_dict(params, 'acf_tilt_err')
        phasegrad = float_array_from_dict(params, 'phasegrad')
        phasegraderr = float_array_from_dict(params, 'phasegraderr')

        # Sort by MJD
        sort_ind = np.argsort(mjd)

        mjd = np.array(mjd[sort_ind]).squeeze()
        df = np.array(df[sort_ind]).squeeze()
        dnu = np.array(dnu[sort_ind]).squeeze()
        dnu_est = np.array(dnu_est[sort_ind]).squeeze()
        dnuerr = np.array(dnuerr[sort_ind]).squeeze()
        tau = np.array(tau[sort_ind]).squeeze()
        tauerr = np.array(tauerr[sort_ind]).squeeze()
        rcvrs = np.array(rcvrs[sort_ind]).squeeze()
        freq = np.array(freq[sort_ind]).squeeze()
        tobs = np.array(tobs[sort_ind]).squeeze()
        scintle_num = np.array(scintle_num[sort_ind]).squeeze()
        bw = np.array(bw[sort_ind]).squeeze()
        acf_tilt = np.array(acf_tilt[sort_ind]).squeeze()
        acf_tilt_err = np.array(acf_tilt_err[sort_ind]).squeeze()
        phasegrad = np.array(phasegrad[sort_ind]).squeeze()
        phasegraderr = np.array(phasegraderr[sort_ind]).squeeze()

        # Do corrections!

        indicies = np.argwhere((tauerr < 0.2*tau) * (dnuerr < 0.2*dnu))

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
        scintle_num = scintle_num[indicies].squeeze()
        bw = bw[indicies].squeeze()
        acf_tilt = np.array(acf_tilt[indicies]).squeeze()
        acf_tilt_err = np.array(acf_tilt_err[indicies]).squeeze()
        phasegrad = np.array(phasegrad[indicies]).squeeze()
        phasegraderr = np.array(phasegraderr[indicies]).squeeze()

        # Make MJD centre of observation, instead of start
        mjd = mjd + tobs/86400/2
        mjd_annual = mjd % 365.2425
        print('Getting Earth velocity')
        vearth_ra, vearth_dec = get_earth_velocity(mjd, pars['RAJ'],
                                                   pars['DECJ'])
        print('Getting true anomaly')
        pars['PBDOT'] *= 10**-12  # add in factor dropped by tempo
        U = get_true_anomaly(mjd, pars)
        om = pars['OM'] + pars['OMDOT']*(mjd - pars['T0'])/365.2425
        # compute orbital phase
        phase = U*180/np.pi + om
        phase[phase > 360] = phase[phase > 360] - 360

        Aiss = 2.78*10**4  # thin screen, table 2 of Cordes & Rickett (1998)
        # D = 1  # kpc

        viss, visserr = scint_velocity(None, dnu, tau, freq, dnuerr,
                                       tauerr, a=Aiss)

        Font = 30
        Size = 80*np.pi  # Determines the size of the datapoints used
        font = {'size': 28}
        matplotlib.rc('font', **font)

        # # Tau # #
        # MJD #
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(1, 1, 1)
        plt.scatter(mjd, tau, c='C3', alpha=0.6, s=Size)
        plt.errorbar(mjd, tau, yerr=tauerr, fmt=' ', ecolor='C3',
                     alpha=0.4, elinewidth=5)
        plt.xlabel('MJD')
        plt.ylabel('Scintillation Timescale (seconds)')
        plt.title(psrname + r', $\tau_d$')
        plt.grid()
        plt.ylim(0, 800)
        plt.savefig(str(Taudir) + "Tau_MJD.png")
        plt.show()
        plt.close()

        # Orbital #
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(1, 1, 1)
        plt.scatter(phase, tau, c='C3', alpha=0.6, s=Size)
        plt.errorbar(phase, tau, yerr=tauerr, fmt=' ', ecolor='C3',
                     alpha=0.4, elinewidth=5)
        plt.xlabel('Orbital Phase (degrees)')
        plt.ylabel('Scintillation Timescale (seconds)')
        plt.title(psrname + r', $\tau_d$')
        plt.grid()
        plt.ylim(0, 800)
        plt.savefig(str(Taudir) + "Tau_Orbital.png")
        plt.show()
        plt.close()

        # Annual #
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(1, 1, 1)
        plt.scatter(mjd_annual, tau, c='C3', alpha=0.6, s=Size)
        plt.errorbar(mjd_annual, tau, yerr=tauerr, fmt=' ', ecolor='C3',
                     alpha=0.4, elinewidth=5)
        plt.xlabel('Annual Phase (days)')
        plt.ylabel('Scintillation Timescale (seconds)')
        plt.title(psrname + r', $\tau_d$')
        plt.grid()
        plt.ylim(0, 800)
        plt.savefig(str(Taudir) + "Tau_Annual.png")
        plt.show()
        plt.close()

        # # Dnu # #
        # MJD #
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(1, 1, 1)
        plt.scatter(mjd, dnu, c='C0', alpha=0.6, s=Size)
        plt.errorbar(mjd, dnu, yerr=dnuerr, fmt=' ', ecolor='C0',
                     alpha=0.4, elinewidth=5)
        xl = plt.xlim()
        plt.plot(xl, (df[0], df[0]), color='C2')
        plt.xlabel('MJD')
        plt.ylabel('Scintillation Bandwidth (MHz)')
        plt.xlim(xl)
        plt.grid()
        # plt.ylim(0, 8)
        plt.title(psrname + r', $\Delta\nu$')
        plt.savefig(str(Dnudir) + "Dnu_MJD.png")
        plt.show()
        plt.close()

        # Orbital #
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(1, 1, 1)
        plt.scatter(phase, dnu, c='C0', alpha=0.6, s=Size)
        plt.errorbar(phase, dnu, yerr=dnuerr, fmt=' ', ecolor='C0',
                     alpha=0.4, elinewidth=5)
        xl = plt.xlim()
        plt.plot(xl, (df[0], df[0]), color='C2')
        plt.xlabel('Orbital Phase (degrees)')
        plt.ylabel('Scintillation Bandwidth (MHz)')
        plt.xlim(xl)
        plt.grid()
        # plt.ylim(0, 8)
        plt.title(psrname + r', $\Delta\nu$')
        plt.savefig(str(Dnudir) + "Dnu_Orbital.png")
        plt.show()
        plt.close()

        # Annual #
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(1, 1, 1)
        plt.scatter(mjd_annual, dnu, c='C0', alpha=0.6, s=Size)
        plt.errorbar(mjd_annual, dnu, yerr=dnuerr, fmt=' ', ecolor='C0',
                     alpha=0.4, elinewidth=5)
        xl = plt.xlim()
        plt.plot(xl, (df[0], df[0]), color='C2')
        plt.xlabel('Annual Phase (days)')
        plt.xlim(xl)
        plt.grid()
        # plt.ylim(0, 8)
        plt.ylabel('Scintillation Bandwidth (MHz)')
        plt.title(psrname + r', $\Delta\nu$')
        plt.savefig(str(Dnudir) + "Dnu_Annual.png")
        plt.show()
        plt.close()

        # # Viss # #
        # MJD #
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(1, 1, 1)
        plt.scatter(mjd, viss, c='black', alpha=0.6, s=Size)
        plt.errorbar(mjd, viss, yerr=visserr, fmt=' ', ecolor='black',
                     alpha=0.4, elinewidth=5)
        plt.xlabel('MJD')
        plt.ylabel(r'Scintillation Velocity ($V_{iss}$)')
        plt.title(psrname + r' $V_{iss}$ MJD')
        plt.grid()
        plt.savefig(str(Vissdir) + "Viss_MJD.png")
        plt.show()
        plt.close()

        # Orbital #
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(1, 1, 1)
        plt.scatter(phase, viss, c='black', alpha=0.6, s=Size)
        plt.errorbar(phase, viss, yerr=visserr, fmt=' ', ecolor='black',
                     alpha=0.4, elinewidth=5)
        plt.xlabel('Orbital Phase (degrees)')
        plt.ylabel(r'Scintillation Velocity ($V_{iss}$)')
        plt.title(psrname + r', $V_{iss}$')
        plt.grid()
        plt.savefig(str(Vissdir) + "Viss_Orbital.png")
        plt.show()
        plt.close()

        # Annual #
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(1, 1, 1)
        plt.scatter(mjd_annual, viss, c='black', alpha=0.6, s=Size)
        plt.errorbar(mjd_annual, viss, yerr=visserr, fmt=' ', ecolor='black',
                     alpha=0.4, elinewidth=5)
        plt.xlabel('Annual Phase (days)')
        plt.ylabel(r'Scintillation Velocity ($V_{iss}$)')
        plt.title(psrname + r', $V_{iss}$')
        plt.grid()
        plt.savefig(str(Vissdir) + "Viss_Annual.png")
        plt.show()
        plt.close()

        # Dnu v Orbital v Frequency
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
        plt.grid(True, which="both", ls="-", color='0.65')
        plt.xlim(xl)
        plt.savefig(str(Dnudir) + "Dnu_Orbital_Freq.png")
        plt.show()
        plt.close()

        # Dnu v Frequency v Time
        fig = plt.figure(figsize=(20, 10))
        fig.subplots_adjust(hspace=0.5, wspace=0.5)
        ax = fig.add_subplot(1, 1, 1)
        cm = plt.cm.get_cmap('viridis')
        z = mjd
        sc = plt.scatter(freq, dnu, c=z, cmap=cm, s=Size, alpha=0.6)
        plt.colorbar(sc)
        plt.errorbar(freq, dnu, yerr=dnuerr, fmt=' ', ecolor='k',
                     elinewidth=2, capsize=3, alpha=0.55)
        xl = plt.xlim()
        plt.plot(xl, (df[0], df[0]), linewidth=4, color='C2')
        freq_range = np.linspace(xl[0], xl[1], 100)
        freq_upper_average = \
            np.average(freq[np.argwhere(freq > 1500)].flatten())
        dnu_upper_average = \
            np.average(dnu[np.argwhere(freq > 1500)].flatten(),
                       weights=1/dnuerr[np.argwhere(freq >
                                                    1500)].flatten())
        # Here we are calculated for the residuals of the powerlaw
        parameters = Parameters()
        parameters.add('amp', value=dnu_upper_average, vary=True,
                       min=0, max=2)
        parameters.add('alpha', value=4, vary=True, min=1, max=5)
        results = minimize(powerlaw, parameters,
                           args=(freq,
                                 dnu,
                                 dnuerr,
                                 1400,
                                 dnu_upper_average),
                           method='emcee', steps=10000, burn=1000)
        Slope = results.params['alpha'].value
        Slopeerr = results.params['alpha'].stderr
        dnu_estimated_individual = (freq_range /
                                    freq_upper_average)**4 * \
            dnu_upper_average
        dnu_estimated_average = (freq_range /
                                 freq_upper_average)**Slope * \
            dnu_upper_average
        dnu_estimated_poserr = (freq_range /
                                freq_upper_average)**(Slope + Slopeerr) * \
            dnu_upper_average
        dnu_estimated_negerr = (freq_range /
                                freq_upper_average)**(Slope - Slopeerr) * \
            dnu_upper_average
        plt.plot(freq_range, dnu_estimated_individual, c='k',
                 linewidth=4, alpha=0.4, label='Model')
        plt.plot(freq_range, dnu_estimated_average, linewidth=4,
                  c='C3', alpha=0.4, label='Data Fit')
        plt.fill_between(freq_range, dnu_estimated_negerr,
                         dnu_estimated_poserr, alpha=0.2, color='C3')
        ax.legend(fontsize="xx-small")
        plt.grid(True, which="both", ls="-", color='0.65')
        plt.xlim(xl)
        plt.xlabel('Frequency (MHz)', fontsize=Font, ha='center')
        plt.ylabel('Scintillation Bandwidth (MHz)')
        plt.title(psrname + r', $\Delta\nu$')
        plt.savefig(str(Dnudir) + "Dnu_freq_MJD.png")
        plt.show()
        plt.close()


##############################################################################
if plotting and not individual:
    modelling_results(plot=True)
    quit

# Group 1 # dyn3 is a loner!
# I don't think this is useable
dyn1 = Dynspec(filename="/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/RawData/J0737-3039A_2019-04-22-11:43:04_ch5.0_sub5.0.ar.dynspec", process=False)#, lamsteps=True)
dyn2 = Dynspec(filename="/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/RawData/J0737-3039A_2019-04-22-12:06:24_ch5.0_sub5.0.ar.dynspec", process=False)#, lamsteps=True)
# dyn3 = Dynspec(filename="/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/RawData/J0737-3039A_2019-05-20-17:12:56_ch5.0_sub5.0.ar.dynspec", process=False)

Fmax = np.inf
Fmin = 0
dyn1.crop_dyn(fmin=Fmin, fmax=Fmax)
# dyn1.zap()
dyn1.refill(linear=False)
dyn2.crop_dyn(fmin=Fmin, fmax=Fmax)
# dyn2.zap()
dyn2.refill(linear=False)
# dyn3.crop_dyn(fmin=Fmin, fmax=Fmax)
# # dyn2.zap()
# dyn3.refill(linear=False)

dyn_tot = dyn1 + dyn2

# Removing any eclispe
start_mjd = dyn_tot.mjd
tobs = dyn_tot.tobs
times = dyn_tot.times
Eclipse_index = SearchEclipse(start_mjd, tobs, times)

# Plotting and Saving the Dynamic Spectra for this group #
dyn_tot.plot_dyn(filename=str(desktopdir) + str(dyn_tot.name.split('.')[0]) +
                 '_Dynspec.png')
dyn_tot.plot_dyn(filename=str(Spectradir) + str(dyn_tot.name.split('.')[0]) +
                 '_Dynspec.png')

# Collecting Results and par file and plotting cropped dynspec
modelling_results(analysis=False, freqmin=1630)

# Group 2 #
# Useable but the dnu is below the observational bandwidth
dyn1 = Dynspec(filename="/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/RawData/J0737-3039A_2019-07-20-06:22:48_ch5.0_sub5.0.ar.dynspec", process=False)#, lamsteps=True)
dyn2 = Dynspec(filename="/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/RawData/J0737-3039A_2019-07-20-06:52:56_ch5.0_sub5.0.ar.dynspec", process=False)#, lamsteps=True)
dyn3 = Dynspec(filename="/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/RawData/J0737-3039A_2019-07-20-08:08:08_ch5.0_sub5.0.ar.dynspec", process=False)#, lamsteps=True)
dyn4 = Dynspec(filename="/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/RawData/J0737-3039A_2019-07-20-08:38:56_ch5.0_sub5.0.ar.dynspec", process=False)#, lamsteps=True)
dyn5 = Dynspec(filename="/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/RawData/J0737-3039A_2019-07-20-09:09:12_ch5.0_sub5.0.ar.dynspec", process=False)#, lamsteps=True)

Fmax = np.inf
Fmin = 0
dyn1.crop_dyn(fmin=Fmin, fmax=Fmax)
# dyn1.zap()
dyn1.refill(linear=False)
dyn2.crop_dyn(fmin=Fmin, fmax=Fmax)
# dyn2.zap()
dyn2.refill(linear=False)
dyn3.crop_dyn(fmin=Fmin, fmax=Fmax)
# dyn2.zap()
dyn3.refill(linear=False)
dyn4.crop_dyn(fmin=Fmin, fmax=Fmax)
# dyn2.zap()
dyn4.refill(linear=False)
dyn5.crop_dyn(fmin=Fmin, fmax=Fmax)
# dyn2.zap()
dyn5.refill(linear=False)

dyn_tot = dyn1 + dyn2 + dyn3 + dyn4 + dyn5

# Removing any eclispe
start_mjd = dyn_tot.mjd
tobs = dyn_tot.tobs
times = dyn_tot.times
Eclipse_index = SearchEclipse(start_mjd, tobs, times)

# Plotting and Saving the Dynamic Spectra for this group #
dyn_tot.plot_dyn(filename=str(desktopdir) + str(dyn_tot.name.split('.')[0]) +
                 '_Dynspec.png')
dyn_tot.plot_dyn(filename=str(Spectradir) + str(dyn_tot.name.split('.')[0]) +
                 '_Dynspec.png')

# Collecting Results and par file and plotting cropped dynspec
modelling_results(analysis=False, freqmin=1630)

# Group 3 #
# Useable however the dnu is less than the observational bandwidth
dyn1 = Dynspec(filename="/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/RawData/J0737-3039A_2019-08-28-02:49:04_ch5.0_sub5.0.ar.dynspec", process=False)#, lamsteps=True)
dyn2 = Dynspec(filename="/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/RawData/J0737-3039A_2019-08-28-03:19:36_ch5.0_sub5.0.ar.dynspec", process=False)#, lamsteps=True)
dyn3 = Dynspec(filename="/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/RawData/J0737-3039A_2019-08-28-05:20:00_ch5.0_sub5.0.ar.dynspec", process=False)#, lamsteps=True)

Fmax = np.inf
Fmin = 0
dyn1.crop_dyn(fmin=Fmin, fmax=Fmax)
# dyn1.zap()
dyn1.refill(linear=False)
dyn2.crop_dyn(fmin=Fmin, fmax=Fmax)
# dyn2.zap()
dyn2.refill(linear=False)
dyn3.crop_dyn(fmin=Fmin, fmax=Fmax)
# dyn2.zap()
dyn3.refill(linear=False)

dyn_tot = dyn1 + dyn2 + dyn3

# Removing any eclispe #
start_mjd = dyn_tot.mjd
tobs = dyn_tot.tobs
times = dyn_tot.times
Eclipse_index = SearchEclipse(start_mjd, tobs, times)

# Interpolating // flagging the joints in the dynspec #
# The first joint:
dyn_tot.dyn[:, 226:229] = np.nan
dyn_tot.dyn[:, 225:230] = interp_nan_2d(dyn_tot.dyn[:, 225:230])
# The middle joint?:
# dyn_tot.dyn[:, 748] = np.median(dyn_tot.dyn)*10
dyn_tot.dyn[:, 748:749] = np.nan
dyn_tot.dyn[:, 747:750] = interp_nan_2d(dyn_tot.dyn[:, 747:750])
# The second joint:
dyn_tot.dyn[:, 1130:1132] = np.nan
dyn_tot.dyn[:, 1129:1133] = interp_nan_2d(dyn_tot.dyn[:, 1129:1133])

# Plotting and Saving the Dynamic Spectra for this group #
dyn_tot.plot_dyn(filename=str(desktopdir) + str(dyn_tot.name.split('.')[0]) +
                 '_Dynspec.png')
dyn_tot.plot_dyn(filename=str(Spectradir) + str(dyn_tot.name.split('.')[0]) +
                 '_Dynspec.png')

# Collecting Results and par file and plotting cropped dynspec
modelling_results(analysis=True, freqmin=0)  # 1630

# Group 4 #
# Useable however the dnu is lower than the observational frequency
dyn1 = Dynspec(filename="/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/RawData/J0737-3039A_2019-09-26-01:13:12_ch5.0_sub5.0.ar.dynspec", process=False)#, lamsteps=True)
dyn2 = Dynspec(filename="/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/RawData/J0737-3039A_2019-09-26-01:43:28_ch5.0_sub5.0.ar.dynspec", process=False)#, lamsteps=True)
dyn3 = Dynspec(filename="/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/RawData/J0737-3039A_2019-09-26-03:43:44_ch5.0_sub5.0.ar.dynspec", process=False)#, lamsteps=True)

Fmax = np.inf
Fmin = 0
dyn1.crop_dyn(fmin=Fmin, fmax=Fmax)
# dyn1.zap()
dyn1.refill(linear=False)
dyn2.crop_dyn(fmin=Fmin, fmax=Fmax)
# dyn2.zap()
dyn2.refill(linear=False)
dyn3.crop_dyn(fmin=Fmin, fmax=Fmax)
# dyn2.zap()
dyn3.refill(linear=False)

dyn_tot = dyn1 + dyn2 + dyn3

# Removing any eclispe
start_mjd = dyn_tot.mjd
tobs = dyn_tot.tobs
times = dyn_tot.times
Eclipse_index = SearchEclipse(start_mjd, tobs, times)

# Interpolating // flagging the joints in the dynspec #
# The second joint:
dyn_tot.dyn[:, 1127:1129] = np.nan
dyn_tot.dyn[:, 1126:1130] = interp_nan_2d(dyn_tot.dyn[:, 1126:1130])

# Plotting and Saving the Dynamic Spectra for this group #
dyn_tot.plot_dyn(filename=str(desktopdir) + str(dyn_tot.name.split('.')[0]) +
                 '_Dynspec.png')
dyn_tot.plot_dyn(filename=str(Spectradir) + str(dyn_tot.name.split('.')[0]) +
                 '_Dynspec.png')

# Collecting Results and par file and plotting cropped dynspec
modelling_results(analysis=True, freqmin=0)  # 1630

# Group 5 #
# I went as low as I am willing to go, still many observations dnu below df
dyn1 = Dynspec(filename="/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/RawData/J0737-3039A_2019-10-27-05:47:12_ch5.0_sub5.0.ar.dynspec", process=False)#, lamsteps=True)
dyn2 = Dynspec(filename="/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/RawData/J0737-3039A_2019-10-27-06:17:28_ch5.0_sub5.0.ar.dynspec", process=False)#, lamsteps=True)
dyn3 = Dynspec(filename="/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/RawData/J0737-3039A_2019-10-27-08:17:52_ch5.0_sub5.0.ar.dynspec", process=False)#, lamsteps=True)

Fmax = np.inf
Fmin = 0
dyn1.crop_dyn(fmin=Fmin, fmax=Fmax)
# dyn1.zap()
dyn1.refill(linear=False)
dyn2.crop_dyn(fmin=Fmin, fmax=Fmax)
# dyn2.zap()
dyn2.refill(linear=False)
dyn3.crop_dyn(fmin=Fmin, fmax=Fmax)
# dyn2.zap()
dyn3.refill(linear=False)

dyn_tot = dyn1 + dyn2 + dyn3

# Removing any eclispe
start_mjd = dyn_tot.mjd
tobs = dyn_tot.tobs
times = dyn_tot.times
Eclipse_index = SearchEclipse(start_mjd, tobs, times)

# Plotting and Saving the Dynamic Spectra for this group #
dyn_tot.plot_dyn(filename=str(desktopdir) + str(dyn_tot.name.split('.')[0]) +
                 '_Dynspec.png')
dyn_tot.plot_dyn(filename=str(Spectradir) + str(dyn_tot.name.split('.')[0]) +
                 '_Dynspec.png')

# Collecting Results and par file and plotting cropped dynspec
modelling_results(analysis=True, freqmin=0)  # 1630

# Group 6 # Has an effect similiar to that of an eclipse but is not an eclipse
# Went slightly below df but can get good measurements going to low freqmin
dyn1 = Dynspec(filename="/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/RawData/J0737-3039A_2019-11-25-23:42:24_ch5.0_sub5.0.ar.dynspec", process=False)#, lamsteps=True)
dyn2 = Dynspec(filename="/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/RawData/J0737-3039A_2019-11-26-00:12:32_ch5.0_sub5.0.ar.dynspec", process=False)#, lamsteps=True)
dyn3 = Dynspec(filename="/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/RawData/J0737-3039A_2019-11-26-02:12:48_ch5.0_sub5.0.ar.dynspec", process=False)#, lamsteps=True)

Fmax = np.inf
Fmin = 0
dyn1.crop_dyn(fmin=Fmin, fmax=Fmax)
# dyn1.zap()
dyn1.refill(linear=False)
dyn2.crop_dyn(fmin=Fmin, fmax=Fmax)
# dyn2.zap()
dyn2.refill(linear=False)
dyn3.crop_dyn(fmin=Fmin, fmax=Fmax)
# dyn2.zap()
dyn3.refill(linear=False)

dyn_tot = dyn1 + dyn2 + dyn3

# Removing any eclispe
start_mjd = dyn_tot.mjd
tobs = dyn_tot.tobs
times = dyn_tot.times
Eclipse_index = SearchEclipse(start_mjd, tobs, times)

# Interpolating // flagging the joints in the dynspec #
# The first joint:
dyn_tot.dyn[:, 225:227] = np.nan
dyn_tot.dyn[:, 224:228] = interp_nan_2d(dyn_tot.dyn[:, 224:228])

# Plotting and Saving the Dynamic Spectra for this group #
dyn_tot.plot_dyn(filename=str(desktopdir) + str(dyn_tot.name.split('.')[0]) +
                 '_Dynspec.png')
dyn_tot.plot_dyn(filename=str(Spectradir) + str(dyn_tot.name.split('.')[0]) +
                 '_Dynspec.png')

# Collecting Results and par file and plotting cropped dynspec
modelling_results(analysis=True, freqmin=0)  # 1430

# Group 7 # Central region of RFI flagged out channels
# Great measurements the majority of which are above df
dyn1 = Dynspec(filename="/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/RawData/J0737-3039A_2019-12-13-21:20:40_ch5.0_sub5.0.ar.dynspec", process=False)#, lamsteps=True)
dyn2 = Dynspec(filename="/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/RawData/J0737-3039A_2019-12-13-21:50:56_ch5.0_sub5.0.ar.dynspec", process=False)#, lamsteps=True)
dyn3 = Dynspec(filename="/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/RawData/J0737-3039A_2019-12-13-23:51:12_ch5.0_sub5.0.ar.dynspec", process=False)#, lamsteps=True)

Fmax = np.inf
Fmin = 0
dyn1.crop_dyn(fmin=Fmin, fmax=Fmax)
# dyn1.zap()
dyn1.refill(linear=False)
dyn2.crop_dyn(fmin=Fmin, fmax=Fmax)
# dyn2.zap()
dyn2.refill(linear=False)
dyn3.crop_dyn(fmin=Fmin, fmax=Fmax)
# dyn2.zap()
dyn3.refill(linear=False)

dyn_tot = dyn1 + dyn2 + dyn3

# The second joint: flagging first due to eclipse interference
dyn_tot.dyn[:, 1127:1129] = np.nan
dyn_tot.dyn[:, 1126:1130] = interp_nan_2d(dyn_tot.dyn[:, 1126:1130])

# Removing any eclispe
start_mjd = dyn_tot.mjd
tobs = dyn_tot.tobs
times = dyn_tot.times
Eclipse_index = SearchEclipse(start_mjd, tobs, times)

# Interpolating // flagging the joints in the dynspec #
# The first joint:
dyn_tot.dyn[:, 225:227] = np.nan
dyn_tot.dyn[:, 224:228] = interp_nan_2d(dyn_tot.dyn[:, 224:228])

# Plotting and Saving the Dynamic Spectra for this group #
dyn_tot.plot_dyn(filename=str(desktopdir) + str(dyn_tot.name.split('.')[0]) +
                 '_Dynspec.png')
dyn_tot.plot_dyn(filename=str(Spectradir) + str(dyn_tot.name.split('.')[0]) +
                 '_Dynspec.png')

# Collecting Results and par file and plotting cropped dynspec
modelling_results(analysis=True, freqmin=0)  # 1300

# Group 8 #
# This is pretty decent observation going down to pretty low frequency
# Anomaly @ 50min?
dyn1 = Dynspec(filename="/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/RawData/J0737-3039A_2019-12-14-19:13:52_ch5.0_sub5.0.ar.dynspec", process=False)#, lamsteps=True)
dyn2 = Dynspec(filename="/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/RawData/J0737-3039A_2019-12-14-19:44:08_ch5.0_sub5.0.ar.dynspec", process=False)#, lamsteps=True)
dyn3 = Dynspec(filename="/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/RawData/J0737-3039A_2019-12-14-21:44:24_ch5.0_sub5.0.ar.dynspec", process=False)#, lamsteps=True)

Fmax = np.inf
Fmin = 0
dyn1.crop_dyn(fmin=Fmin, fmax=Fmax)
# dyn1.zap()
dyn1.refill(linear=False)
dyn2.crop_dyn(fmin=Fmin, fmax=Fmax)
# dyn2.zap()
dyn2.refill(linear=False)
dyn3.crop_dyn(fmin=Fmin, fmax=Fmax)
# dyn2.zap()
dyn3.refill(linear=False)

dyn_tot = dyn1 + dyn2 + dyn3

# Removing any eclispe
start_mjd = dyn_tot.mjd
tobs = dyn_tot.tobs
times = dyn_tot.times
Eclipse_index = SearchEclipse(start_mjd, tobs, times)

# Interpolating // flagging the joints in the dynspec #
# The first joint:
dyn_tot.dyn[:, 226:228] = np.nan
dyn_tot.dyn[:, 225:229] = interp_nan_2d(dyn_tot.dyn[:, 225:229])
# The second joint: flagging first due to eclipse interference
dyn_tot.dyn[:, 1128:1130] = np.nan
dyn_tot.dyn[:, 1127:1131] = interp_nan_2d(dyn_tot.dyn[:, 1127:1131])

# Plotting and Saving the Dynamic Spectra for this group #
dyn_tot.plot_dyn(filename=str(desktopdir) + str(dyn_tot.name.split('.')[0]) +
                 '_Dynspec.png')
dyn_tot.plot_dyn(filename=str(Spectradir) + str(dyn_tot.name.split('.')[0]) +
                 '_Dynspec.png')
# Collecting Results and par file and plotting cropped dynspec
modelling_results(analysis=True, freqmin=0)  # 1310

# Group 9 # dyn1 is a loner!
# Good measurements above df
# dyn1 = Dynspec(filename="/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/RawData/J0737-3039A_2020-01-21-22:42:16_ch5.0_sub5.0.ar.dynspec", process=False)#, lamsteps=True)
dyn2 = Dynspec(filename="/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/RawData/J0737-3039A_2020-01-23-16:19:52_ch5.0_sub5.0.ar.dynspec", process=False)#, lamsteps=True)
dyn3 = Dynspec(filename="/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/RawData/J0737-3039A_2020-01-23-16:50:08_ch5.0_sub5.0.ar.dynspec", process=False)#, lamsteps=True)
dyn4 = Dynspec(filename="/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/RawData/J0737-3039A_2020-01-23-18:50:24_ch5.0_sub5.0.ar.dynspec", process=False)#, lamsteps=True)

Fmax = np.inf
Fmin = 0
# dyn1.crop_dyn(fmin=Fmin, fmax=Fmax)
# # dyn1.zap()
# dyn1.refill(linear=False)
dyn2.crop_dyn(fmin=Fmin, fmax=Fmax)
# dyn2.zap()
dyn2.refill(linear=False)
dyn3.crop_dyn(fmin=Fmin, fmax=Fmax)
# dyn2.zap()
dyn3.refill(linear=False)
dyn4.crop_dyn(fmin=Fmin, fmax=Fmax)
# dyn2.zap()
dyn4.refill(linear=False)

dyn_tot = dyn2 + dyn3 + dyn4

# Removing any eclispe
start_mjd = dyn_tot.mjd
tobs = dyn_tot.tobs
times = dyn_tot.times
Eclipse_index = SearchEclipse(start_mjd, tobs, times)

# Interpolating // flagging the joints in the dynspec #
# The first joint:
# dyn_tot.dyn[:, 226:228] = np.nan
# dyn_tot.dyn[:, 225:229] = interp_nan_2d(dyn_tot.dyn[:, 225:229])
# The second joint: flagging first due to eclipse interference
dyn_tot.dyn[:, 1126:1127] = np.nan
dyn_tot.dyn[:, 1125:1128] = interp_nan_2d(dyn_tot.dyn[:, 1125:1128])

# Plotting and Saving the Dynamic Spectra for this group #
dyn_tot.plot_dyn(filename=str(desktopdir) + str(dyn_tot.name.split('.')[0]) +
                 '_Dynspec.png')
dyn_tot.plot_dyn(filename=str(Spectradir) + str(dyn_tot.name.split('.')[0]) +
                 '_Dynspec.png')

# Collecting Results and par file and plotting cropped dynspec
modelling_results(analysis=True, freqmin=0)  # 1300

# Group 10 # dyn3 is a loner! # Pretty heavy RFI and artefact as in group 6
# Uncertain measurements well above df
dyn1 = Dynspec(filename="/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/RawData/J0737-3039A_2020-02-21-19:42:48_ch5.0_sub5.0.ar.dynspec", process=False)#, lamsteps=True)
dyn2 = Dynspec(filename="/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/RawData/J0737-3039A_2020-02-21-20:13:12_ch5.0_sub5.0.ar.dynspec", process=False)#, lamsteps=True)
# dyn3 = Dynspec(filename="/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/RawData/J0737-3039A_2020-03-20-20:18:32_ch5.0_sub5.0.ar.dynspec", process=False)#, lamsteps=True)

Fmax = np.inf
Fmin = 875
dyn1.crop_dyn(fmin=Fmin, fmax=Fmax)
# dyn1.zap()
dyn1.refill(linear=False)
dyn2.crop_dyn(fmin=Fmin, fmax=Fmax)
# dyn2.zap()
dyn2.refill(linear=False)
# dyn3.crop_dyn(fmin=Fmin, fmax=Fmax)
# # dyn2.zap()
# dyn3.refill(linear=False)

dyn_tot = dyn1 + dyn2

# Removing any eclispe
start_mjd = dyn_tot.mjd
tobs = dyn_tot.tobs
times = dyn_tot.times
Eclipse_index = SearchEclipse(start_mjd, tobs, times)

# Interpolating // flagging the joints in the dynspec #
# The first joint:
dyn_tot.dyn[:, 224:228] = np.nan
dyn_tot.dyn[:, 223:229] = interp_nan_2d(dyn_tot.dyn[:, 223:229])
# The second joint: flagging first due to eclipse interference
# dyn_tot.dyn[:, 1126:1127] = np.nan
# dyn_tot.dyn[:, 1125:1128] = interp_nan_2d(dyn_tot.dyn[:, 1125:1128])

# Plotting and Saving the Dynamic Spectra for this group #
dyn_tot.plot_dyn(filename=str(desktopdir) + str(dyn_tot.name.split('.')[0]) +
                 '_Dynspec.png')
dyn_tot.plot_dyn(filename=str(Spectradir) + str(dyn_tot.name.split('.')[0]) +
                 '_Dynspec.png')

# Collecting Results and par file and plotting cropped dynspec
modelling_results(analysis=True, freqmin=0)  # 1300

# # Group 11 # UHF !!
# dyn1 = Dynspec(filename="/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/RawData/J0737-3039A_2020-03-28-12:16:24_ch5.0_sub5.0.ar.dynspec", process=False)#, lamsteps=True)
# dyn2 = Dynspec(filename="/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/RawData/J0737-3039A_2020-03-28-12:39:12_ch5.0_sub5.0.ar.dynspec", process=False)#, lamsteps=True)
# dyn3 = Dynspec(filename="/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/RawData/J0737-3039A_2020-03-28-14:46:08_ch5.0_sub5.0.ar.dynspec", process=False)#, lamsteps=True)

# Fmax = np.inf
# Fmin = 0
# dyn1.crop_dyn(fmin=Fmin, fmax=Fmax)
# # dyn1.zap()
# dyn1.refill(linear=False)
# dyn2.crop_dyn(fmin=Fmin, fmax=Fmax)
# # dyn2.zap()
# dyn2.refill(linear=False)
# dyn3.crop_dyn(fmin=Fmin, fmax=Fmax)
# # dyn2.zap()
# dyn3.refill(linear=False)

# dyn_tot = dyn1 + dyn2 + dyn3

# dyn_tot.plot_dyn(filename='/Users/jacobaskew/Desktop/DynSpec.png')

# # Group 12 # UHF !!!
# dyn1 = Dynspec(filename="/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/RawData/J0737-3039A_2020-04-27-16:10:16_ch5.0_sub5.0.ar.dynspec", process=False)#, lamsteps=True)
# dyn2 = Dynspec(filename="/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/RawData/J0737-3039A_2020-04-27-16:40:40_ch5.0_sub5.0.ar.dynspec", process=False)#, lamsteps=True)
# dyn3 = Dynspec(filename="/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/RawData/J0737-3039A_2020-04-27-18:40:48_ch5.0_sub5.0.ar.dynspec", process=False)#, lamsteps=True)

# Fmax = np.inf
# Fmin = 0
# dyn1.crop_dyn(fmin=Fmin, fmax=Fmax)
# # dyn1.zap()
# dyn1.refill(linear=False)
# dyn2.crop_dyn(fmin=Fmin, fmax=Fmax)
# # dyn2.zap()
# dyn2.refill(linear=False)
# dyn3.crop_dyn(fmin=Fmin, fmax=Fmax)
# # dyn2.zap()
# dyn3.refill(linear=False)

# dyn_tot = dyn1 + dyn2 + dyn3

# dyn_tot.plot_dyn(filename='/Users/jacobaskew/Desktop/DynSpec.png')

# Group 13 # UHF !!!
# dyn1 = Dynspec(filename="/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/RawData/J0737-3039A_2020-05-30-09:28:32_ch5.0_sub5.0.ar.dynspec", process=False)#, lamsteps=True)
# dyn2 = Dynspec(filename="/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/RawData/J0737-3039A_2020-05-30-09:58:32_ch5.0_sub5.0.ar.dynspec", process=False)#, lamsteps=True)
# dyn3 = Dynspec(filename="/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/RawData/J0737-3039A_2020-05-30-11:58:48_ch5.0_sub5.0.ar.dynspec", process=False)#, lamsteps=True)

# Fmax = np.inf
# Fmin = 0
# dyn1.crop_dyn(fmin=Fmin, fmax=Fmax)
# # dyn1.zap()
# dyn1.refill(linear=False)
# dyn2.crop_dyn(fmin=Fmin, fmax=Fmax)
# # dyn2.zap()
# dyn2.refill(linear=False)
# dyn3.crop_dyn(fmin=Fmin, fmax=Fmax)
# # dyn2.zap()
# dyn3.refill(linear=False)

# dyn_tot = dyn1 + dyn2 + dyn3

# dyn_tot.plot_dyn(filename='/Users/jacobaskew/Desktop/DynSpec.png')

# Group 14 # UHF !!!
# dyn1 = Dynspec(filename="/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/RawData/J0737-3039A_2020-06-27-12:21:36_ch5.0_sub5.0.ar.dynspec", process=False)#, lamsteps=True)
# dyn2 = Dynspec(filename="/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/RawData/J0737-3039A_2020-06-27-12:51:36_ch5.0_sub5.0.ar.dynspec", process=False)#, lamsteps=True)
# dyn3 = Dynspec(filename="/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/RawData/J0737-3039A_2020-06-27-14:51:44_ch5.0_sub5.0.ar.dynspec", process=False)#, lamsteps=True)

# Fmax = np.inf
# Fmin = 0
# dyn1.crop_dyn(fmin=Fmin, fmax=Fmax)
# # dyn1.zap()
# dyn1.refill(linear=False)
# dyn2.crop_dyn(fmin=Fmin, fmax=Fmax)
# # dyn2.zap()
# dyn2.refill(linear=False)
# dyn3.crop_dyn(fmin=Fmin, fmax=Fmax)
# # dyn2.zap()
# dyn3.refill(linear=False)

# dyn_tot = dyn1 + dyn2 + dyn3

# dyn_tot.plot_dyn(filename='/Users/jacobaskew/Desktop/DynSpec.png')

# Group 15 #
# Much better when you consider more of the band, good measurements
dyn1 = Dynspec(filename="/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/RawData/J0737-3039A_2020-07-28-04:33:04_ch5.0_sub5.0.ar.dynspec", process=False)#, lamsteps=True)
dyn2 = Dynspec(filename="/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/RawData/J0737-3039A_2020-07-28-05:03:12_ch5.0_sub5.0.ar.dynspec", process=False)#, lamsteps=True)
dyn3 = Dynspec(filename="/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/RawData/J0737-3039A_2020-07-28-07:03:20_ch5.0_sub5.0.ar.dynspec", process=False)#, lamsteps=True)

Fmax = 1690
Fmin = 875
dyn1.crop_dyn(fmin=Fmin, fmax=Fmax)
# dyn1.zap()
dyn1.refill(linear=False)
dyn2.crop_dyn(fmin=Fmin, fmax=Fmax)
# dyn2.zap()
dyn2.refill(linear=False)
dyn3.crop_dyn(fmin=Fmin, fmax=Fmax)
# dyn2.zap()
dyn3.refill(linear=False)

dyn_tot = dyn1 + dyn2 + dyn3

# Removing any eclispe
start_mjd = dyn_tot.mjd
tobs = dyn_tot.tobs
times = dyn_tot.times
Eclipse_index = SearchEclipse(start_mjd, tobs, times)

# Interpolating // flagging the joints in the dynspec #
# The first joint:
dyn_tot.dyn[:, 357:358] = np.nan
dyn_tot.dyn[:, 356:359] = interp_nan_2d(dyn_tot.dyn[:, 356:359])
# The second joint: flagging first due to eclipse interference
dyn_tot.dyn[:, 1126:1127] = np.nan
dyn_tot.dyn[:, 1125:1128] = interp_nan_2d(dyn_tot.dyn[:, 1125:1128])

# Plotting and Saving the Dynamic Spectra for this group #
dyn_tot.plot_dyn(filename=str(desktopdir) + str(dyn_tot.name.split('.')[0]) +
                 '_Dynspec.png')
dyn_tot.plot_dyn(filename=str(Spectradir) + str(dyn_tot.name.split('.')[0]) +
                 '_Dynspec.png')

# Collecting Results and par file and plotting cropped dynspec
modelling_results(analysis=True, freqmin=0)  # 1300

# Group 16 # UHF !!!
# dyn1 = Dynspec(filename="/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/RawData/J0737-3039A_2020-08-30-05:10:16_ch5.0_sub5.0.ar.dynspec", process=False)#, lamsteps=True)
# dyn2 = Dynspec(filename="/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/RawData/J0737-3039A_2020-08-30-05:40:24_ch5.0_sub5.0.ar.dynspec", process=False)#, lamsteps=True)
# dyn3 = Dynspec(filename="/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/RawData/J0737-3039A_2020-08-30-07:41:36_ch5.0_sub5.0.ar.dynspec", process=False)#, lamsteps=True)

# Fmax = np.inf
# Fmin = 0
# dyn1.crop_dyn(fmin=Fmin, fmax=Fmax)
# # dyn1.zap()
# dyn1.refill(linear=False)
# dyn2.crop_dyn(fmin=Fmin, fmax=Fmax)
# # dyn2.zap()
# dyn2.refill(linear=False)
# dyn3.crop_dyn(fmin=Fmin, fmax=Fmax)
# # dyn2.zap()
# dyn3.refill(linear=False)

# dyn_tot = dyn1 + dyn2 + dyn3

# dyn_tot.plot_dyn(filename='/Users/jacobaskew/Desktop/DynSpec.png')

# Group 17 # UHF !!!
# dyn1 = Dynspec(filename="/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/RawData/J0737-3039A_2020-10-14-02:26:24_ch5.0_sub5.0.ar.dynspec", process=False)#, lamsteps=True)
# dyn2 = Dynspec(filename="/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/RawData/J0737-3039A_2020-10-14-02:56:32_ch5.0_sub5.0.ar.dynspec", process=False)#, lamsteps=True)
# dyn3 = Dynspec(filename="/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/RawData/J0737-3039A_2020-10-14-04:56:32_ch5.0_sub5.0.ar.dynspec", process=False)#, lamsteps=True)

# Fmax = np.inf
# Fmin = 0
# dyn1.crop_dyn(fmin=Fmin, fmax=Fmax)
# # dyn1.zap()
# dyn1.refill(linear=False)
# dyn2.crop_dyn(fmin=Fmin, fmax=Fmax)
# # dyn2.zap()
# dyn2.refill(linear=False)
# dyn3.crop_dyn(fmin=Fmin, fmax=Fmax)
# # dyn2.zap()
# dyn3.refill(linear=False)

# dyn_tot = dyn1 + dyn2 + dyn3

# dyn_tot.plot_dyn(filename='/Users/jacobaskew/Desktop/DynSpec.png')

# Group 18 # UHF !!!
# dyn1 = Dynspec(filename="/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/RawData/J0737-3039A_2020-11-21-00:48:48_ch5.0_sub5.0.ar.dynspec", process=False)#, lamsteps=True)
# dyn2 = Dynspec(filename="/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/RawData/J0737-3039A_2020-11-21-01:18:56_ch5.0_sub5.0.ar.dynspec", process=False)#, lamsteps=True)
# dyn3 = Dynspec(filename="/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/RawData/J0737-3039A_2020-11-21-03:18:56_ch5.0_sub5.0.ar.dynspec", process=False)#, lamsteps=True)

# Fmax = np.inf
# Fmin = 0
# dyn1.crop_dyn(fmin=Fmin, fmax=Fmax)
# # dyn1.zap()
# dyn1.refill(linear=False)
# dyn2.crop_dyn(fmin=Fmin, fmax=Fmax)
# # dyn2.zap()
# dyn2.refill(linear=False)
# dyn3.crop_dyn(fmin=Fmin, fmax=Fmax)
# # dyn2.zap()
# dyn3.refill(linear=False)

# dyn_tot = dyn1 + dyn2 + dyn3

# dyn_tot.plot_dyn(filename='/Users/jacobaskew/Desktop/DynSpec.png')

# Group 19 #
dyn1 = Dynspec(filename="/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/RawData/J0737-3039A_2020-12-22-02:45:28_ch5.0_sub5.0.ar.dynspec", process=False)#, lamsteps=True)
dyn2 = Dynspec(filename="/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/RawData/J0737-3039A_2020-12-22-03:15:28_ch5.0_sub5.0.ar.dynspec", process=False)#, lamsteps=True)
dyn3 = Dynspec(filename="/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/RawData/J0737-3039A_2020-12-22-05:15:36_ch5.0_sub5.0.ar.dynspec", process=False)#, lamsteps=True)

Fmax = 1700
Fmin = 875
dyn1.crop_dyn(fmin=Fmin, fmax=Fmax)
# dyn1.zap()
dyn1.refill(linear=False)
dyn2.crop_dyn(fmin=Fmin, fmax=Fmax)
# dyn2.zap()
dyn2.refill(linear=False)
dyn3.crop_dyn(fmin=Fmin, fmax=Fmax)
# dyn2.zap()
dyn3.refill(linear=False)

dyn_tot = dyn1 + dyn2 + dyn3

# Removing any eclispe
start_mjd = dyn_tot.mjd
tobs = dyn_tot.tobs
times = dyn_tot.times
Eclipse_index = SearchEclipse(start_mjd, tobs, times)

# Interpolating // flagging the joints in the dynspec #
# The first joint:
dyn_tot.dyn[:, 225:227] = np.nan
dyn_tot.dyn[:, 224:228] = interp_nan_2d(dyn_tot.dyn[:, 224:228])
# The second joint: flagging first due to eclipse interference
dyn_tot.dyn[:, 812:814] = np.nan
dyn_tot.dyn[:, 811:815] = interp_nan_2d(dyn_tot.dyn[:, 811:815])

# Plotting and Saving the Dynamic Spectra for this group #
dyn_tot.plot_dyn(filename=str(desktopdir) + str(dyn_tot.name.split('.')[0]) +
                 '_Dynspec.png')
dyn_tot.plot_dyn(filename=str(Spectradir) + str(dyn_tot.name.split('.')[0]) +
                 '_Dynspec.png')

# Collecting Results and par file and plotting cropped dynspec
modelling_results(analysis=True, freqmin=0)  # 1300

# Group 20 # UHF !!!
# dyn1 = Dynspec(filename="/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/RawData/J0737-3039A_2021-01-28-00:37:28_ch5.0_sub5.0.ar.dynspec", process=False)#, lamsteps=True)
# dyn2 = Dynspec(filename="/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/RawData/J0737-3039A_2021-01-28-01:07:36_ch5.0_sub5.0.ar.dynspec", process=False)#, lamsteps=True)
# dyn3 = Dynspec(filename="/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/RawData/J0737-3039A_2021-01-28-03:07:36_ch5.0_sub5.0.ar.dynspec", process=False)#, lamsteps=True)

# Fmax = np.inf
# Fmin = 0
# dyn1.crop_dyn(fmin=Fmin, fmax=Fmax)
# # dyn1.zap()
# dyn1.refill(linear=False)
# dyn2.crop_dyn(fmin=Fmin, fmax=Fmax)
# # dyn2.zap()
# dyn2.refill(linear=False)
# dyn3.crop_dyn(fmin=Fmin, fmax=Fmax)
# # dyn2.zap()
# dyn3.refill(linear=False)

# dyn_tot = dyn1 + dyn2 + dyn3

# dyn_tot.plot_dyn(filename='/Users/jacobaskew/Desktop/DynSpec.png')

if plotting:
    modelling_results(plot=True)
