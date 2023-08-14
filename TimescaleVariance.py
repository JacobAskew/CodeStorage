#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 12:27:10 2021

@author: jacobaskew
"""

# Modelling the scintillation bandwidth and timescale variations across
# a single observation in time

##############################################################################
# Common #
from scintools.dynspec import Dynspec
from scintools.scint_utils import write_results, read_results, read_par, \
        float_array_from_dict, scint_velocity
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import glob
from copy import deepcopy as cp

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
spectradir = '/Users/jacobaskew/Desktop/DoublePulsar_Project/' + str(pulsar) + '/Spectra/'
par_dir = '/Users/jacobaskew/Desktop/Swinburne_Daniel/ParFiles/'
plotdir = '/Users/jacobaskew/Desktop/DoublePulsar_Project/' + str(pulsar) + '/Plots/'
eclipsefile = '/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/Datafiles/Eclipse_mjd.txt'
outdir = '/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/Datafiles/' 
HighFreqDir = '/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/HighFreq/Plots/'
##############################################################################
# Also Common #
dynspecs = sorted(glob.glob(datadir + '/*ar.dynspec'))
outfiles = str(outdir) + str(psrname) + \
    '_ScintillationResults_TimescaleVariance_'
##############################################################################
# Manual Inputs #
measure = True
model = True
zap = False
linear = False
##############################################################################


def SearchEclipse(start_mjd, tobs):
    end_mjd = start_mjd + tobs/86400
    Eclipse_mjd = np.genfromtxt(eclipsefile, delimiter=',', encoding=None,
                                dtype=float)
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
if measure:

    for dynspec in dynspecs:
        
        File1 = dynspec.split(str(datadir))[1]
        Filename = str(File1.split('.')[0])
    
        dyn = Dynspec(filename=dynspec, process=False)
        dyn.trim_edges()
        start_mjd = dyn.mjd
        tobs = dyn.tobs
        Eclipse_index = SearchEclipse(start_mjd, tobs)
        if Eclipse_index is not None:
            for i in range(0, len(Eclipse_index)):
                dyn.dyn[:, Eclipse_index[i]-3:Eclipse_index[i]+3] = 0
        dyn.plot_dyn(filename=str(spectradir) + str(Filename) + '_Spectra.png')
        #
        if dyn.freq < 1000:
            continue
        counter = 0
        outfile = outfiles + str(dyn.name) + ".txt"
        time_len = int((round(dyn.tobs, 0) - 10))
        dyn_crop = cp(dyn)
        dyn_crop.crop_dyn(fmin=1580, fmax=np.inf)
        for i in range(0, time_len):
            try:
                dyn = cp(dyn_crop)
                dyn.crop_dyn(fmin=1580, fmax=np.inf, tmin=0+i, tmax=10+i)
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
                dyn.plot_dyn()
                write_results(outfile, dyn=dyn)
            except Exception as e:
                print(e)
                print("THIS FILE DIDN'T WORK")
                counter += 1
                continue

        print("this observation " + str(counter) + " dynspecs were not included")
    
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
        scintle_num = float_array_from_dict(params, 'scintle_num')
        tobs = float_array_from_dict(params, 'tobs')  # tobs in second
        rcvrs = np.array([rcvr[0] for rcvr in params['name']])
        acf_tilt = float_array_from_dict(params, 'acf_tilt')
        acf_tilt_err = float_array_from_dict(params, 'acf_tilt_err')
    
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
    
        """
        Do corrections!
        """
    
        indicies = np.argwhere((tauerr < 0.5*tau) * (dnuerr < 0.5*dnu))
    
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
    
        # Make MJD centre of observation, instead of start
        mjd = mjd + tobs/86400/2
        mjd_min = (mjd*(60*24))
        mjd_min = mjd_min - mjd_min[0]
    
        # Form Viss from the data
        Aiss = 2.78*10**4  # thin screen, table 2 of Cordes & Rickett (1998)
        D = 1  # kpc
        ind_low = np.argwhere((freq < 1100))
    
        viss, visserr = scint_velocity(None, dnu, tau, freq, dnuerr, tauerr,
                                       a=Aiss)
        Font = 35
        Size = 80*np.pi  # Determines the size of the datapoints used
        font = {'size': 32}
        matplotlib.rc('font', **font)
    
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(1, 1, 1)
        plt.scatter(mjd_min, tau, c='C3', alpha=0.6, s=Size)
        plt.errorbar(mjd_min, tau, yerr=tauerr, fmt=' ', ecolor='red',
                     alpha=0.4, elinewidth=5)
        plt.xlabel('Time (mins)')
        plt.ylabel('Scintillation Timescale (mins)')
        plt.title(psrname + ' Timescale v Time')
        plt.savefig(plotdir + str(dyn.name.split('.')[0]) + "_" +
                    "Tau_TimeSeries.png")
        plt.show()
        plt.close()
    
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
        plt.title(psrname + ' Scintillation Bandwidth')
        plt.savefig(plotdir + str(dyn.name.split('.')[0]) + "_" +
                    "Dnu_TimeSeries.png")
        plt.show()
        plt.close()

        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(1, 1, 1)
        plt.scatter(mjd_min, acf_tilt, c='C4', alpha=0.6, s=Size)
        plt.errorbar(mjd_min, acf_tilt, yerr=acf_tilt_err, fmt=' ',
                     ecolor='C4', alpha=0.4, elinewidth=5)
        xl = plt.xlim()
        plt.xlabel('Time (mins)')
        plt.ylabel('ACF Tilt (?)')
        plt.xlim(xl)
        plt.title(psrname + ' Scintillation Bandwidth')
        plt.savefig(plotdir + str(dyn.name.split('.')[0]) + "_" +
                    "Tilt_Timeseries.png")
        plt.show()
        plt.close()

        norm_dnu = (dnu-np.min(dnu))/(np.max(dnu) - np.min(dnu))
        norm_tau = (tau-np.min(tau))/(np.max(tau) - np.min(tau))
        norm_tilt = (acf_tilt-np.min(acf_tilt))/(np.max(acf_tilt) - np.min(acf_tilt))

        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(1, 1, 1)
        plt.scatter(mjd_min, norm_dnu, c='C0', alpha=0.6, s=Size)
        plt.scatter(mjd_min, norm_tau, c='C3', alpha=0.6, s=Size)
        plt.scatter(mjd_min, norm_tilt, c='C4', alpha=0.6, s=Size)
        plt.xlabel('Time (mins)')
        plt.ylabel('Normalised Values')
        plt.title(psrname + ' Normalised Scintillation Results')
        plt.savefig(plotdir + str(dyn.name.split('.')[0]) + "_" +
                    "Norm_TimeSeries_Comparison.png")
        plt.show()
        plt.close()
    
        outfile = None
