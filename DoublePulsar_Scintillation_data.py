#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 14:49:52 2022

@author: jacobaskew
"""

# This script should only need processed .dynspec arcive files to work.

# Outputs should include pdfs of the dynamic spectra, before and after zapping
# with and without flagging out the eclipse.

# Also should include measurements of scintillation bandwidth and timescale for
# giving frequency and time chunks. I want a plot comparing all Lband-UHF and
# highRes lowRes data all being compared in one plot using the same method.

# Finally a power law fit comparing old and new data.
###############################################################################
# Importing neccessary things
from scintools.dynspec import Dynspec
from scintools.scint_utils import write_results, read_results, read_par, \
        float_array_from_dict, get_ssb_delay, get_earth_velocity, \
        get_true_anomaly, pars_to_params
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import glob
from copy import deepcopy as cp
import os
import shutil
from scintools.scint_sim import Simulation
from astropy.table import Table, vstack
###############################################################################
# This function is used as a slower version of FFT for areas with RFI
# Plot ACF, use get_scint_params.


def autocorr_func(data, dyn_name, ACFbindir, istart_t, outfile):
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
    dyn_name.acf = autocorr
    dyn_name.plot_acf(filename=str(ACFbindir)+str(round(istart_t, 1)) +
                      '/ACF_chunk_'+str(int(dyn_name.freq))+'.pdf',
                      input_acf=autocorr, contour=True)
    dyn_name.get_scint_params(display=False)
    dyn_name.write_results(outfile, dyn=dyn_name)
    return autocorr


##############################################################################
# This function aims to try one method to remove the effect of the eclipse
# this is based on the assumption that there is two eclispes in the groups
# of three observations and they are at the lowest flux in the observation


def remove_eclipse(start_mjd, tobs, dyn, fluxes):
    end_mjd = start_mjd + tobs/86400
    wd0 = '/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/'
    eclipsefile = wd0+'Datafiles/Eclipse_mjd.txt'
    Eclipse_mjd = np.genfromtxt(eclipsefile, delimiter=',',
                                encoding=None, dtype=float)
    Eclipse_events = np.array(np.where((Eclipse_mjd > start_mjd) *
                              (Eclipse_mjd < end_mjd)))
    # If the eclipse event list finds no eclipse across the mjd of observations
    if Eclipse_events.size == 0:
        median_flux_list = []
        for i in range(0, np.shape(fluxes)[1]):
            median_flux_list.append(np.median(fluxes[:, i]))
        median_fluxes = np.asarray(median_flux_list)
        Eclipse_index = int(np.argmin(median_fluxes))
        if Eclipse_index is not None:
            fluxes[:, Eclipse_index-3:Eclipse_index+3] = 0
            # dyn.refill(method='mean')
            print("Eclispe in dynspec")

        else:
            print("No Eclispe in dynspec")
    # If the list and mjd of our observations line up
    else:
        Eclipse_events_mjd = Eclipse_mjd[Eclipse_events]
        mjds = start_mjd + dyn.times/86400
        Eclipse_index = np.argmin(abs(mjds - Eclipse_events_mjd))
        if Eclipse_index is not None:
            print("Eclispe in dynspec")
            fluxes[:, Eclipse_index-3:Eclipse_index+3] = 0
            # dyn.refill(method='mean')
        else:
            print("No Eclispe in dynspec")
    return Eclipse_index


def plot_dynspec(correct_dynspec=False, zap=False, mean=True, linear=False,
                 median=False, overwrite=True, Observations=None):
    font = {'size': 28}
    matplotlib.rc('font', **font)
    #
    wd0 = '/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/'
    wd = wd0+'New/'
    datadir = wd + 'Dynspec/'
    outdir = wd + 'DataFiles/'
    dynspecs = sorted(glob.glob(datadir + '/*.dynspec'))
    #
    if Observations:
        observations = [Observations]
    else:
        observations = []
        for i in range(0, len(dynspecs)):
            observations.append((dynspecs[i].split(datadir)[1].split('-')[1] +
                                '-' +
                                dynspecs[i].split(datadir)[1].split('-')[2] +
                                '-' +
                                dynspecs[i].split(datadir)[1].split('-')[3]).split('3039A_')[1])
        observations = np.unique(np.asarray(observations))
    for i in range(0, len(observations)):
        observation_date = observations[i]+'/'
        observation_date2 = observations[i]
        spectradir2 = wd + 'SpectraPlots/' + observation_date2
        dynspecs = \
            sorted(glob.glob(datadir + 'J0737-3039A_' + str(observation_date.split('/')[0])
                             + '*.dynspec'))
        dynspecfile = \
            outdir+'DynspecPlotFiles/'+observation_date2 + \
            '_CompleteDynspec.dynspec'
        dynspecfile2 = \
            outdir+'DynspecPlotFiles/'+observation_date2 + \
            'Zap_CompleteDynspec.dynspec'
        dynspecfile3 = \
            outdir+'DynspecPlotFiles/'+observation_date2 + \
            'Zeroed_CompleteDynspec.dynspec'
        filedir = str(outdir)+str(observation_date)
        #
        try:
            os.mkdir(filedir)
        except OSError as error:
            print(error)
        if overwrite or not os.path.exists(dynspecfile2):
            print("overwriting old dynspec for", observation_date)
            if len(dynspecs) == 3:
                dyn1 = Dynspec(filename=dynspecs[0], process=False)
                remove_eclipse(dyn1.mjd, dyn1.tobs, dyn1, dyn1.dyn)
                dyn2 = Dynspec(filename=dynspecs[1], process=False)
                dyn3 = Dynspec(filename=dynspecs[2], process=False)
                remove_eclipse(dyn3.mjd, dyn3.tobs, dyn3, dyn3.dyn)
                dyn = dyn1 + dyn2 + dyn3
            elif len(dynspecs) == 2:
                dyn1 = Dynspec(filename=dynspecs[0], process=False)
                remove_eclipse(dyn1.mjd, dyn1.tobs, dyn1, dyn1.dyn)
                dyn2 = Dynspec(filename=dynspecs[1], process=False)
                remove_eclipse(dyn2.mjd, dyn2.tobs, dyn2, dyn2.dyn)
                dyn = dyn1 + dyn2
            else:
                dyn = Dynspec(filename=dynspecs[0], process=False)
                remove_eclipse(dyn, dyn.dyn)
        else:
            continue
        dyn.trim_edges()
        Fmax = np.max(dyn.freqs) - 48
        Fmin = np.min(dyn.freqs) + 48
        dyn.crop_dyn(fmin=Fmin, fmax=Fmax)
        # Save the spectra before zapping
        dyn.plot_dyn(filename=str(spectradir2)+'_Eclipse_FullSpectra.pdf',
                     dpi=400)
        dyn.write_file(filename=str(dynspecfile))
        if zap:
            dyn_crop = cp(dyn)
            len_min = dyn_crop.tobs/60
            len_min_chunk = len_min/20
            len_minimum = 0
            len_maximum = int(len_min_chunk)
            #
            dyn_crop.zap()
            try:
                dyn_crop1 = cp(dyn_crop)
                dyn_crop1.crop_dyn(tmin=0, tmax=len_maximum)
                dyn_crop1.refill(method='mean')
            except ValueError as e:
                print(e)
            try:
                len_minimum += int(len_min_chunk)
                len_maximum += int(len_min_chunk)
                dyn_crop2 = cp(dyn_crop)
                dyn_crop2.crop_dyn(tmin=len_minimum, tmax=len_maximum)
                dyn_crop2.refill(method='mean')
            except ValueError as e:
                print(e)
            try:
                len_minimum += int(len_min_chunk)
                len_maximum += int(len_min_chunk)
                dyn_crop3 = cp(dyn_crop)
                dyn_crop3.crop_dyn(tmin=len_minimum, tmax=len_maximum)
                dyn_crop3.refill(method='mean')
            except ValueError as e:
                print(e)
            try:
                len_minimum += int(len_min_chunk)
                len_maximum += int(len_min_chunk)
                dyn_crop4 = cp(dyn_crop)
                dyn_crop4.crop_dyn(tmin=len_minimum, tmax=len_maximum)
                dyn_crop4.refill(method='mean')
            except ValueError as e:
                print(e)
            try:
                len_minimum += int(len_min_chunk)
                len_maximum += int(len_min_chunk)
                dyn_crop5 = cp(dyn_crop)
                dyn_crop5.crop_dyn(tmin=len_minimum, tmax=len_maximum)
                dyn_crop5.refill(method='mean')
            except ValueError as e:
                print(e)
            try:
                len_minimum += int(len_min_chunk)
                len_maximum += int(len_min_chunk)
                dyn_crop6 = cp(dyn_crop)
                dyn_crop6.crop_dyn(tmin=len_minimum, tmax=len_maximum)
                dyn_crop6.refill(method='mean')
            except ValueError as e:
                print(e)
            try:
                len_minimum += int(len_min_chunk)
                len_maximum += int(len_min_chunk)
                dyn_crop7 = cp(dyn_crop)
                dyn_crop7.crop_dyn(tmin=len_minimum, tmax=len_maximum)
                dyn_crop7.refill(method='mean')
            except ValueError as e:
                print(e)
            try:
                len_minimum += int(len_min_chunk)
                len_maximum += int(len_min_chunk)
                dyn_crop8 = cp(dyn_crop)
                dyn_crop8.crop_dyn(tmin=len_minimum, tmax=len_maximum)
                dyn_crop8.refill(method='mean')
            except ValueError as e:
                print(e)
            try:
                len_minimum += int(len_min_chunk)
                len_maximum += int(len_min_chunk)
                dyn_crop9 = cp(dyn_crop)
                dyn_crop9.crop_dyn(tmin=len_minimum, tmax=len_maximum)
                dyn_crop9.refill(method='mean')
            except ValueError as e:
                print(e)
            try:
                len_minimum += int(len_min_chunk)
                len_maximum += int(len_min_chunk)
                dyn_crop10 = cp(dyn_crop)
                dyn_crop10.crop_dyn(tmin=len_minimum, tmax=len_maximum)
                dyn_crop10.refill(method='mean')
            except ValueError as e:
                print(e)
            try:
                len_minimum += int(len_min_chunk)
                len_maximum += int(len_min_chunk)
                dyn_crop11 = cp(dyn_crop)
                dyn_crop11.crop_dyn(tmin=len_minimum, tmax=len_maximum)
                dyn_crop11.refill(method='mean')
            except ValueError as e:
                print(e)
            try:
                len_minimum += int(len_min_chunk)
                len_maximum += int(len_min_chunk)
                dyn_crop12 = cp(dyn_crop)
                dyn_crop12.crop_dyn(tmin=len_minimum, tmax=len_maximum)
                dyn_crop12.refill(method='mean')
            except ValueError as e:
                print(e)
            try:
                len_minimum += int(len_min_chunk)
                len_maximum += int(len_min_chunk)
                dyn_crop13 = cp(dyn_crop)
                dyn_crop13.crop_dyn(tmin=len_minimum, tmax=len_maximum)
                dyn_crop13.refill(method='mean')
            except ValueError as e:
                print(e)
            try:
                len_minimum += int(len_min_chunk)
                len_maximum += int(len_min_chunk)
                dyn_crop14 = cp(dyn_crop)
                dyn_crop14.crop_dyn(tmin=len_minimum, tmax=len_maximum)
                dyn_crop14.refill(method='mean')
            except ValueError as e:
                print(e)
            try:
                len_minimum += int(len_min_chunk)
                len_maximum += int(len_min_chunk)
                dyn_crop15 = cp(dyn_crop)
                dyn_crop15.crop_dyn(tmin=len_minimum, tmax=len_maximum)
                dyn_crop15.refill(method='mean')
            except ValueError as e:
                print(e)
            try:
                len_minimum += int(len_min_chunk)
                len_maximum += int(len_min_chunk)
                dyn_crop16 = cp(dyn_crop)
                dyn_crop16.crop_dyn(tmin=len_minimum, tmax=len_maximum)
                dyn_crop16.refill(method='mean')
            except ValueError as e:
                print(e)
            try:
                len_minimum += int(len_min_chunk)
                len_maximum += int(len_min_chunk)
                dyn_crop17 = cp(dyn_crop)
                dyn_crop17.crop_dyn(tmin=len_minimum, tmax=len_maximum)
                dyn_crop17.refill(method='mean')
            except ValueError as e:
                print(e)
            try:

                len_minimum += int(len_min_chunk)
                len_maximum += int(len_min_chunk)
                dyn_crop18 = cp(dyn_crop)
                dyn_crop18.crop_dyn(tmin=len_minimum, tmax=len_maximum)
                dyn_crop18.refill(method='mean')
            except ValueError as e:
                print(e)
            try:

                len_minimum += int(len_min_chunk)
                len_maximum += int(len_min_chunk)
                dyn_crop19 = cp(dyn_crop)
                dyn_crop19.crop_dyn(tmin=len_minimum, tmax=len_maximum)
                dyn_crop19.refill(method='mean')
            except ValueError as e:
                print(e)
            try:

                len_minimum += int(len_min_chunk)
                len_maximum += int(len_min_chunk)
                dyn_crop20 = cp(dyn_crop)
                dyn_crop20.crop_dyn(tmin=len_minimum, tmax=np.inf)
                dyn_crop20.refill(method='mean')
            except ValueError as e:
                print(e)
            try:
                dyn_all = dyn_crop1 + dyn_crop2 + dyn_crop3 + dyn_crop4 + \
                    dyn_crop5 + dyn_crop6 + dyn_crop7 + dyn_crop8+dyn_crop9 + \
                    dyn_crop10 + dyn_crop11 + dyn_crop12 + dyn_crop13 + \
                    dyn_crop14 + dyn_crop15 + dyn_crop16 + dyn_crop17 + \
                    dyn_crop18 + dyn_crop19 + dyn_crop20
            except Exception as e:
                print(e)
            dyn_all.plot_dyn(filename=str(spectradir2) +
                             '_Zap_Eclipse_FullSpectra.pdf',
                             dpi=400)
            dyn_all.write_file(filename=str(dynspecfile2))
        if observation_date2 == '2022-06-30':
            bad_freq_low = 1027.5
            bad_freq_high = 1031.5
            bad_index_low = int((bad_freq_low - np.min(dyn.freqs)) /
                                (dyn.df))
            bad_index_high = int((bad_freq_high - np.min(dyn.freqs)) /
                                 (dyn.df))
            dyn.dyn[bad_index_low:bad_index_high, :] = 0
            bad_freq_low = 955
            bad_freq_high = 961
            bad_index_low = int((bad_freq_low - np.min(dyn.freqs)) /
                                (dyn.df))
            bad_index_high = int((bad_freq_high - np.min(dyn.freqs)) /
                                 (dyn.df))
            dyn.dyn[bad_index_low:bad_index_high, :] = 0
            bad_freq_low = 946
            bad_freq_high = 949
            bad_index_low = int((bad_freq_low - np.min(dyn.freqs)) /
                                (dyn.df))
            bad_index_high = int((bad_freq_high - np.min(dyn.freqs)) /
                                 (dyn.df))
            dyn.dyn[bad_index_low:bad_index_high, :] = 0
            bad_time_low = 28
            bad_time_high = 45
            bad_index_time_low = int((bad_time_low*60)/(dyn.dt))
            bad_index_time_high = int((bad_time_high*60)/(dyn.dt))
            dyn.dyn[:, bad_index_time_low:bad_index_time_high] = 0
            bad_time_low = 149.5
            bad_time_high = 150.5
            bad_index_time_low = int((bad_time_low*60)/(dyn.dt))
            bad_index_time_high = int((bad_time_high*60)/(dyn.dt))
            dyn.dyn[:, bad_index_time_low:bad_index_time_high] = 0
        elif observation_date2 == '2022-07-30':
            bad_freq_low = 1028
            bad_freq_high = 1031
            bad_index_low = int((bad_freq_low - np.min(dyn.freqs)) /
                                (dyn.df))
            bad_index_high = int((bad_freq_high - np.min(dyn.freqs)) /
                                 (dyn.df))
            dyn.dyn[bad_index_low:bad_index_high, :] = 0
            bad_freq_low = 955
            bad_freq_high = 960
            bad_index_low = int((bad_freq_low - np.min(dyn.freqs)) /
                                (dyn.df))
            bad_index_high = int((bad_freq_high - np.min(dyn.freqs)) /
                                 (dyn.df))
            dyn.dyn[bad_index_low:bad_index_high, :] = 0
            bad_time_low = 150
            bad_time_high = 151
            bad_index_time_low = int((bad_time_low*60)/(dyn.dt))
            bad_index_time_high = int((bad_time_high*60)/(dyn.dt))
            dyn.dyn[:, bad_index_time_low:bad_index_time_high] = 0
            #
            bad_time_low = 17.5
            bad_time_high = 18.5
            bad_index_time_low = int((bad_time_low*60)/(dyn.dt))
            bad_index_time_high = int((bad_time_high*60)/(dyn.dt))
            dyn.dyn[:, bad_index_time_low:bad_index_time_high] = 0
            #
            bad_time_low = 30
            bad_time_high = 30.5
            bad_index_time_low = int((bad_time_low*60)/(dyn.dt))
            bad_index_time_high = int((bad_time_high*60)/(dyn.dt))
            dyn.dyn[:, bad_index_time_low:bad_index_time_high] = 0
            #
            bad_time_low = 150.25
            bad_time_high = 151
            bad_index_time_low = int((bad_time_low*60)/(dyn.dt))
            bad_index_time_high = int((bad_time_high*60)/(dyn.dt))
            dyn.dyn[:, bad_index_time_low:bad_index_time_high] = 0
            #
            bad_time_low = 165
            bad_time_high = 166
            bad_index_time_low = int((bad_time_low*60)/(dyn.dt))
            bad_index_time_high = int((bad_time_high*60)/(dyn.dt))
            dyn.dyn[:, bad_index_time_low:bad_index_time_high] = 0
        elif observation_date2 == '2022-08-27':
            bad_freq_low = 1028.5
            bad_freq_high = 1031
            bad_index_low = int((bad_freq_low - np.min(dyn.freqs)) /
                                (dyn.df))
            bad_index_high = int((bad_freq_high - np.min(dyn.freqs)) /
                                 (dyn.df))
            dyn.dyn[bad_index_low:bad_index_high, :] = 0
            bad_freq_low = 925
            bad_freq_high = 960
            bad_index_low = int((bad_freq_low - np.min(dyn.freqs)) /
                                (dyn.df))
            bad_index_high = int((bad_freq_high - np.min(dyn.freqs)) /
                                 (dyn.df))
            dyn.dyn[bad_index_low:bad_index_high, :] = 0
            bad_freq_low = 767
            bad_freq_high = 780
            bad_index_low = int((bad_freq_low - np.min(dyn.freqs)) /
                                (dyn.df))
            bad_index_high = int((bad_freq_high - np.min(dyn.freqs)) /
                                 (dyn.df))
            dyn.dyn[bad_index_low:bad_index_high, :int(60*60/dyn.dt)] = 0
            bad_time_low = 30
            bad_time_high = 30.5
            bad_index_time_low = int((bad_time_low*60)/(dyn.dt))
            bad_index_time_high = int((bad_time_high*60)/(dyn.dt))
            dyn.dyn[:, bad_index_time_low:bad_index_time_high] = 0
            bad_time_low = 147.5
            bad_time_high = 153
            bad_index_time_low = int((bad_time_low*60)/(dyn.dt))
            bad_index_time_high = int((bad_time_high*60)/(dyn.dt))
            dyn.dyn[:, bad_index_time_low:bad_index_time_high] = 0
        elif observation_date2 == '2022-10-22':
            bad_freq_low = 1028.5
            bad_freq_high = 1031
            bad_index_low = int((bad_freq_low - np.min(dyn.freqs)) /
                                (dyn.df))
            bad_index_high = int((bad_freq_high - np.min(dyn.freqs)) /
                                 (dyn.df))
            dyn.dyn[bad_index_low:bad_index_high, :] = 0
            bad_freq_low = 935
            bad_freq_high = 948.5
            bad_index_low = int((bad_freq_low - np.min(dyn.freqs)) /
                                (dyn.df))
            bad_index_high = int((bad_freq_high - np.min(dyn.freqs)) /
                                 (dyn.df))
            dyn.dyn[bad_index_low:bad_index_high, :int(60*60/dyn.dt)] = 0
            dyn.dyn[bad_index_low:bad_index_high, int(140*60/dyn.dt):] = 0
            bad_time_low = 9.75
            bad_time_high = 10.75
            bad_index_time_low = int((bad_time_low*60)/(dyn.dt))
            bad_index_time_high = int((bad_time_high*60)/(dyn.dt))
            dyn.dyn[:, bad_index_time_low:bad_index_time_high] = 0
            dyn.dyn[:, bad_index_time_low:bad_index_time_high] = 0
            bad_time_low = 29.5
            bad_time_high = 30.75
            bad_index_time_low = int((bad_time_low*60)/(dyn.dt))
            bad_index_time_high = int((bad_time_high*60)/(dyn.dt))
            dyn.dyn[:, bad_index_time_low:bad_index_time_high] = 0
            dyn.dyn[:, bad_index_time_low:bad_index_time_high] = 0
            bad_time_low = 150
            bad_time_high = 151
            bad_index_time_low = int((bad_time_low*60)/(dyn.dt))
            bad_index_time_high = int((bad_time_high*60)/(dyn.dt))
            bad_time_low = 156.75
            bad_time_high = 157.75
            bad_index_time_low = int((bad_time_low*60)/(dyn.dt))
            bad_index_time_high = int((bad_time_high*60)/(dyn.dt))
            dyn.dyn[:, bad_index_time_low:bad_index_time_high] = 0
        elif observation_date2 == '2022-11-21':
            bad_freq_low = 954.5
            bad_freq_high = 961.5
            bad_index_low = int((bad_freq_low - np.min(dyn.freqs)) /
                                (dyn.df))
            bad_index_high = int((bad_freq_high - np.min(dyn.freqs)) /
                                 (dyn.df))
            dyn.dyn[bad_index_low:bad_index_high, :] = 0
            bad_freq_low = 935
            bad_freq_high = 950
            bad_index_low = int((bad_freq_low - np.min(dyn.freqs)) /
                                (dyn.df))
            bad_index_high = int((bad_freq_high - np.min(dyn.freqs)) /
                                 (dyn.df))
            dyn.dyn[bad_index_low:bad_index_high, :] = 0
            bad_time_low = 8.75
            bad_time_high = 9.75
            bad_index_time_low = int((bad_time_low*60)/(dyn.dt))
            bad_index_time_high = int((bad_time_high*60)/(dyn.dt))
            dyn.dyn[:, bad_index_time_low:bad_index_time_high] = 0
            bad_time_low = 29.75
            bad_time_high = 30.75
            bad_index_time_low = int((bad_time_low*60)/(dyn.dt))
            bad_index_time_high = int((bad_time_high*60)/(dyn.dt))
            dyn.dyn[:, bad_index_time_low:bad_index_time_high] = 0
            bad_time_low = 128
            bad_time_high = 132
            bad_index_time_low = int((bad_time_low*60)/(dyn.dt))
            bad_index_time_high = int((bad_time_high*60)/(dyn.dt))
            dyn.dyn[:, bad_index_time_low:bad_index_time_high] = 0
            bad_time_low = 150.3
            bad_time_high = 150.7
            bad_index_time_low = int((bad_time_low*60)/(dyn.dt))
            bad_index_time_high = int((bad_time_high*60)/(dyn.dt))
            dyn.dyn[:, bad_index_time_low:bad_index_time_high] = 0
            bad_time_low = 155.75
            bad_time_high = 156.75
            bad_index_time_low = int((bad_time_low*60)/(dyn.dt))
            bad_index_time_high = int((bad_time_high*60)/(dyn.dt))
            dyn.dyn[:, bad_index_time_low:bad_index_time_high] = 0
        elif observation_date2 == '2022-12-30':
            dyn = cp(dyn)
            bad_freq_low = 935
            bad_freq_high = 948
            bad_index_low = int((bad_freq_low - np.min(dyn.freqs)) /
                                (dyn.df))
            bad_index_high = int((bad_freq_high - np.min(dyn.freqs)) /
                                 (dyn.df))
            dyn.dyn[bad_index_low:bad_index_high, :] = 0
            bad_time_low = 16.75
            bad_time_high = 18
            bad_index_time_low = int((bad_time_low*60)/(dyn.dt))
            bad_index_time_high = int((bad_time_high*60)/(dyn.dt))
            dyn.dyn[:, bad_index_time_low:bad_index_time_high] = 0
            bad_time_low = 29.75
            bad_time_high = 31
            bad_index_time_low = int((bad_time_low*60)/(dyn.dt))
            bad_index_time_high = int((bad_time_high*60)/(dyn.dt))
            dyn.dyn[:, bad_index_time_low:bad_index_time_high] = 0
            bad_time_low = 150.25
            bad_time_high = 151
            bad_index_time_low = int((bad_time_low*60)/(dyn.dt))
            bad_index_time_high = int((bad_time_high*60)/(dyn.dt))
            dyn.dyn[:, bad_index_time_low:bad_index_time_high] = 0
            bad_time_low = 164.25
            bad_time_high = 165.25
            bad_index_time_low = int((bad_time_low*60)/(dyn.dt))
            bad_index_time_high = int((bad_time_high*60)/(dyn.dt))
            dyn.dyn[:, bad_index_time_low:bad_index_time_high] = 0
        elif observation_date2 == '2023-02-27':
            dyn = cp(dyn)
            bad_freq_low = 935
            bad_freq_high = 952
            bad_index_low = int((bad_freq_low - np.min(dyn.freqs)) /
                                (dyn.df))
            bad_index_high = int((bad_freq_high - np.min(dyn.freqs)) /
                                 (dyn.df))
            dyn.dyn[bad_index_low:bad_index_high, :] = 0
            bad_time_low = 150.25
            bad_time_high = 151
            bad_index_time_low = int((bad_time_low*60)/(dyn.dt))
            bad_index_time_high = int((bad_time_high*60)/(dyn.dt))
            dyn.dyn[:, bad_index_time_low:bad_index_time_high] = 0
        else:
            print("===== The observation date had no bad freq/time",
                  "Observation", str(observation_date2), "=====")
        dyn.plot_dyn(filename=str(spectradir2) +
                     '_Zeroed_Eclipse_FullSpectra.pdf',
                     dpi=400)
        dyn.write_file(filename=str(dynspecfile3))


def measure_dynspec(overwrite=True, measure_tilt=False, correct_dynspec=False,
                    zap=True, linear=False, median=False, mean=True,
                    time_bin=10, freq_bin=30, Observations=None, wfreq=False,
                    SlowACF=False, nscale=5, display=False, phasewrapper=False,
                    weights_2dacf=None, cutoff=False, alpha=5/3,
                    dnuscale_ceil=np.inf, tauscale_ceil=np.inf):
    psrname = 'J0737-3039A'
    font = {'size': 28}
    matplotlib.rc('font', **font)

    # zeroed_fraction = []
    # fractional_bandwidth = []
    # fractional_time = []
    # ACF_residual_std = []
    # ACF_fraction_dnu = []
    # ACF_fraction_tau = []
    # fractional_uncertainty_dnu = []
    # fractional_uncertainty_tau = []
    # fractional_dnu_df = []
# /Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/New/Dynspec/J0737-3039A_2022-06-30-08:09:36_zap.dynspec
    # Dynspec/J0737-3039A_2022-06-30-08:09:36_zap.dynspec
    wd0 = '/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/'
    wd = wd0+'New/'
    datadir = wd + 'Dynspec/'
    outdir = wd + 'DataFiles/'
    dynspecs = sorted(glob.glob(datadir + '/*.dynspec'))
    if Observations:
        observations = [Observations]
    else:
        observations = []
        for i in range(0, len(dynspecs)):
            observations.append((dynspecs[i].split(datadir)[1].split('-')[1] +
                                '-' +
                                dynspecs[i].split(datadir)[1].split('-')[2] +
                                '-' +
                                dynspecs[i].split(datadir)[1].split('-')[3]).split('3039A_')[1])
        observations = np.unique(np.asarray(observations))
        print(observations)
    #
    fail_counter = 0
    good_counter = 0
    counter = 0
    counter_nodynspec = 0
    #
    error_data = []
    error_file = "/Users/jacobaskew/Desktop/error_data.txt"
    #
    if os.path.exists(error_file) and overwrite:
        os.remove(error_file)
    #
    for i in range(0, len(observations)):
        observation_date = observations[i]+'/'
        observation_date2 = observations[i]
        spectradir = wd + 'SpectraPlots/' + observation_date
        spectrabindir = wd + 'SpectraPlotsBin/' + observation_date
        ACFdir = wd + 'ACFPlots/' + observation_date
        ACFbindir = wd + 'ACFPlotsBin/' + observation_date
        dynspecs = \
            sorted(glob.glob(datadir + str(observation_date.split('/')[0])
                             + '*.dynspec'))
        if zap:
            dynspecfile2 = \
                outdir+'DynspecPlotFiles/'+observation_date2 + \
                'Zap_CompleteDynspec.dynspec'
        else:
            dynspecfile2 = \
                outdir+'DynspecPlotFiles/'+observation_date2 + \
                'Zeroed_CompleteDynspec.dynspec'
        filedir = str(outdir)+str(observation_date)
        try:
            os.mkdir(filedir)
        except OSError as error:
            print(error)
        outfile = str(filedir)+str(psrname)+'_'+str(observation_date2) + \
            '_freq'+str(freq_bin)+'_time'+str(time_bin) + \
            '_ScintillationResults_UHF.txt'
        if os.path.exists(outfile) and overwrite:
            os.remove(outfile)
        freq_bin_string = str(freq_bin)
        try:
            if overwrite and os.path.exists(spectradir):
                shutil.rmtree(spectradir)
                os.mkdir(spectradir)
            else:
                os.mkdir(spectradir)
        except OSError as error:
            print(error)
        try:
            if overwrite and os.path.exists(spectrabindir):
                shutil.rmtree(spectrabindir)
                os.mkdir(spectrabindir)
            else:
                os.mkdir(spectrabindir)
        except OSError as error:
            print(error)
        try:
            if overwrite and os.path.exists(ACFdir):
                shutil.rmtree(ACFdir)
                os.mkdir(ACFdir)
            else:
                os.mkdir(ACFdir)
        except OSError as error:
            print(error)
        try:
            if overwrite and os.path.exists(ACFbindir):
                shutil.rmtree(ACFbindir)
                os.mkdir(ACFbindir)
            else:
                os.mkdir(ACFbindir)
        except OSError as error:
            print(error)
        try:
            os.mkdir(spectradir+'time'+str(time_bin)+'freq' +
                     str(freq_bin_string)+'/')
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

        spectradir = spectradir+'time'+str(time_bin)+'freq' + \
            str(freq_bin_string)+'/'
        spectrabindir = spectrabindir+'time'+str(time_bin)+'freq' + \
            str(freq_bin_string)+'/'
        ACFdir = ACFdir+'time'+str(time_bin)+'freq' + \
            str(freq_bin_string)+'/'
        ACFbindir = ACFbindir+'time'+str(time_bin)+'freq' + \
            str(freq_bin_string)+'/'
        if os.path.exists(dynspecfile2):
            sim = Simulation()
            dyn = Dynspec(dyn=sim, process=False)
            dyn.load_file(filename=dynspecfile2)
        Fmin = np.min(dyn.freqs)  # + 48
        dyn_crop = cp(dyn)

        # I want to create a for loop that looks at each chunk of frequency
        # until it hits the bottom.
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
                    elif mean:
                        dyn_new_freq.refill(method='mean')
                    else:
                        SlowACF = True
                    if correct_dynspec:
                        dyn_new_freq.correct_dyn()
                    counter += 1
                    if dyn_new_freq.tobs < 540 or dyn_new_freq.bw < 10:  # or \
                        #    dyn_new_freq.freq < 1000:
                        # and dyn_new_freq.freq >
                        # 890:
                        error_data.append(dyn_new_freq.error_message)
                        fail_counter += 1
                        try:
                            dyn_new_freq.get_scint_params(
                                method='acf2d_approx', nscale=nscale,
                                plot=False, dnuscale_ceil=dnuscale_ceil,
                                weights_2dacf=weights_2dacf,
                                redchisqr=weights_2dacf,
                                cutoff=cutoff, wfreq=wfreq,
                                alpha=alpha, phasewrapper=phasewrapper)
                            dyn_new_freq.name = observation_date2 + \
                                "_Time" + str(round(istart_t, 1)) + "_Freq" + \
                                str(int(dyn_new_freq.freq)) + "_Dnu" + \
                                str(round(dyn_new_freq.dnu, 4)) + "_Tau" + \
                                str(round(dyn_new_freq.tau, 4)) + "_" + \
                                str(dyn_new_freq.scint_param_method) + "_" + \
                                ".dynspec"
                            dyn_new_freq.plot_dyn(filename=str(spectrabindir) +
                                                  str(dyn_new_freq.name) +
                                                  '_Dynspec.pdf', dpi=400,
                                                  display=display)
                            if measure_tilt:
                                dyn_new_freq.get_acf_tilt(
                                    filename=str(ACFbindir) +
                                    str(dyn_new_freq.name) +
                                    '_tiltplot.pdf', plot=True,
                                    display=display)
                            dyn_new_freq.get_scint_params(
                                filename=str(ACFbindir) +
                                str(dyn_new_freq.name) +
                                '.pdf', nscale=nscale, wfreq=wfreq,
                                phasewrapper=phasewrapper,
                                method='acf2d_approx', plot=True,
                                display=display, dnuscale_ceil=dnuscale_ceil,
                                weights_2dacf=weights_2dacf,
                                redchisqr=weights_2dacf,
                                cutoff=cutoff, alpha=alpha)
                            dyn_new_freq.get_scint_params(
                                filename=str(ACFbindir) +
                                str(dyn_new_freq.name) +
                                '.pdf', method='acf1d', plot=True,
                                display=display, nscale=nscale*2, wfreq=wfreq,
                                dnuscale_ceil=dnuscale_ceil, alpha=alpha,
                                phasewrapper=phasewrapper)
                        except Exception as e:
                            print(e)
                            counter_nodynspec += 1
                        continue
                    else:
                        if SlowACF:
                            data = dyn_new_freq.dyn
                            data = np.ma.masked_where(data == 0, data)
                            data = np.ma.masked_invalid(data)
                            dyn_new_freq.get_scint_params(
                                method='acf2d_approx', plot=False,
                                nscale=nscale, dnuscale_ceil=dnuscale_ceil,
                                weights_2dacf=weights_2dacf,
                                redchisqr=weights_2dacf, wfreq=wfreq,
                                cutoff=cutoff, alpha=alpha,
                                phasewrapper=phasewrapper)
                            dyn_new_freq.name = observation_date2 + \
                                "_Time" + str(round(istart_t, 1)) + "_Freq" + \
                                str(int(dyn_new_freq.freq)) + "_Dnu" + \
                                str(round(dyn_new_freq.dnu, 4)) + "_Tau" + \
                                str(round(dyn_new_freq.tau, 4)) + "_" + \
                                str(dyn_new_freq.scint_param_method) + "_" + \
                                ".dynspec"
                            dyn_name = dyn_new_freq.name
                            autocorr_func(data, dyn_name, ACFbindir, istart_t,
                                          outfile)
                        else:
                            # if dyn_new_freq.freq > 900:
                            #     nscale = 2
                            # elif dyn_new_freq.freq > 700 and \
                            #         dyn_new_freq.freq < 900:
                            #     nscale = 4
                            # else:
                            #     nscale = 5
                            # for i in range(0, 10):
                            dyn_new_freq.get_scint_params(
                                method='acf2d_approx', plot=False,
                                nscale=nscale, dnuscale_ceil=dnuscale_ceil,
                                weights_2dacf=weights_2dacf,
                                redchisqr=weights_2dacf, wfreq=wfreq,
                                cutoff=cutoff, alpha=alpha,
                                phasewrapper=phasewrapper)
                            dyn_new_freq.name = observation_date2 + \
                                "_Time" + str(round(istart_t, 1)) + "_Freq" + \
                                str(int(dyn_new_freq.freq)) + "_Dnu" + \
                                str(round(dyn_new_freq.dnu, 4)) + "_Tau" + \
                                str(round(dyn_new_freq.tau, 4)) + "_" + \
                                str(dyn_new_freq.scint_param_method) + "_" + \
                                ".dynspec"
                            #
                            if 0.4*(dyn_new_freq.freq/1024)**4 > 0.1:
                                dnuscale_ceil = 0.4*(dyn_new_freq.freq/1024)**4
                            #
                            dyn_new_freq.get_scint_params(
                                filename=str(ACFdir) +
                                str(dyn_new_freq.name)+'.pdf', wfreq=wfreq,
                                method='acf1d', plot=True, display=display,
                                nscale=nscale*2, dnuscale_ceil=dnuscale_ceil,
                                alpha=alpha, phasewrapper=phasewrapper)
                            # acf1d_dnu = dyn_new_freq.dnu
                            # acf1d_tau = dyn_new_freq.tau
                            dyn_new_freq.get_scint_params(
                                filename=str(ACFdir) +
                                str(dyn_new_freq.name)+'.pdf',
                                method='acf2d_approx', display=display,
                                plot=True, nscale=nscale,
                                dnuscale_ceil=dnuscale_ceil,
                                weights_2dacf=weights_2dacf,
                                redchisqr=weights_2dacf, wfreq=wfreq,
                                phasewrapper=phasewrapper,
                                cutoff=cutoff, alpha=alpha)
                            # Optionally: cutting at dnu best fit * 3
                            if dnuscale_ceil < dyn_new_freq.dnu*3:
                                dnuscale_ceil_new = dnuscale_ceil
                            else:
                                dnuscale_ceil_new = dyn_new_freq.dnu*3
                            # if tauscale_ceil < dyn_new_freq.tau*3:
                            #     tauscale_ceil_new = tauscale_ceil
                            # else:
                            #     tauscale_ceil_new = dyn_new_freq.dnu*3
                            dyn_new_freq.get_scint_params(
                                filename=str(ACFdir) +
                                str(dyn_new_freq.name)+'.pdf',
                                method='acf2d_approx', display=display,
                                plot=True, nscale=nscale,
                                dnuscale_ceil=dnuscale_ceil_new,
                                weights_2dacf=weights_2dacf,
                                redchisqr=weights_2dacf, wfreq=wfreq,
                                phasewrapper=phasewrapper,
                                cutoff=cutoff, alpha=alpha)
                            #
                            if measure_tilt:
                                dyn_new_freq.get_acf_tilt(
                                    filename=str(ACFdir) +
                                    str(dyn_new_freq.name) +
                                    '_tiltplot.pdf',
                                    plot=True, display=display)
                                dyn_new_freq.get_scint_params(
                                    filename=str(ACFdir) +
                                    str(dyn_new_freq.name) +
                                    '_tilted.pdf', wfreq=wfreq,
                                    dnuscale_ceil=dnuscale_ceil_new,
                                    method='acf2d_approx', plot=True,
                                    nscale=nscale, display=display,
                                    weights_2dacf=weights_2dacf,
                                    redchisqr=weights_2dacf,
                                    cutoff=cutoff, alpha=alpha,
                                    phasewrapper=phasewrapper)
                            #
                                # if len(np.argwhere((dyn_new_freq.dyn == 0) *
                                #                    (dyn_new_freq.dyn ==
                                # np.mean(dyn_new_freq.dyn)))) \
                                #         == 0:
                                #     zeroed_fraction.append(0)
                                # else:
                                #     zeroed_fraction.append(
                                #         (len(np.argwhere(
                                #             (dyn_new_freq.dyn == 0) *
                                #             (dyn_new_freq.dyn ==
                                #              np.mean(dyn_new_freq.dyn))))
                                #          / len(dyn_new_freq.dyn))*100)
                                # fractional_bandwidth.append(dyn_new_freq.bw/30)
                                # fractional_time.append(dyn_new_freq.tobs/(600))
                                # ACF_residual_std.append(
                                #     np.std(dyn_new_freq.acf_residuals))
                                # #
                                # acf2d_dnu = dyn_new_freq.dnu
                                # acf2d_dnuerr = dyn_new_freq.dnuerr
                                # acf2d_tau = dyn_new_freq.tau
                                # acf2d_tauerr = dyn_new_freq.tauerr
                                # #
                                # ACF_fraction_dnu.append(acf2d_dnu/acf1d_dnu*100)
                                # ACF_fraction_tau.append(acf2d_tau/acf1d_tau*100)
                                # fractional_uncertainty_dnu.append(
                                #     acf2d_dnuerr/acf2d_dnu*100)
                                # fractional_uncertainty_tau.append(
                                #     acf2d_tauerr/acf2d_tau*100)
                                # fractional_dnu_df.append(
                                #     acf2d_dnu/dyn_new_freq.df*100)
                                #
                                # if dyn_new_freq.scint_param_method !=
                                # 'nofit':
                            if dyn_new_freq.scint_param_method == 'nofit':
                                1/0
                            if dyn_new_freq.tauerr > 10*dyn_new_freq.tau:
                                1/0
                            if dyn_new_freq.dnuerr > 10*dyn_new_freq.dnu:
                                1/0
                            error_data.append(dyn_new_freq.error_message)
                            write_results(outfile, dyn=dyn_new_freq)
                            good_counter += 1
                            dyn_new_freq.plot_dyn(
                                filename=str(spectradir) +
                                str(dyn_new_freq.name) +
                                '_Dynspec.pdf', dpi=400, display=display)
                except Exception as e:
                    print(e)
                    fail_counter += 1
                    try:
                        error_data.append(dyn_new_freq.error_message)
                        dyn_new_freq.plot_dyn(filename=str(spectrabindir) +
                                              str(dyn_new_freq.name) +
                                              '_Dynspec.pdf', dpi=400,
                                              display=display)
                        if measure_tilt:
                            dyn_new_freq.get_acf_tilt(
                                filename=str(ACFbindir) +
                                str(dyn_new_freq.name) +
                                '_tiltplot.pdf',
                                plot=True, display=display)
                        dyn_new_freq.get_scint_params(
                            filename=str(ACFbindir) +
                            str(dyn_new_freq.name) +
                            '.pdf', dnuscale_ceil=dnuscale_ceil,
                            method='acf2d_approx', weights_2dacf=weights_2dacf,
                            cutoff=cutoff, phasewrapper=phasewrapper,
                            redchisqr=weights_2dacf, plot=True,
                            display=display, wfreq=wfreq,
                            nscale=nscale, alpha=alpha)
                        dyn_new_freq.get_scint_params(
                            filename=str(ACFbindir) +
                            str(dyn_new_freq.name) +
                            '.pdf', alpha=alpha, phasewrapper=phasewrapper,
                            method='acf1d', dnuscale_ceil=dnuscale_ceil,
                            plot=True, display=display, nscale=nscale*2,
                            wfreq=wfreq)
                    except Exception as e:
                        print(e)
                        counter_nodynspec += 1
                    continue
    # TestingParams = np.zeros((len(zeroed_fraction)+1, 9))
    # # TestingParams[0, :] = np.asarray(['zeroed_fraction',
    # #                                   'fractional_bandwidth',
    # #                                   'fractional_time',
    # 'ACF_residual_std',
    # #                                   'ACF_fraction_dnu',
    # 'ACF_fraction_tau',
    # #                                   'fractional_uncertainty_dnu',
    # #                                   'fractional_uncertainty_tau',
    # #                                   'fractional_dnu_df'])
    # print(zeroed_fraction, len(zeroed_fraction))
    # TestingParams[:, 0] = np.asarray(zeroed_fraction)
    # TestingParams[:, 1] = np.asarray(fractional_bandwidth)
    # TestingParams[:, 2] = np.asarray(fractional_time)
    # TestingParams[:, 3] = np.asarray(ACF_residual_std)
    # TestingParams[:, 4] = np.asarray(ACF_fraction_dnu)
    # TestingParams[:, 5] = np.asarray(ACF_fraction_tau)
    # TestingParams[:, 6] = np.asarray(fractional_uncertainty_dnu)
    # TestingParams[:, 7] = np.asarray(fractional_uncertainty_tau)
    # TestingParams[:, 8] = np.asarray(fractional_dnu_df)
    # TestingParams_File = str(outdir)+'freq'+str(freq_bin)+'_time' + \
    #     str(time_bin)+'_TestingParams_UHF.txt'
    # np.savetxt(TestingParams_File, TestingParams, delimiter=',', fmt='%s')
    error_data = np.asarray(error_data)
    np.savetxt(error_file, error_data, fmt='%s')
    Fs_len = \
        len(
            error_data[
                np.argwhere(
                    'One or more variable did not affect the fit. ' +
                    'Could not estimate error-bars.'
                    == error_data)])
    Natf_len = \
        len(error_data[np.argwhere(
            'Fit succeeded. Could not estimate error-bars.' == error_data)])
    quality_counter = good_counter - Natf_len - Fs_len
    print("========== successfully completed measuring the data ==========")
    print("========== there were " + str(counter) +
          " measurements taken ==========")
    print("========== there were " + str(quality_counter) +
          " quality measurements taken ==========")
    print("========== fractional failure was ... ==========")
    print("========== " + str(round(fail_counter/counter*100, 3)) +
          " ==========")
    print("==========  fractional success was ... ==========")
    print("========== " + str(round(quality_counter/counter*100, 3)) +
          " ==========")
    print("==========  fractional error message 'variable did not affect" +
          " the fit' was ... ==========")
    print("========== " + str(round(Fs_len/counter*100, 3)) +
          " ==========")
    print("==========  fractional error message " +
          " 'Could not estimate error-bars' was ... ==========")
    print("========== " + str(round(Natf_len/counter*100, 3)) +
          " ==========")
    print("==============================================================")


def plot_data(merge_data=True, load_data=True, basic_plot=True, compare=False,
              compare_filename1=None, compare_filename2=None, time_bin=30,
              freq_bin=30, spectral_index=False, Observations=None,
              filtered=False, filtered_dnu=5, filtered_tau=5, overwrite=False,
              modelling_data=False, nlive=200, resume=True,
              measure_tilt=False, weights_2dacf=False,
              filtered_acf_redchisqr=1e4):
    psrname = 'J0737-3039A'
    # pulsar = '0737-3039A'

    # Pre-define any plotting arrangements
    Size = 80*np.pi  # Determines the size of the datapoints used
    font = {'size': 28}
    matplotlib.rc('font', **font)

    wd0 = '/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/'
    wd = wd0+'New/'
    datadir = wd + 'Dynspec/'
    outdir = wd + 'DataFiles/'
    par_dir = '/Users/jacobaskew/Desktop/Swinburne_Daniel/ParFiles/'
    plotdir = wd + 'Plots/'
    dynspecs = sorted(glob.glob(datadir + '/*.dynspec'))
    if Observations:
        observations = [Observations]
    else:
        observations = []
        for i in range(0, len(dynspecs)):
            observations.append((dynspecs[i].split(datadir)[1].split('-')[1] +
                                '-' +
                                dynspecs[i].split(datadir)[1].split('-')[2] +
                                '-' +
                                dynspecs[i].split(datadir)[1].split('-')[3]).split('3039A_')[1])
        observations = np.unique(np.asarray(observations))
    
   
    if len(observations) > 1:
        # Merge existing files together if many observations #
        if merge_data:
            merge_data(Observations=Observations, freq_bin=freq_bin,
                       time_bin=time_bin, weights_2dacf=weights_2dacf,
                       measure_tilt=measure_tilt)
    # Load results if they exist, if they don't split and save them #
    if load_data:
                
        if len(observations) == 1:
            filepath = wd + "DataFiles/" + observations[0] + "/"
            outfile_total = \
                str(filepath)+str(psrname)+'_'+str(observations[0]) + \
                '_freq'+str(freq_bin)+'_time'+str(time_bin) + \
                '_ScintillationResults_UHF.txt'
        else:
            filepath = outdir + '/ScintData/'         
            outfile_total = str(outdir)+str(psrname)+'_freq' + \
                str(freq_bin)+'_time'+str(time_bin) + \
                '_ScintillationResults_UHF_Total.txt'
    
        if os.path.exists(filepath+'_visserr.txt') and not overwrite:
            mjd = np.loadtxt(filepath+'_mjd.txt', dtype='float')
            freqMHz = np.loadtxt(filepath+'_freqMHz.txt', dtype='float')
            df = np.loadtxt(filepath+'_df.txt', dtype='float')
            dnu = np.loadtxt(filepath+'_dnu.txt', dtype='float')
            dnuerr = np.loadtxt(filepath+'_dnuerr.txt', dtype='float')
            tau = np.loadtxt(filepath+'_tau.txt', dtype='float')
            tauerr = np.loadtxt(filepath+'_tauerr.txt', dtype='float')
            name = np.loadtxt(filepath+'_name.txt', dtype='str')
            phasegrad = np.loadtxt(filepath+'_phasegrad.txt', dtype='float')
            phasegraderr = np.loadtxt(filepath+'_phasegraderr.txt', dtype='float')
            phase = np.loadtxt(filepath+'_phase.txt', dtype='float')
            U = np.loadtxt(filepath+'_U.txt', dtype='float')
            viss = np.loadtxt(filepath+'_viss.txt', dtype='float')
            visserr = np.loadtxt(filepath+'_visserr.txt', dtype='float')
            if weights_2dacf:
                np.loadtxt(filepath+'_acf_redchisqr.txt', dtype='float')
                np.loadtxt(filepath+'_acf_model_redchisqr.txt', dtype='float')
            if measure_tilt:
                np.loadtxt(filepath+'_acf_tilt.txt', dtype='float')
                np.loadtxt(filepath+'_acf_tilt_err.txt', dtype='float')
            mjd_annual = mjd % 365.2425
        else:
            params = read_results(outfile_total)
            pars = read_par(str(par_dir) + str(psrname) + '.par')
        
            # Read in arrays
            mjd = float_array_from_dict(params, 'mjd')
            len_mjd_before = len(mjd)
            df = float_array_from_dict(params, 'df')  # channel bandwidth
            dnu = float_array_from_dict(params, 'dnu')  # scint bandwidth
            dnuerr = float_array_from_dict(params, 'dnuerr')
            tau = float_array_from_dict(params, 'tau')
            tauerr = float_array_from_dict(params, 'tauerr')
            freqMHz = float_array_from_dict(params, 'freq')
            bw = float_array_from_dict(params, 'bw')
            name = np.asarray(params['name'])
            tobs = float_array_from_dict(params, 'tobs')  # tobs in second
            scint_param_method = np.asarray(params['scint_param_method'])
            phasegrad = float_array_from_dict(params, 'phasegrad')
            phasegraderr = float_array_from_dict(params, 'phasegraderr')
            if weights_2dacf:
                acf_redchisqr = np.asarray(params['acf_redchisqr'])
                acf_redchisqr = acf_redchisqr.astype(np.float)
                acf_model_redchisqr = np.asarray(params['acf_model_redchisqr'])
                acf_model_redchisqr = acf_model_redchisqr.astype(np.float)
            if measure_tilt:
                acf_tilt = float_array_from_dict(params, 'acf_tilt')
                acf_tilt_err = float_array_from_dict(params, 'acf_tilt_err')
    
            # Sort by MJD
            sort_ind = np.argsort(mjd)
    
            df = np.array(df[sort_ind]).squeeze()
            dnu = np.array(dnu[sort_ind]).squeeze()
            dnuerr = np.array(dnuerr[sort_ind]).squeeze()
            tau = np.array(tau[sort_ind]).squeeze()
            tauerr = np.array(tauerr[sort_ind]).squeeze()
            mjd = np.array(mjd[sort_ind]).squeeze()
            freqMHz = np.array(freqMHz[sort_ind]).squeeze()
            tobs = np.array(tobs[sort_ind]).squeeze()
            name = np.array(name[sort_ind]).squeeze()
            phasegrad = np.array(phasegrad[sort_ind]).squeeze()
            phasegraderr = np.array(phasegraderr[sort_ind]).squeeze()
            bw = np.array(bw[sort_ind]).squeeze()
            scint_param_method = np.array(scint_param_method[sort_ind]).squeeze()
            if weights_2dacf:
                acf_redchisqr = np.array(acf_redchisqr[sort_ind]).squeeze()
                acf_model_redchisqr = \
                    np.array(acf_model_redchisqr[sort_ind]).squeeze()
            if measure_tilt:
                acf_tilt = np.array(acf_tilt[sort_ind]).squeeze()
                acf_tilt_err = np.array(acf_tilt_err[sort_ind]).squeeze()
    
            # Used to filter the data
            if filtered:
                if weights_2dacf:
                    indicies = np.argwhere((tauerr < filtered_tau*tau) *
                                           (dnuerr < filtered_dnu*dnu) *
                                           (scint_param_method == "acf2d_approx") *
                                           (acf_redchisqr <
                                            filtered_acf_redchisqr))
                else:
                    indicies = np.argwhere((tauerr < filtered_tau*tau) *
                                           (dnuerr < filtered_dnu*dnu) *
                                           (scint_param_method == "acf2d_approx"))
    
                df = df[indicies].squeeze()
                dnu = dnu[indicies].squeeze()
                dnuerr = dnuerr[indicies].squeeze()
                tau = tau[indicies].squeeze()
                tauerr = tauerr[indicies].squeeze()
                mjd = mjd[indicies].squeeze()
                freqMHz = freqMHz[indicies].squeeze()
                tobs = tobs[indicies].squeeze()
                name = name[indicies].squeeze()
                bw = bw[indicies].squeeze()
                phasegrad = phasegrad[indicies].squeeze()
                phasegraderr = phasegraderr[indicies].squeeze()
                scint_param_method = scint_param_method[indicies].squeeze()
                if weights_2dacf:
                    acf_redchisqr = acf_redchisqr[indicies].squeeze()
                    acf_model_redchisqr = acf_model_redchisqr[indicies].squeeze()
                if measure_tilt:
                    acf_tilt = acf_tilt[indicies].squeeze()
                    acf_tilt_err = acf_tilt_err[indicies].squeeze()
                print("========== filtering was done ==========")
                print("========== fractional difference was ... ==========")
                print("========== " + str(round(len(mjd)/len_mjd_before*100, 3)) +
                      " ==========")
                np.savetxt(filepath+'_freqMHz.txt', freqMHz, delimiter=',', fmt='%s')
                np.savetxt(filepath+'_df.txt', df, delimiter=',', fmt='%s')
                np.savetxt(filepath+'_dnu.txt', dnu, delimiter=',', fmt='%s')
                np.savetxt(filepath+'_dnuerr.txt', dnuerr, delimiter=',', fmt='%s')
                np.savetxt(filepath+'_tau.txt', tau, delimiter=',', fmt='%s')
                np.savetxt(filepath+'_tauerr.txt', tauerr, delimiter=',', fmt='%s')
                np.savetxt(filepath+'_name.txt', name, delimiter=',', fmt='%s')
                np.savetxt(filepath+'_phasegrad.txt', phasegrad, delimiter=',', fmt='%s')
                np.savetxt(filepath+'_phasegraderr.txt', phasegraderr, delimiter=',', fmt='%s')
                #
                if weights_2dacf:
                    np.savetxt(filepath+'_acf_redchisqr.txt', acf_redchisqr, delimiter=',', fmt='%s')
                    np.savetxt(filepath+'_acf_model_redchisqr.txt', acf_model_redchisqr, delimiter=',', fmt='%s')
                if measure_tilt:
                    np.savetxt(filepath+'_acf_tilt.txt', acf_tilt, delimiter=',', fmt='%s')
                    np.savetxt(filepath+'_acf_tilt_err.txt', acf_tilt_err, delimiter=',', fmt='%s')
                #
                pars = read_par(str(par_dir) + str(psrname) + '.par')
                params = pars_to_params(pars)
                
                np.savetxt(filepath+'noTOBS_noSSB_mjd.txt', mjd, delimiter=',', fmt='%s')
                                    
                mjd += (tobs / 2) / 86400
                
                np.savetxt(filepath+'TOBS_noSSB_mjd.txt', mjd, delimiter=',', fmt='%s')
    
                ssb_delays = get_ssb_delay(mjd, pars['RAJ'], pars['DECJ'])
                mjd += np.divide(ssb_delays, 86400)  # add ssb delay
                
                np.savetxt(filepath+'_mjd.txt', mjd, delimiter=',', fmt='%s')
                
                mjd_annual = mjd % 365.2425

                noTOBS_SSB_mjd = mjd - (tobs / 2) / 86400
                
                np.savetxt(filepath+'noTOBS_SSB_mjd.txt', noTOBS_SSB_mjd,
                           delimiter=',', fmt='%s')
                
                vearth_ra, vearth_dec = get_earth_velocity(mjd, pars['RAJ'], pars['DECJ'])
                U = get_true_anomaly(mjd, pars)
                vearth_ra = vearth_ra.squeeze()
                vearth_dec = vearth_dec.squeeze()
                om = pars['OM'] + pars['OMDOT']*(mjd - pars['T0'])/365.2425
                phase = U*180/np.pi + om
                phase = phase % 360
                #
                np.savetxt(filepath+'_phase.txt', phase, delimiter=',', fmt='%s')
                np.savetxt(filepath+'_U.txt', U, delimiter=',', fmt='%s')
                np.savetxt(filepath+'_ve_ra.txt', vearth_ra, delimiter=',', fmt='%s')
                np.savetxt(filepath+'_ve_dec.txt', vearth_dec, delimiter=',', fmt='%s')
                #
                D = 0.735  # kpc
                # D_err = 0.06
                s = 0.7
                kappa = 1
                freqGHz = freqMHz / 1e3
    
                Aiss = kappa * 3.347290399e4 * np.sqrt((2*(1-s))/(s))
                viss = Aiss * (np.sqrt(D*dnu))/(freqGHz*tau)
                visserr = viss * np.sqrt((dnuerr/(2*dnu))**2+(-tauerr/tau)**2)
    
                np.savetxt(filepath+'_viss.txt', viss, delimiter=',', fmt='%s')
                np.savetxt(filepath+'_visserr.txt', visserr, delimiter=',', fmt='%s')

    # Perform plotting with loaded/saved results #
    if basic_plot:
        # ORBITAL phase against bandwidth for each 'observation run'
        fig = plt.figure(figsize=(20, 10))
        fig.subplots_adjust(hspace=0.5, wspace=0.5)
        cm = plt.cm.get_cmap('gist_ncar')
        z = mjd_annual
        sc = plt.scatter(phase, dnu, c=z, cmap=cm, s=Size, alpha=0.7)
        # plt.colorbar(sc)
        plt.errorbar(phase, dnu, yerr=dnuerr, fmt=' ', ecolor='k',
                     elinewidth=2, capsize=3, alpha=0.55)
        xl = plt.xlim()
        plt.plot(xl, (df[0], df[0]), color='C2')
        plt.xlabel('Orbital Phase (deg)')
        plt.ylabel('Scintillation Bandwidth (MHz)')
        plt.title('Orbital Phase and "Annual Phase"')
        plt.xlim(xl)
        plt.savefig("/Users/jacobaskew/Desktop/Dnu_Orbital_Freq.png")
        plt.savefig(plotdir+"Dnu_Orbital_Freq.pdf", dpi=400)
        plt.show()
        plt.close()

        # A plot showing the annual modulation if any?! ANNUAL #
        fig = plt.figure(figsize=(20, 10))
        fig.subplots_adjust(hspace=0.5, wspace=0.5)
        ax = fig.add_subplot(1, 1, 1)
        cm = plt.cm.get_cmap('gist_ncar')
        z = mjd_annual
        sc = plt.scatter(freqMHz, dnu, c=z, cmap=cm, s=Size, alpha=0.2)
        # plt.colorbar(sc)
        plt.errorbar(freqMHz, dnu, yerr=dnuerr, fmt=' ', ecolor='k',
                     elinewidth=2, capsize=3, alpha=0.1)
        xl = plt.xlim()
        plt.plot(xl, (0.0332, 0.0332), color='C2')
        freq_range = np.linspace(xl[0], xl[1], 10000)
        freq_sort = np.argsort(freq_range)
        estimated_si = 0.05*(freq_range/800)**4
        plt.plot(freq_range[freq_sort], estimated_si[freq_sort], color='k',
                 alpha=0.7)
        # predicted_si = 0.05*(freq_range/800)**2
        # plt.plot(freq_range[freq_sort], predicted_si[freq_sort], color='C1',
        #          alpha=0.7)
        plt.xlabel('Observation Frequency (MHz)')
        plt.ylabel('Scintillation Bandwidth (MHz)')
        plt.title('Spectral Index and "Annual Phase"')
        plt.xlim(xl)
        plt.savefig("/Users/jacobaskew/Desktop/Dnu_Freq_Observation.png")
        plt.savefig(plotdir+"Dnu_Freq_Observation.pdf", dpi=400)
        ax.set_xscale('log')
        ax.set_yscale('log')
        plt.savefig(
            "/Users/jacobaskew/Desktop/Dnu_Freq_annual_Observation_log.png")
        plt.savefig(plotdir+"Dnu_Freq_annual_Observation_log.pdf", dpi=400)
        plt.show()
        plt.close()
        # if weights_2dacf:

            # # A plot showing the annual modulation if any?! ANNUAL #
            # fig = plt.figure(figsize=(20, 10))
            # fig.subplots_adjust(hspace=0.5, wspace=0.5)
            # ax = fig.add_subplot(1, 1, 1)
            # cm = plt.cm.get_cmap('viridis')
            # z = acf_redchisqr
            # sc = plt.scatter(freqMHz, dnu, c=z, cmap=cm, s=Size, alpha=0.2)
            # plt.colorbar(sc)
            # plt.errorbar(freqMHz, dnu, yerr=dnuerr, fmt=' ', ecolor='k',
            #              elinewidth=2, capsize=3, alpha=0.1)
            # xl = plt.xlim()
            # plt.plot(xl, (0.0332, 0.0332), color='C2')
            # freq_range = np.linspace(xl[0], xl[1], 10000)
            # freq_sort = np.argsort(freq_range)
            # estimated_si = 0.05*(freq_range/800)**4
            # plt.plot(freq_range[freq_sort], estimated_si[freq_sort], color='k',
            #          alpha=0.7)
            # predicted_si = 0.05*(freq_range/800)**2
            # plt.plot(freq_range[freq_sort], predicted_si[freq_sort],
            #          color='C1', alpha=0.7)
            # plt.xlabel('Observation Frequency (MHz)')
            # plt.ylabel('Scintillation Bandwidth (MHz)')
            # plt.title('Spectral Index and "red-chisqr"')
            # plt.xlim(xl)
            # plt.savefig(
            #     "/Users/jacobaskew/Desktop/Dnu_Freq_chisqr_Observation.png")
            # plt.savefig(plotdir+"Dnu_Freq_chisqr_Observation.pdf", dpi=400)
            # ax.set_xscale('log')
            # ax.set_yscale('log')
            # plt.savefig(
            #     "/Users/jacobaskew/Desktop/Dnu_Freq_chisqr_Observation_ln.png")
            # plt.savefig(plotdir+"Dnu_Freq_chisqr_Observation_log.pdf", dpi=400)
            # plt.show()
            # plt.close()

            # acf_redchisqr_log = np.log(acf_redchisqr)
            # acf_model_redchisqr_log = np.log(acf_model_redchisqr)

            # fig, ax = plt.subplots(figsize=(15, 15))
            # plt.hist(acf_redchisqr_log, color='C0', bins=50, alpha=0.6,
            #          label='acf_redchisqr_log')
            # plt.hist(acf_model_redchisqr_log, color='C3', bins=50, alpha=0.6,
            #          label='model_redchisqr_log')
            # plt.xlabel("Reduced chi-sqr (log)")
            # plt.ylabel("Frequency")
            # plt.show()
            # plt.close()

        # ORBITAL phase against timescale
        fig = plt.figure(figsize=(20, 10))
        fig.subplots_adjust(hspace=0.5, wspace=0.5)
        cm = plt.cm.get_cmap('gist_ncar')
        z = mjd_annual
        sc = plt.scatter(phase, tau, c=z, cmap=cm, s=Size, alpha=0.7)
        # plt.colorbar(sc)
        plt.errorbar(phase, tau, yerr=tauerr, fmt=' ', ecolor='k',
                     elinewidth=2, capsize=3, alpha=0.55)
        xl = plt.xlim()
        plt.xlabel('Orbital Phase (deg)')
        plt.ylabel('Scintillation Timescale (s)')
        plt.title('Timescale and "Annual Phase"')
        plt.xlim(xl)
        plt.savefig("/Users/jacobaskew/Desktop/Dnu_Orbital_Freq.png")
        plt.savefig(plotdir+"Dnu_Orbital_Freq.pdf", dpi=400)
        plt.show()
        plt.close()

        # Viss against orbital phase with annual phase
        fig = plt.figure(figsize=(20, 10))
        fig.subplots_adjust(hspace=0.5, wspace=0.5)
        ax = fig.add_subplot(1, 1, 1)
        cm = plt.cm.get_cmap('gist_ncar')
        z = mjd_annual
        sc = plt.scatter(phase, viss, c=z, cmap=cm, s=Size, alpha=0.7)
        # plt.colorbar(sc)
        plt.errorbar(phase, viss, yerr=visserr, fmt=' ', ecolor='k',
                     elinewidth=2, capsize=3, alpha=0.55)
        xl = plt.xlim()
        plt.xlabel('Orbital Phase (deg)')
        plt.ylabel(r'Scintillation Velocity (km$\,$s$^{-1}$)')
        plt.title('Velocity and Orbital/Annual Phase')
        plt.xlim(xl)
        # plt.ylim(0, 0.2)
        plt.savefig("/Users/jacobaskew/Desktop/Viss_Observation.png")
        plt.savefig(plotdir+"Viss_Observation.pdf", dpi=400)
        # ax.set_xscale('log')
        # ax.set_yscale('log')
        plt.show()
        plt.close()

        fig = plt.figure(figsize=(20, 10))
        fig.subplots_adjust(hspace=0.5, wspace=0.5)
        ax = fig.add_subplot(1, 1, 1)
        cm = plt.cm.get_cmap('gist_ncar')
        z = mjd_annual
        sc = plt.scatter(phase, phasegrad, c=z, cmap=cm, s=Size, alpha=0.7)
        plt.errorbar(phase, phasegrad, yerr=phasegraderr, fmt=' ',
                     ecolor='k', elinewidth=2, capsize=3, alpha=0.55)
        xl = plt.xlim()
        plt.xlabel('Orbital Phase (deg)')
        plt.ylabel('Phase Gradient (min/MHz)')
        plt.title('Phase Gradient across Annual/Orbital Phase')
        plt.xlim(xl)
        plt.savefig("/Users/jacobaskew/Desktop/PhaseGradient.png")
        plt.savefig(plotdir+"PhaseGradient.pdf", dpi=400)
        # ax.set_xscale('log')
        # ax.set_yscale('log')
        plt.show()
        plt.close()

        if measure_tilt:

            acf_tilt_sort = np.argsort(acf_tilt)

            fig = plt.figure(figsize=(20, 10))
            fig.subplots_adjust(hspace=0.5, wspace=0.5)
            ax = fig.add_subplot(1, 1, 1)
            cm = plt.cm.get_cmap('gist_ncar')
            z = mjd_annual
            sc = plt.scatter(phase, acf_tilt, c=z, cmap=cm, s=Size, alpha=0.7)
            plt.errorbar(phase[acf_tilt_sort], acf_tilt[acf_tilt_sort],
                         yerr=acf_tilt_err[acf_tilt_sort], fmt=' ',
                         ecolor='k', elinewidth=2, capsize=3, alpha=0.3)
            # plt.colorbar(sc)
            xl = plt.xlim()
            plt.xlabel('Orbital Phase (deg)')
            plt.ylabel('ACF Tilt (min/MHz)')
            plt.title('Tilt in the ACF across Annual/Orbital Phase')
            plt.xlim(xl)
            plt.ylim(np.min(acf_tilt)*0.9,
                     np.max(acf_tilt)*1.1)
            plt.savefig("/Users/jacobaskew/Desktop/ACF_Tilt.png")
            plt.savefig(plotdir+"ACF_Tilt.pdf", dpi=400)
            # ax.set_xscale('log')
            # ax.set_yscale('log')
            plt.show()
            plt.close()

            # Now both the tilt and phase gradient, tilt without errorbars
            fig = plt.figure(figsize=(20, 10))
            fig.subplots_adjust(hspace=0.5, wspace=0.5)
            ax = fig.add_subplot(1, 1, 1)
            plt.scatter(phase, phasegrad, c='C0', s=Size, alpha=0.4)
            plt.errorbar(phase, phasegrad, yerr=phasegraderr, fmt=' ',
                         ecolor='C0', elinewidth=2, capsize=3, alpha=0.05)
            plt.scatter(phase, acf_tilt, c='C1', s=Size, alpha=0.4)
            plt.errorbar(phase[acf_tilt_sort], acf_tilt[acf_tilt_sort],
                         yerr=acf_tilt_err[acf_tilt_sort], fmt=' ',
                         ecolor='C1', elinewidth=2, capsize=3, alpha=0.05)
            xl = plt.xlim()
            plt.xlabel('Orbital Phase (deg)')
            plt.ylabel('ACF Tilt (min/MHz)')
            plt.title('Comparison between tilt measuring and fitting')
            plt.xlim(xl)
            # plt.ylim(np.min(acf_tilt)*0.9,
            #           np.max(acf_tilt)*1.1)
            plt.ylim(-30, 30)
            plt.savefig("/Users/jacobaskew/Desktop/ACF_Tilt.png")
            plt.savefig(plotdir+"ACF_Tilt.pdf", dpi=400)
            # ax.set_xscale('log')
            # ax.set_yscale('log')
            plt.show()
            plt.close()

    if compare:
        # Here we are going to combine the two datasets and attempt to fit the
        # powerlaw
        params = read_results(compare_filename2)

        pars = read_par(str(par_dir) + str(psrname) + '.par')

        # Read in arrays
        old_mjd = float_array_from_dict(params, 'mjd')
        old_df = float_array_from_dict(params, 'df')  # channel bandwidth
        old_dnu = float_array_from_dict(params, 'dnu')  # scint bandwidth
        old_dnu_est = float_array_from_dict(params, 'dnu_est')  # estimated
        old_dnuerr = float_array_from_dict(params, 'dnuerr')
        old_tau = float_array_from_dict(params, 'tau')
        old_tauerr = float_array_from_dict(params, 'tauerr')
        old_phasegrad = float_array_from_dict(params, 'phasegrad')
        old_phasegraderr = float_array_from_dict(params, 'phasegraderr')
        old_freq = float_array_from_dict(params, 'freq')
        old_bw = float_array_from_dict(params, 'bw')
        # old_scintle_num = float_array_from_dict(params, 'scintle_num')
        old_tobs = float_array_from_dict(params, 'tobs')  # tobs in second
        old_rcvrs = np.array([rcvr[0] for rcvr in params['name']])
        # old_scint_param_method = float_array_from_dict(params,
        #                                                'scint_param_method')

        # Sort by MJD
        sort_ind = np.argsort(old_mjd)

        old_df = np.array(old_df[sort_ind]).squeeze()
        old_dnu = np.array(old_dnu[sort_ind]).squeeze()
        old_dnu_est = np.array(old_dnu_est[sort_ind]).squeeze()
        old_dnuerr = np.array(old_dnuerr[sort_ind]).squeeze()
        old_tau = np.array(old_tau[sort_ind]).squeeze()
        old_tauerr = np.array(old_tauerr[sort_ind]).squeeze()
        old_phasegrad = np.array(old_phasegrad[sort_ind]).squeeze()
        old_phasegraderr = np.array(old_phasegraderr[sort_ind]).squeeze()
        old_mjd = np.array(old_mjd[sort_ind]).squeeze()
        old_rcvrs = np.array(old_rcvrs[sort_ind]).squeeze()
        old_freq = np.array(old_freq[sort_ind]).squeeze()
        old_tobs = np.array(old_tobs[sort_ind]).squeeze()
        # old_scintle_num = np.array(old_scintle_num[sort_ind]).squeeze()
        old_bw = np.array(old_bw[sort_ind]).squeeze()
        # old_scint_param_method = \
        #     np.array(old_scint_param_method[sort_ind]).squeeze()

        # Used to filter the data
        if filtered:
            indicies = np.argwhere((old_tauerr < filtered_tau*old_tau) *
                                   (old_dnuerr < filtered_dnu*old_dnu))

            old_df = old_df[indicies].squeeze()
            old_dnu = old_dnu[indicies].squeeze()
            old_dnu_est = old_dnu_est[indicies].squeeze()
            old_dnuerr = old_dnuerr[indicies].squeeze()
            old_tau = old_tau[indicies].squeeze()
            old_tauerr = old_tauerr[indicies].squeeze()
            old_phasegrad = old_phasegrad[indicies].squeeze()
            old_phasegraderr = old_phasegraderr[indicies].squeeze()
            old_mjd = old_mjd[indicies].squeeze()
            old_rcvrs = old_rcvrs[indicies].squeeze()
            old_freq = old_freq[indicies].squeeze()
            old_tobs = old_tobs[indicies].squeeze()
            # old_scintle_num = old_scintle_num[indicies].squeeze()
            old_bw = old_bw[indicies].squeeze()
            # old_scint_param_method = \
            # old_scint_param_method[indicies].squeeze()
        old_phase_n_mjd_file = outdir + 'old_phase_data.txt'
        # old_mjd_annual = old_mjd % 365.2425

        if os.path.exists(old_phase_n_mjd_file):
            print('Getting phase measurements ...')
            old_phase_n_mjd = np.genfromtxt(old_phase_n_mjd_file)
            # old_mjd_test = old_phase_n_mjd[:, 0]
            # if len(np.argwhere(old_mjd_test != old_mjd)) != 0:
            #     old_ssb_delays = get_ssb_delay(old_mjd, pars['RAJ'],
            #                                    pars['DECJ'])
            #     old_mjd += np.divide(old_ssb_delays, 86400)  # add ssb delay
            #     """
            #     Model Viss
            #     """
            #     print('Getting Earth velocity')
            #     old_vearth_ra, old_vearth_dec = get_earth_velocity(
            #         old_mjd, pars['RAJ'], pars['DECJ'])
            #     print('Getting true anomaly')
            #     old_U = get_true_anomaly(old_mjd, pars)

            #     old_vearth_ra = old_vearth_ra.squeeze()
            #     old_vearth_dec = old_vearth_dec.squeeze()

            #     old_om = pars['OM'] + pars['OMDOT']*(old_mjd - pars['T0']) / \
            #         365.2425
            #     # compute orbital phase
            #     old_phase = old_U*180/np.pi + old_om
            #     old_phase = old_phase % 360
            #     old_phase_n_mjd = np.zeros((len(old_mjd), 2))
            #     old_phase_n_mjd[:, 0] = old_mjd
            #     old_phase_n_mjd[:, 1] = old_phase
            #     np.savetxt(old_phase_n_mjd_file, old_phase_n_mjd, fmt='%s')
            # else:
            old_phase = old_phase_n_mjd[:, 1]
        else:
            # old_ssb_delays = get_ssb_delay(old_mjd, pars['RAJ'], pars['DECJ'])
            # old_mjd += np.divide(old_ssb_delays, 86400)  # add ssb delay
            """
            Model Viss
            """
            print('Getting Earth velocity')
            old_vearth_ra, old_vearth_dec = get_earth_velocity(old_mjd, pars['RAJ'],
                                                       pars['DECJ'])
            print('Getting true anomaly')
            old_U = get_true_anomaly(old_mjd, pars)

            old_vearth_ra = old_vearth_ra.squeeze()
            old_vearth_dec = old_vearth_dec.squeeze()

            old_om = pars['OM'] + pars['OMDOT']*(old_mjd - pars['T0'])/365.2425
            # compute orbital phase
            old_phase = old_U*180/np.pi + old_om
            old_phase = old_phase % 360
            old_phase_n_mjd = np.zeros((len(old_mjd), 2))
            old_phase_n_mjd[:, 0] = old_mjd
            old_phase_n_mjd[:, 1] = old_phase
            np.savetxt(old_phase_n_mjd_file, old_phase_n_mjd, fmt='%s')

        if compare:
            # Combine the data
            # total_freq = np.concatenate((freq, old_freq))
            # total_dnu = np.concatenate((dnu, old_dnu))
            # total_dnuerr = np.concatenate((dnuerr, old_dnuerr))
            # total_phase = np.concatenate((phase, old_phase))

            # match_index = []
            # for i in range(0, np.min([len(old_mjd), len(mjd)])):
            #     if mjd[i] == old_mjd[i]:
            #         match_index.append(i)
            # match_index = np.zeros((np.min([len(old_mjd), len(mjd)]), 2))
            # for i in range(0, len(mjd)):
            #     match_index[i, 0] == i
            #     for ii in range(0, len(old_mjd)):
            #         if mjd[i] == old_mjd[ii]:
            #             match_index[i, 1] == ii
            #             continue

            # match_index_1 = match_index[:, 0].astype(int)
            # match_index_2 = match_index[:, 1].astype(int)
            matching_indices = [(np.where(old_mjd == x)[0][0],
                                np.where(mjd == x)[0][0]) for x in set(mjd) &
                                set(old_mjd)]
            matching_indices_array = np.asarray(matching_indices)
            matching_indices_array1 = matching_indices_array[:, 0]
            matching_indices_array2 = matching_indices_array[:, 1]

            # compare_freq = freq[match_index]
            compare_dnu = dnu[matching_indices_array1].flatten()
            compare_dnuerr = dnuerr[matching_indices_array1].flatten()
            compare_tau = tau[matching_indices_array1].flatten()
            compare_tauerr = tauerr[matching_indices_array1].flatten()
            compare_phasegrad = phasegrad[matching_indices_array1].flatten()
            compare_phasegraderr = \
                phasegraderr[matching_indices_array1].flatten()

            # compare_old_freq = old_freq[match_index]
            compare_old_dnu = old_dnu[matching_indices_array2].flatten()
            compare_old_dnuerr = \
                old_dnuerr[matching_indices_array2].flatten()
            compare_old_tau = old_tau[matching_indices_array2].flatten()
            compare_old_tauerr = \
                old_tauerr[matching_indices_array2].flatten()
            compare_old_phasegrad = \
                old_phasegrad[matching_indices_array2].flatten()
            compare_old_phasegraderr = \
                old_phasegraderr[matching_indices_array2].flatten()

            # comparing dnu
            fig = plt.figure(figsize=(20, 10))
            fig.subplots_adjust(hspace=0.5, wspace=0.5)
            sc = plt.scatter(compare_dnu, compare_old_dnu, c='C0', s=Size,
                             alpha=0.4)
            plt.errorbar(compare_dnu, compare_old_dnu, xerr=compare_dnuerr,
                         yerr=compare_old_dnuerr, fmt=' ',
                         ecolor='k', elinewidth=2, capsize=3, alpha=0.3)
            yl = plt.ylim()
            xl = plt.xlim()
            plt.plot([xl[0], yl[1]], [xl[0], yl[1]], 'k')
            plt.xlabel(r'MultiFitPhasegradient: $\Delta\nu_d$ (MHz)')
            plt.ylabel(r'SingleFitPhasegradient: $\Delta\nu_d$ (MHz)')
            plt.xlim(xl)
            plt.ylim(yl)
            plt.savefig("/Users/jacobaskew/Desktop/Dnu_compare.png")
            plt.savefig(plotdir+"Dnu_compare.pdf", dpi=400)
            plt.show()
            plt.close()
            # comparing tau
            fig = plt.figure(figsize=(20, 10))
            fig.subplots_adjust(hspace=0.5, wspace=0.5)
            sc = plt.scatter(compare_tau, compare_old_tau, c='C1', s=Size,
                             alpha=0.4)
            plt.errorbar(compare_tau, compare_old_tau, xerr=compare_tauerr,
                         yerr=compare_old_tauerr, fmt=' ',
                         ecolor='k', elinewidth=2, capsize=3, alpha=0.3)
            yl = plt.ylim()
            xl = plt.xlim()
            plt.plot([xl[0], yl[1]], [xl[0], yl[1]], 'k')
            plt.xlabel(r'MultiFitPhasegradient: $\tau_d$ (s)')
            plt.ylabel(r'SingleFitPhasegradient: $\tau_d$ (s)')
            plt.xlim(xl)
            plt.ylim(yl)
            plt.savefig("/Users/jacobaskew/Desktop/Tau_compare.png")
            plt.savefig(plotdir+"Tau_compare.pdf", dpi=400)
            plt.show()
            plt.close()
            # comparing phasegrad
            fig = plt.figure(figsize=(20, 10))
            fig.subplots_adjust(hspace=0.5, wspace=0.5)
            sc = plt.scatter(compare_phasegrad, compare_old_phasegrad, c='C2',
                             s=Size, alpha=0.4)
            plt.errorbar(compare_phasegrad, compare_old_phasegrad,
                         xerr=compare_phasegraderr,
                         yerr=compare_old_phasegraderr, fmt=' ',
                         ecolor='k', elinewidth=2, capsize=3, alpha=0.3)
            yl = plt.ylim()
            xl = plt.xlim()
            plt.plot([xl[0], yl[1]], [xl[0], yl[1]], 'k')
            plt.xlabel(r'MultiFitPhasegradient: $\phi$ (mins/MHz)')
            plt.ylabel(r'SingleFitPhasegradient: $\phi$ (mins/MHz)')
            plt.xlim(xl)
            plt.ylim(yl)
            plt.savefig("/Users/jacobaskew/Desktop/phasegrad_compare.png")
            plt.savefig(plotdir+"phasegrad_compare.pdf", dpi=400)
            plt.show()
            plt.close()

    # This should merge the data from all of the observations given and save
    # them to a big txt file and saves many copies
    #
    # viss = np.loadtxt(datadir + 'Full_VissData.txt', dtype='float')
    # visserr = np.loadtxt(datadir + 'Full_VisserrData.txt', dtype='float')
    # mjd = np.loadtxt(datadir + 'Full_MJDData.txt', dtype='float')
    # freqMHz = np.loadtxt(datadir + 'Full_FreqData.txt', dtype='float')
    # dnu = np.loadtxt(datadir + 'Full_DnuData.txt', dtype='float')
    # dnuerr = np.loadtxt(datadir + 'Full_DnuerrData.txt', dtype='float')
    # tau = np.loadtxt(datadir + 'Full_TauData.txt', dtype='float')
    # tauerr = np.loadtxt(datadir + 'Full_TauerrData.txt', dtype='float')
    # phase = np.loadtxt(datadir + 'Full_PhaseData.txt', dtype='float')
    # U = np.loadtxt(datadir + 'Full_UData.txt', dtype='float')
    # ve_ra = np.loadtxt(datadir + 'Full_ve_raData.txt', dtype='float')
    # ve_dec = np.loadtxt(datadir + 'Full_ve_decData.txt', dtype='float')

    # mjd = np.loadtxt(filepath+'_mjd.txt', dtype='float')
    # freqMHz = np.loadtxt(filepath+'_freqMHz.txt', dtype='float')
    # dnu = np.loadtxt(filepath+'_dnu.txt', dtype='float')
    # dnuerr = np.loadtxt(filepath+'_dnuerr.txt', dtype='float')
    # tau = np.loadtxt(filepath+'_tau.txt', dtype='float')
    # tauerr = np.loadtxt(filepath+'_tauerr.txt', dtype='float')
    # name = np.loadtxt(filepath+'_name.txt', dtype='float')
    # phasegrad = np.loadtxt(filepath+'_phasegrad.txt', dtype='float')
    # phasegraderr = np.loadtxt(filepath+'_phasegraderr.txt', dtype='float')
    # phase = np.loadtxt(filepath+'_phase.txt', dtype='float')
    # U = np.loadtxt(filepath+'_U.txt', dtype='float')
    # ve_ra = np.loadtxt(filepath+'_ve_ra.txt', dtype='float')
    # ve_dec = np.loadtxt(filepath+'_ve_dec.txt', dtype='float')
    # viss = np.loadtxt(filepath+'_viss.txt', dtype='float')
    # visserr = np.loadtxt(filepath+'_visserr.txt', dtype='float')
    # mjd_annual = mjd % 365.25

    # This section will do two things, load in data that already exists OR
    # Create data and save it        
    # if load_data:
    #     wd0 = '/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/'
    #     wd = wd0+'New/'
    #     outdir = wd + 'DataFiles/'
    #     psrname = 'J0737-3039A'
    #     if len(observations) == 1:
    #         filepath = wd + "DataFiles/" + observations[i] + "/"
    #     else:
    #         filepath = outdir + '/ScintData/'         
    #         # outfile_total = str(outdir) + \
    #         #     'J0737-3039A' + \
    #         #     '_freq'+str(freq_bin)+'_time'+str(time_bin) + \
    #         #     '_ScintillationResults_UHF_Total.txt'
    #         # filepath = str(outdir)+str(observation_date2)+'/J0737-3039A_' + \
    #         #     str(observation_date2)+'_freq'+str(freq_bin)+'_time' + \
    #         #     str(time_bin)+'_UHF'            
    #     if os.path.exists(filepath+'_visserr.txt') and not overwrite:
    #         mjd = np.loadtxt(filepath+'_mjd.txt', dtype='float')
    #         freqMHz = np.loadtxt(filepath+'_freqMHz.txt', dtype='float')
    #         df = np.loadtxt(filepath+'_df.txt', dtype='float')
    #         dnu = np.loadtxt(filepath+'_dnu.txt', dtype='float')
    #         dnuerr = np.loadtxt(filepath+'_dnuerr.txt', dtype='float')
    #         tau = np.loadtxt(filepath+'_tau.txt', dtype='float')
    #         tauerr = np.loadtxt(filepath+'_tauerr.txt', dtype='float')
    #         name = np.loadtxt(filepath+'_name.txt', dtype='str')
    #         phasegrad = np.loadtxt(filepath+'_phasegrad.txt', dtype='float')
    #         phasegraderr = np.loadtxt(filepath+'_phasegraderr.txt', dtype='float')
    #         phase = np.loadtxt(filepath+'_phase.txt', dtype='float')
    #         U = np.loadtxt(filepath+'_U.txt', dtype='float')
    #         # ve_ra = np.loadtxt(filepath+'_ve_ra.txt', dtype='float')
    #         # ve_dec = np.loadtxt(filepath+'_ve_dec.txt', dtype='float')
    #         viss = np.loadtxt(filepath+'_viss.txt', dtype='float')
    #         visserr = np.loadtxt(filepath+'_visserr.txt', dtype='float')
    #         mjd_annual = mjd % 365.25
    #     else:
    #         for i in range(0, len(observations)):
    #             observation_date = observations[i] + '/'
    #             observation_date2 = observations[i]
    #             wd0 = '/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/'
    #             wd = wd0+'New/'
    #             outdir = wd + 'DataFiles/'
    #             filedir = str(outdir)+str(observation_date)
    #             outfile = \
    #                 str(filedir)+str(psrname)+'_'+str(observation_date2) + \
    #                 '_freq'+str(freq_bin)+'_time'+str(time_bin) + \
    #                 '_ScintillationResults_UHF.txt'
    #             outfile_total = str(outdir)+str(psrname)+'_freq' + \
    #                 str(freq_bin)+'_time'+str(time_bin) + \
    #                 '_ScintillationResults_UHF_Total.txt'
    #             if i == 0:
    #                 Data0 = Table.read(outfile, format='ascii.csv')
    #                 if len(observations) == 1:
    #                     Data2 = Data0
    #             elif i == 1:
    #                 Data1 = Table.read(outfile, format='ascii.csv')
    #                 Data2 = vstack([Data0, Data1])
    #             else:
    #                 Data3 = Table.read(outfile, format='ascii.csv')
    #                 Data2 = vstack([Data2, Data3])
    #         np.savetxt(outfile_total, Data2, delimiter=',', fmt='%s')
    #         DataTest = np.genfromtxt(outfile_total, delimiter=',', dtype=str)
    #             # if weights_2dacf:
    #             #     if measure_tilt:
    #             #         TitleRow = ['name', 'mjd', 'freq', 'bw', 'tobs', 'dt', 'df',
    #             #                     'tau', 'tauerr', 'dnu', 'dnuerr', 'fse_tau',
    #             #                     'fse_dnu', 'scint_param_method', 'dnu_est',
    #             #                     'nscint', 'acf_tilt', 'acf_tilt_err', 'fse_tilt',
    #             #                     'acf_redchisqr', 'acf_model_redchisqr',
    #             #                     'phasegrad', 'phasegraderr', 'fse_phasegrad']
    #             #         DataTest = np.vstack((TitleRow, DataTest))
    #             #     else:
    #             #         TitleRow = ['name', 'mjd', 'freq', 'bw', 'tobs', 'dt', 'df',
    #             #                     'tau', 'tauerr', 'dnu', 'dnuerr', 'fse_tau',
    #             #                     'fse_dnu', 'scint_param_method', 'dnu_est',
    #             #                     'nscint', 'acf_redchisqr', 'acf_model_redchisqr',
    #             #                     'phasegrad', 'phasegraderr', 'fse_phasegrad']
    #             # else:
    #             #     if measure_tilt:
    #             #         TitleRow = ['name', 'mjd', 'freq', 'bw', 'tobs', 'dt', 'df',
    #             #                     'tau', 'tauerr', 'dnu', 'dnuerr', 'fse_tau',
    #             #                     'fse_dnu', 'scint_param_method', 'dnu_est',
    #             #                     'nscint', 'acf_tilt', 'acf_tilt_err', 'fse_tilt',
    #             #                     'phasegrad', 'phasegraderr', 'fse_phasegrad']
    #             #         DataTest = np.vstack((TitleRow, DataTest))
    #             #     else:
    #             #         TitleRow = ['name', 'mjd', 'freq', 'bw', 'tobs', 'dt', 'df',
    #             #                     'tau', 'tauerr', 'dnu', 'dnuerr', 'fse_tau',
    #             #                     'fse_dnu', 'scint_param_method', 'dnu_est',
    #             #                     'nscint', 'phasegrad', 'phasegraderr',
    #             #                     'fse_phasegrad']
    #             # DataTest = np.vstack((TitleRow, DataTest))
    #         np.savetxt(outfile_total, DataTest, delimiter=',', fmt='%s')
    #             # filenames = ['file1.txt', 'file2.txt', ...]
    #             # with open('path/to/output/file', 'w') as outfile:
    #             #     for fname in filenames:
    #             #         with open(fname) as infile:
    #             #             for line in infile:
    #             #                 outfile.write(line)
    #         print("outfile_total", outfile_total)
    #         print()
    #         params = read_results(outfile_total)
    #         pars = read_par(str(par_dir) + str(psrname) + '.par')
        
    #         # Read in arrays
    #         mjd = float_array_from_dict(params, 'mjd')
    #         len_mjd_before = len(mjd)
    #         df = float_array_from_dict(params, 'df')  # channel bandwidth
    #         dnu = float_array_from_dict(params, 'dnu')  # scint bandwidth
    #         dnuerr = float_array_from_dict(params, 'dnuerr')
    #         tau = float_array_from_dict(params, 'tau')
    #         tauerr = float_array_from_dict(params, 'tauerr')
    #         freq = float_array_from_dict(params, 'freq')
    #         bw = float_array_from_dict(params, 'bw')
    #         name = np.asarray(params['name'])
    #         tobs = float_array_from_dict(params, 'tobs')  # tobs in second
    #         scint_param_method = np.asarray(params['scint_param_method'])
    #         phasegrad = float_array_from_dict(params, 'phasegrad')
    #         phasegraderr = float_array_from_dict(params, 'phasegraderr')
    #         if weights_2dacf:
    #             acf_redchisqr = np.asarray(params['acf_redchisqr'])
    #             acf_redchisqr = acf_redchisqr.astype(np.float)
    #             acf_model_redchisqr = np.asarray(params['acf_model_redchisqr'])
    #             acf_model_redchisqr = acf_model_redchisqr.astype(np.float)
    #         if measure_tilt:
    #             acf_tilt = float_array_from_dict(params, 'acf_tilt')
    #             acf_tilt_err = float_array_from_dict(params, 'acf_tilt_err')
    
    #         # Sort by MJD
    #         sort_ind = np.argsort(mjd)
    
    #         df = np.array(df[sort_ind]).squeeze()
    #         dnu = np.array(dnu[sort_ind]).squeeze()
    #         dnuerr = np.array(dnuerr[sort_ind]).squeeze()
    #         tau = np.array(tau[sort_ind]).squeeze()
    #         tauerr = np.array(tauerr[sort_ind]).squeeze()
    #         mjd = np.array(mjd[sort_ind]).squeeze()
    #         freq = np.array(freq[sort_ind]).squeeze()
    #         tobs = np.array(tobs[sort_ind]).squeeze()
    #         name = np.array(name[sort_ind]).squeeze()
    #         phasegrad = np.array(phasegrad[sort_ind]).squeeze()
    #         phasegraderr = np.array(phasegraderr[sort_ind]).squeeze()
    #         bw = np.array(bw[sort_ind]).squeeze()
    #         scint_param_method = np.array(scint_param_method[sort_ind]).squeeze()
    #         if weights_2dacf:
    #             acf_redchisqr = np.array(acf_redchisqr[sort_ind]).squeeze()
    #             acf_model_redchisqr = \
    #                 np.array(acf_model_redchisqr[sort_ind]).squeeze()
    #         if measure_tilt:
    #             acf_tilt = np.array(acf_tilt[sort_ind]).squeeze()
    #             acf_tilt_err = np.array(acf_tilt_err[sort_ind]).squeeze()
    
    #         # Used to filter the data
    #         if filtered:
    #             if weights_2dacf:
    #                 indicies = np.argwhere((tauerr < filtered_tau*tau) *
    #                                        (dnuerr < filtered_dnu*dnu) *
    #                                        (scint_param_method == "acf2d_approx") *
    #                                        (acf_redchisqr <
    #                                         filtered_acf_redchisqr))
    #             else:
    #                 indicies = np.argwhere((tauerr < filtered_tau*tau) *
    #                                        (dnuerr < filtered_dnu*dnu) *
    #                                        (scint_param_method == "acf2d_approx"))
    
    #             df = df[indicies].squeeze()
    #             dnu = dnu[indicies].squeeze()
    #             dnuerr = dnuerr[indicies].squeeze()
    #             tau = tau[indicies].squeeze()
    #             tauerr = tauerr[indicies].squeeze()
    #             mjd = mjd[indicies].squeeze()
    #             freq = freq[indicies].squeeze()
    #             tobs = tobs[indicies].squeeze()
    #             name = name[indicies].squeeze()
    #             bw = bw[indicies].squeeze()
    #             phasegrad = phasegrad[indicies].squeeze()
    #             phasegraderr = phasegraderr[indicies].squeeze()
    #             scint_param_method = scint_param_method[indicies].squeeze()
    #             if weights_2dacf:
    #                 acf_redchisqr = acf_redchisqr[indicies].squeeze()
    #                 acf_model_redchisqr = acf_model_redchisqr[indicies].squeeze()
    #             if measure_tilt:
    #                 acf_tilt = acf_tilt[indicies].squeeze()
    #                 acf_tilt_err = acf_tilt_err[indicies].squeeze()
    #             print("========== filtering was done ==========")
    #             print("========== fractional difference was ... ==========")
    #             print("========== " + str(round(len(mjd)/len_mjd_before*100, 3)) +
    #                   " ==========")
    #             freqMHz = freq
    #             np.savetxt(filepath+'_freqMHz.txt', freqMHz, delimiter=',', fmt='%s')
    #             np.savetxt(filepath+'_df.txt', df, delimiter=',', fmt='%s')
    #             np.savetxt(filepath+'_dnu.txt', dnu, delimiter=',', fmt='%s')
    #             np.savetxt(filepath+'_dnuerr.txt', dnuerr, delimiter=',', fmt='%s')
    #             np.savetxt(filepath+'_tau.txt', tau, delimiter=',', fmt='%s')
    #             np.savetxt(filepath+'_tauerr.txt', tauerr, delimiter=',', fmt='%s')
    #             np.savetxt(filepath+'_name.txt', name, delimiter=',', fmt='%s')
    #             np.savetxt(filepath+'_phasegrad.txt', phasegrad, delimiter=',', fmt='%s')
    #             np.savetxt(filepath+'_phasegraderr.txt', phasegraderr, delimiter=',', fmt='%s')
    #             #
    #             pars = read_par(str(par_dir) + str(psrname) + '.par')
    #             params = pars_to_params(pars)
                
    #             np.savetxt(filepath+'noTOBS_noSSB_mjd.txt', mjd, delimiter=',', fmt='%s')
                                    
    #             mjd += (tobs / 2) / 86400
                
    #             np.savetxt(filepath+'TOBS_noSSB_mjd.txt', mjd, delimiter=',', fmt='%s')

    #             mjd_annual = mjd % 365.25
    #             ssb_delays = get_ssb_delay(mjd, pars['RAJ'], pars['DECJ'])
    #             mjd += np.divide(ssb_delays, 86400)  # add ssb delay
                
    #             np.savetxt(filepath+'_mjd.txt', mjd, delimiter=',', fmt='%s')
                
    #             mjd -= (tobs / 2) / 86400
                
    #             np.savetxt(filepath+'noTOBS_SSB_mjd.txt', mjd, delimiter=',', fmt='%s')
                
    #             vearth_ra, vearth_dec = get_earth_velocity(mjd, pars['RAJ'], pars['DECJ'])
    #             U = get_true_anomaly(mjd, pars)
    #             vearth_ra = vearth_ra.squeeze()
    #             vearth_dec = vearth_dec.squeeze()
    #             om = pars['OM'] + pars['OMDOT']*(mjd - pars['T0'])/365.2425
    #             phase = U*180/np.pi + om
    #             phase = phase % 360
    #             #
    #             np.savetxt(filepath+'_phase.txt', phase, delimiter=',', fmt='%s')
    #             np.savetxt(filepath+'_U.txt', U, delimiter=',', fmt='%s')
    #             np.savetxt(filepath+'_ve_ra.txt', vearth_ra, delimiter=',', fmt='%s')
    #             np.savetxt(filepath+'_ve_dec.txt', vearth_dec, delimiter=',', fmt='%s')
    #             #
    #             D = 0.735  # kpc
    #             # D_err = 0.06
    #             s = 0.7
    #             kappa = 1
    #             freqGHz = freqMHz / 1e3

    #             Aiss = kappa * 3.347290399e4 * np.sqrt((2*(1-s))/(s))
    #             viss = Aiss * (np.sqrt(D*dnu))/(freqGHz*tau)
    #             visserr = viss * np.sqrt((dnuerr/(2*dnu))**2+(-tauerr/tau)**2)

    #             # Aiss = kappa * 2.78e4 * np.sqrt((2*(1-s))/(s))
    #             # viss = Aiss * (np.sqrt(D*dnu))/(freqGHz*tau)
    #             # visserr = viss * np.sqrt((D_err/(2*D))**2 +(dnuerr/(2*dnu))**2 +
    #             #                          (-tauerr/tau)**2)
    #             #
    #             np.savetxt(filepath+'_viss.txt', viss, delimiter=',', fmt='%s')
    #             np.savetxt(filepath+'_visserr.txt', visserr, delimiter=',', fmt='%s')

        # name_num = []
        # for i in range(0, len(name)):
        #     for ii in range(0, len(np.unique(name))):
        #         if name[i] == np.unique(name)[ii]:
        #             name_num.append(ii)
        # name_num = np.asarray(name_num)
    # if merge_data and len(observations) > 1:
    #     for i in range(0, len(observations)):
    #         observation_date = observations[i] + '/'
    #         observation_date2 = observations[i]
    #         wd0 = '/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/'
    #         wd = wd0+'New/'
    #         outdir = wd + 'DataFiles/'
    #         filedir = str(outdir)+str(observation_date)
    #         outfile = \
    #             str(filedir)+str(psrname)+'_'+str(observation_date2) + \
    #             '_freq'+str(freq_bin)+'_time'+str(time_bin) + \
    #             '_ScintillationResults_UHF.txt'
    #         outfile_total = str(outdir)+str(psrname)+'_freq' + \
    #             str(freq_bin)+'_time'+str(time_bin) + \
    #             '_ScintillationResults_UHF_Total.txt'
    #         if i == 0:
    #             Data0 = Table.read(outfile, format='ascii.csv')
    #             if len(observations) == 1:
    #                 Data2 = Data0
    #         elif i == 1:
    #             Data1 = Table.read(outfile, format='ascii.csv')
    #             Data2 = vstack([Data0, Data1])
    #         else:
    #             Data3 = Table.read(outfile, format='ascii.csv')
    #             Data2 = vstack([Data2, Data3])
    #     np.savetxt(outfile_total, Data2, delimiter=',', fmt='%s')
    #     DataTest = np.genfromtxt(outfile_total, delimiter=',', dtype=str)
    #     if weights_2dacf:
    #         if measure_tilt:
    #             TitleRow = ['name', 'mjd', 'freq', 'bw', 'tobs', 'dt', 'df',
    #                         'tau', 'tauerr', 'dnu', 'dnuerr', 'fse_tau',
    #                         'fse_dnu', 'scint_param_method', 'dnu_est',
    #                         'nscint', 'acf_tilt', 'acf_tilt_err', 'fse_tilt',
    #                         'acf_redchisqr', 'acf_model_redchisqr',
    #                         'phasegrad', 'phasegraderr', 'fse_phasegrad']
    #             DataTest = np.vstack((TitleRow, DataTest))
    #         else:
    #             TitleRow = ['name', 'mjd', 'freq', 'bw', 'tobs', 'dt', 'df',
    #                         'tau', 'tauerr', 'dnu', 'dnuerr', 'fse_tau',
    #                         'fse_dnu', 'scint_param_method', 'dnu_est',
    #                         'nscint', 'acf_redchisqr', 'acf_model_redchisqr',
    #                         'phasegrad', 'phasegraderr', 'fse_phasegrad']
    #     else:
    #         if measure_tilt:
    #             TitleRow = ['name', 'mjd', 'freq', 'bw', 'tobs', 'dt', 'df',
    #                         'tau', 'tauerr', 'dnu', 'dnuerr', 'fse_tau',
    #                         'fse_dnu', 'scint_param_method', 'dnu_est',
    #                         'nscint', 'acf_tilt', 'acf_tilt_err', 'fse_tilt',
    #                         'phasegrad', 'phasegraderr', 'fse_phasegrad']
    #             DataTest = np.vstack((TitleRow, DataTest))
    #         else:
    #             TitleRow = ['name', 'mjd', 'freq', 'bw', 'tobs', 'dt', 'df',
    #                         'tau', 'tauerr', 'dnu', 'dnuerr', 'fse_tau',
    #                         'fse_dnu', 'scint_param_method', 'dnu_est',
    #                         'nscint', 'phasegrad', 'phasegraderr',
    #                         'fse_phasegrad']
    #     DataTest = np.vstack((TitleRow, DataTest))
    #     np.savetxt(outfile_total, DataTest, delimiter=',', fmt='%s')
    #     # filenames = ['file1.txt', 'file2.txt', ...]
    #     # with open('path/to/output/file', 'w') as outfile:
    #     #     for fname in filenames:
    #     #         with open(fname) as infile:
    #     #             for line in infile:
    #     #                 outfile.write(line)


    # if spectral_index:
    #     if compare:
    #         # Here i want to explore all the possible fits to the entire data
    #         #
    #         total_phase_list = []
    #         total_dnu_list = []
    #         total_freq_list = []
    #         total_dnuerr_list = []
    #         total_df_list = []
    #         for i in range(0, 18):
    #             total_dnu_list.append(
    #                 np.asarray(
    #                     total_dnu[np.argwhere((total_phase > i*20) *
    #                                           (total_phase < (i+1)*20))]))
    #             total_dnuerr_list.append(
    #                 np.asarray(
    #                     total_dnuerr[np.argwhere((total_phase > i*20) *
    #                                              (total_phase < (i+1)*20))]))
    #             total_phase_list.append(
    #                 np.asarray(
    #                     total_phase[np.argwhere((total_phase > i*20) *
    #                                             (total_phase < (i+1)*20))]))
    #             total_freq_list.append(
    #                 np.asarray(
    #                     total_freq[np.argwhere((total_phase > i*20) *
    #                                            (total_phase < (i+1)*20))]))
    #             total_df_list.append(
    #                 np.asarray(df[np.argwhere((total_phase > i*20) *
    #                                           (total_phase < (i+1)*20))]))
    #         total_dnu_array = np.asarray(total_dnu_list)
    #         total_freq_array = np.asarray(total_freq_list)
    #         total_dnuerr_array = np.asarray(total_dnuerr_list)
    #         total_df_array = np.asarray(total_df_list)
    #         #
    #         phase_list2 = []
    #         dnu_list2 = []
    #         freq_list2 = []
    #         dnuerr_list2 = []
    #         df_list2 = []
    #         for i in range(0, 18):
    #             dnu_list2.append(
    #                 np.asarray(old_dnu[np.argwhere((old_phase > i*20) *
    #                                                (old_phase < (i+1)*20))]))
    #             dnuerr_list2.append(
    #                 np.asarray(
    #                     old_dnuerr[np.argwhere((old_phase > i*20) *
    #                                            (old_phase < (i+1)*20))]))
    #             phase_list2.append(
    #                 np.asarray(old_phase[np.argwhere((old_phase > i*20) *
    #                                                  (old_phase < (i+1)*20))]))
    #             freq_list2.append(
    #                 np.asarray(old_freq[np.argwhere((old_phase > i*20) *
    #                                                 (old_phase < (i+1)*20))]))
    #             df_list2.append(
    #                 np.asarray(old_df[np.argwhere((old_phase > i*20) *
    #                                               (old_phase < (i+1)*20))]))
    #         dnu_array2 = np.asarray(dnu_list2)
    #         freq_array2 = np.asarray(freq_list2)
    #         dnuerr_array2 = np.asarray(dnuerr_list2)
    #         df_array2 = np.asarray(df_list2)
    #         #
    #     phase_list = []
    #     dnu_list = []
    #     freq_list = []
    #     dnuerr_list = []
    #     df_list = []
    #     for i in range(0, 18):
    #         if df[np.argwhere((phase > i*20) * (phase < (i+1)*20))] == ' ':
    #             continue
    #         dnu_list.append(
    #             np.asarray(dnu[np.argwhere((phase > i*20) *
    #                                        (phase < (i+1)*20))]))
    #         dnuerr_list.append(
    #             np.asarray(
    #                 dnuerr[np.argwhere((phase > i*20) *
    #                                    (phase < (i+1)*20))]))
    #         phase_list.append(
    #             np.asarray(phase[np.argwhere((phase > i*20) *
    #                                          (phase < (i+1)*20))]))
    #         freq_list.append(
    #             np.asarray(freq[np.argwhere((phase > i*20) *
    #                                         (phase < (i+1)*20))]))
    #         df_list.append(
    #             np.asarray(df[np.argwhere((phase > i*20) *
    #                                       (phase < (i+1)*20))]))
    #     dnu_array = np.asarray(dnu_list)
    #     freq_array = np.asarray(freq_list)
    #     dnuerr_array = np.asarray(dnuerr_list)
    #     df_array = np.asarray(df_list)
    #     #
    #     Slopes1 = []
    #     Slopeerrs1 = []
    #     Slopes2 = []
    #     Slopeerrs2 = []
    #     Slopes3 = []
    #     Slopeerrs3 = []
    #     for i in range(0, 18):
    #         # new method
    #         #
    #         unresolved_fraction = 1
    #         condition = dnu_array[i] > unresolved_fraction*df_array[1][0][0]
    #         xdata1 = freq_array[i][np.argwhere(condition)].flatten()
    #         if len(xdata1) == 0:
    #             continue
    #         ydata1 = dnu_array[i][np.argwhere(condition)].flatten()
    #         if len(ydata1) == 0:
    #             continue
    #         ydataerr1 = dnuerr_array[i][np.argwhere(condition)].flatten()
    #         if len(ydataerr1) == 0:
    #             continue
    #         xdata4 = np.linspace(np.min(freq_array[1])*0.75,
    #                              np.max(freq_array[1])*1.25,
    #                              1000)
    #         #
    #         if compare:
    #             condition2 = dnu_array2[i] > unresolved_fraction*df_array2[i]
    #             xdata2 = freq_array2[i][np.argwhere(condition2)].flatten()
    #             ydata2 = dnu_array2[i][np.argwhere(condition2)].flatten()
    #             ydataerr2 = dnuerr_array2[i][np.argwhere(condition2)].flatten()
    #             #
    #             condition3 = total_dnu_array[i] > \
    #                 unresolved_fraction*total_df_array[i]
    #             xdata3 = total_freq_array[i][np.argwhere(condition3)].flatten()
    #             ydata3 = total_dnu_array[i][np.argwhere(condition3)].flatten()
    #             ydataerr3 = \
    #                 total_dnuerr_array[i][np.argwhere(condition3)].flatten()
    #             xdata4 = np.linspace(np.min(total_freq_array[1])*0.75,
    #                                  np.max(total_freq_array[1])*1.25,
    #                                  1000)

    #             # plotting spectral fit for data at that phase
    #             if xdata1.size > 0 and xdata2.size > 0:
                    # fig = plt.figure(figsize=(20, 10))
                    # ax = fig.add_subplot(1, 1, 1)
    #                 fig.subplots_adjust(hspace=0.5, wspace=0.5)
    #                 plt.scatter(xdata3, ydata3, c='C0', s=Size/4)
    #                 plt.errorbar(xdata3, ydata3, yerr=ydataerr3,
    #                              fmt=' ', ecolor='k',
    #                              elinewidth=2, capsize=3, alpha=0.55)
    #                 popt3, pcov3 = curve_fit(func1, xdata3, ydata3)
    #                 perr3 = np.sqrt(np.diag(pcov3))
    #                 plt.plot(xdata4, func1(xdata4, *popt3), 'C4',
    #                          label=r'Both $\alpha$='+str(round(popt3[0], 2)) +
    #                          r'$\pm$'+str(round(perr3[0], 2)))
    #                 plt.fill_between(xdata4.flatten(),
    #                                  func1(xdata4, *[popt3[0]+perr3[0],
    #                                                  popt3[1]]).flatten(),
    #                                  func1(xdata4, *[popt3[0]-perr3[0],
    #                                                  popt3[1]]).flatten(),
    #                                  alpha=0.5, color='C4')
    #                 Slopes3.append(popt3)
    #                 Slopeerrs3.append(perr3[0])
    #                 popt1, pcov1 = curve_fit(func1, xdata1, ydata1)
    #                 perr1 = np.sqrt(np.diag(pcov1))
    #                 plt.plot(xdata4, func1(xdata4, *popt1), 'C4',
    #                          label=r'Dataset1 $\alpha$='+str(round(popt1[0],
    #                                                                2)) +
    #                          r'$\pm$'+str(round(perr1[0], 2)))
    #                 xl = plt.xlim()
    #                 plt.fill_between(xdata4.flatten(),
    #                                  func1(xdata4, *[popt1[0]+perr1[0],
    #                                                  popt1[1]]).flatten(),
    #                                  func1(xdata4, *[popt1[0]-perr1[0],
    #                                                  popt1[1]]).flatten(),
    #                                  alpha=0.5, color='C4')
    #                 Slopes1.append(popt1)
    #                 Slopeerrs1.append(perr1[0])
    #                 popt2, pcov2 = curve_fit(func1, xdata2, ydata2)
    #                 perr2 = np.sqrt(np.diag(pcov2))
    #                 plt.plot(xdata4, func1(xdata4, *popt2), 'C4',
    #                          label=r'Dataset2 $\alpha$='+str(round(popt2[0],
    #                                                                2)) +
    #                          r'$\pm$'+str(round(perr2[0], 2)))
    #                 xl = plt.xlim()
    #                 plt.fill_between(xdata4.flatten(),
    #                                  func1(xdata4, *[popt2[0]+perr2[0],
    #                                                  popt2[1]]).flatten(),
    #                                  func1(xdata4, *[popt2[0]-perr2[0],
    #                                                  popt2[1]]).flatten(),
    #                                  alpha=0.5, color='C4')
    #                 Slopes2.append(popt2)
    #                 Slopeerrs2.append(perr2[0])
    #                 plt.plot(xl, (df_array2[0][0], df_array2[0][0]),
    #                          color='C2',
    #                          linestyle='dashed', label='Dataset2 df=' +
    #                          str(round(df_array2[0][0], 2))+'MHz')
    #                 plt.plot(xl, (df_array[1][0][0], df_array[1][0][0]),
    #                          color='C2', label='Dataset1 df=' +
    #                          str(round(df_array[1][0][0], 2))+'MHz')
    #                 plt.xlabel('Frequency (MHz)')
    #                 plt.ylabel('Scintillation Bandwidth (MHz)')
    #                 ax.legend()
    #                 ax.set_xscale('log')
    #                 ax.set_yscale('log')
    #                 plt.xlim(xl)
    #                 # plt.ylim(0, np.max(ydata1)*1.05)
    #                 plt.show()
    #             elif xdata1.size > 0 and xdata2.size <= 0:
    #                 fig = plt.figure(figsize=(20, 10))
    #                 ax = fig.add_subplot(1, 1, 1)
    #                 fig.subplots_adjust(hspace=0.5, wspace=0.5)
    #                 plt.scatter(xdata1, ydata1, c='C0', s=Size/4)
    #                 plt.errorbar(xdata1, ydata1, yerr=ydataerr1,
    #                              fmt=' ', ecolor='k',
    #                              elinewidth=2, capsize=3, alpha=0.55)
    #                 popt1, pcov1 = curve_fit(func1, xdata1, ydata1)
    #                 perr1 = np.sqrt(np.diag(pcov1))
    #                 plt.plot(xdata4, func1(xdata4, *popt1), 'C4',
    #                          label=r'Dataset1 $\alpha$='+str(round(popt1[0],
    #                                                                2)) +
    #                          r'$\pm$'+str(round(perr1[0], 2)))
    #                 xl = plt.xlim()
    #                 plt.fill_between(xdata4.flatten(),
    #                                  func1(xdata4, *[popt1[0]+perr1[0],
    #                                                  popt1[1]]).flatten(),
    #                                  func1(xdata4, *[popt1[0]-perr1[0],
    #                                                  popt1[1]]).flatten(),
    #                                  alpha=0.5, color='C4')
    #                 Slopes1.append(popt1)
    #                 Slopeerrs1.append(perr1[0])
    #                 plt.plot(xl, (df_array[0][0][0], df_array[0][0][0]),
    #                          color='C2', label='Dataset1 df=' +
    #                          str(round(df_array[0][0][0], 2))+'MHz')
    #                 plt.xlabel('Frequency (MHz)')
    #                 plt.ylabel('Scintillation Bandwidth (MHz)')
    #                 ax.legend()
    #                 ax.set_xscale('log')
    #                 ax.set_yscale('log')
    #                 plt.xlim(xl)
    #                 # plt.ylim(0, np.max(ydata1)*1.05)
    #                 plt.show()
    #             elif xdata2.size > 0 and xdata1.size <= 0:
    #                 fig = plt.figure(figsize=(20, 10))
    #                 ax = fig.add_subplot(1, 1, 1)
    #                 fig.subplots_adjust(hspace=0.5, wspace=0.5)
    #                 plt.scatter(xdata2, ydata2, c='C0', s=Size/4)
    #                 plt.errorbar(xdata2, ydata2, yerr=ydataerr2,
    #                              fmt=' ', ecolor='k',
    #                              elinewidth=2, capsize=3, alpha=0.55)
    #                 popt2, pcov2 = curve_fit(func1, xdata2, ydata2)
    #                 perr2 = np.sqrt(np.diag(pcov2))
    #                 plt.plot(xdata4, func1(xdata4, *popt2), 'C4',
    #                          label=r'Dataset2 $\alpha$=' +
    #                          str(round(popt2[0], 2)) +
    #                          r'$\pm$'+str(round(perr2[0], 2)))
    #                 xl = plt.xlim()
    #                 plt.fill_between(xdata4.flatten(),
    #                                  func1(xdata4, *[popt2[0]+perr2[0],
    #                                                  popt2[1]]).flatten(),
    #                                  func1(xdata4, *[popt2[0]-perr2[0],
    #                                                  popt2[1]]).flatten(),
    #                                  alpha=0.5, color='C4')
    #                 Slopes2.append(popt2)
    #                 Slopeerrs2.append(perr2[0])
    #                 plt.plot(xl, (df_array2[1][0], df_array2[1][0]),
    #                          color='C2',
    #                          linestyle='dashed', label='Dataset2 df=' +
    #                          str(round(df_array2[1][0], 2))+'MHz')
    #                 plt.xlabel('Frequency (MHz)')
    #                 plt.ylabel('Scintillation Bandwidth (MHz)')
    #                 ax.legend()
    #                 ax.set_xscale('log')
    #                 ax.set_yscale('log')
    #                 plt.xlim(xl)
    #                 plt.ylim(0, np.max(ydata2)*1.05)
    #                 plt.show()
    #             else:
    #                 print("something went bad")
    #                 continue

    #         else:
    #             fig = plt.figure(figsize=(20, 10))
    #             ax = fig.add_subplot(1, 1, 1)
    #             fig.subplots_adjust(hspace=0.5, wspace=0.5)
    #             plt.scatter(xdata1, ydata1, c='C0', s=Size/4)
    #             plt.errorbar(xdata1, ydata1, yerr=ydataerr1,
    #                          fmt=' ', ecolor='k',
    #                          elinewidth=2, capsize=3, alpha=0.55)
    #             popt1, pcov1 = curve_fit(func1, xdata1, ydata1)
    #             perr1 = np.sqrt(np.diag(pcov1))
    #             plt.plot(xdata4, func1(xdata4, *popt1), 'C4',
    #                      label=r'Dataset1 $\alpha$='+str(round(popt1[0],
    #                                                            2)) +
    #                      r'$\pm$'+str(round(perr1[0], 2)))
    #             xl = plt.xlim()
    #             plt.fill_between(xdata4.flatten(),
    #                              func1(xdata4, *[popt1[0]+perr1[0],
    #                                              popt1[1]]).flatten(),
    #                              func1(xdata4, *[popt1[0]-perr1[0],
    #                                              popt1[1]]).flatten(),
    #                              alpha=0.5, color='C4')
    #             Slopes1.append(popt1)
    #             Slopeerrs1.append(perr1[0])
    #             plt.plot(xl, (df_array[1][0][0], df_array[1][0][0]),
    #                      color='C2', label='Dataset1 df=' +
    #                      str(round(df_array[1][0][0], 2)) + 'MHz')
    #             plt.xlabel('Frequency (MHz)')
    #             plt.ylabel('Scintillation Bandwidth (MHz)')
    #             ax.legend()
    #             ax.set_xscale('log')
    #             ax.set_yscale('log')
    #             plt.xlim(np.min(xdata1)*0.95, np.max(xdata1)*1.05)
    #             # plt.ylim(0, np.max(ydata1)*1.05)
    #             plt.show()

    # # Final Plot
    #     Slopes_average1 = np.mean(Slopes1, axis=0)
    #     Slopeerrs_median1 = np.median(Slopeerrs1)
    #     if compare:
    #         Slopes_average2 = np.mean(Slopes2, axis=0)
    #         Slopeerrs_median2 = np.median(Slopeerrs2)
    #         Slopes_average3 = np.mean(Slopes3, axis=0)
    #         Slopeerrs_median3 = np.median(Slopeerrs3)

    #     fig = plt.figure(figsize=(20, 10))
    #     ax = fig.add_subplot(1, 1, 1)
    #     fig.subplots_adjust(hspace=0.5, wspace=0.5)
    #     ax.set_xscale('log')
    #     ax.set_yscale('log')
    #     if compare:
    #         xdata4 = np.linspace(np.min(total_freq)*0.75,
    #                              np.max(total_freq)*1.25, 1000)
    #         plt.scatter(freq, dnu, c='C1', s=Size, alpha=0.7)
    #         plt.scatter(old_freq, old_dnu, c='C0', marker='v', s=Size)
    #         plt.errorbar(total_freq, total_dnu, yerr=total_dnuerr, fmt=' ',
    #                      ecolor='k', elinewidth=4, capsize=6, alpha=0.4)
    #         # Dataset 1
    #         plt.plot(xdata4, func1(xdata4, *Slopes_average1), 'C1',
    #                  label=r'UHF $\alpha$=' +
    #                  str(round(Slopes_average1[0], 2))+r'$\pm$' +
    #                  str(round(Slopeerrs_median1, 2)))
    #         plt.fill_between(xdata4.flatten(),
    #                          func1(xdata4,
    #                                *[Slopes_average1[0]+Slopeerrs_median1,
    #                                  Slopes_average1[1]]).flatten(),
    #                          func1(xdata4,
    #                                *[Slopes_average1[0]-Slopeerrs_median1,
    #                                  Slopes_average1[1]]).flatten(),
    #                          alpha=0.2, color='C1')
    #         # Dataset 2
    #         plt.plot(xdata4, func1(xdata4, *Slopes_average2), 'C0',
    #                  label=r'Dataset2 $\alpha$=' +
    #                  str(round(Slopes_average2[0], 2))+r'$\pm$' +
    #                  str(round(Slopeerrs_median2, 2)))
    #         plt.fill_between(xdata4.flatten(),
    #                          func1(xdata4,
    #                                *[Slopes_average2[0]+Slopeerrs_median2,
    #                                  Slopes_average2[1]]).flatten(),
    #                          func1(xdata4,
    #                                *[Slopes_average2[0]-Slopeerrs_median2,
    #                                  Slopes_average2[1]]).flatten(),
    #                          alpha=0.2, color='C0')
    #         # Dataset 3
    #         plt.plot(xdata4, func1(xdata4, *Slopes_average3), 'C3',
    #                  label=r'ALL $\alpha$=' +
    #                  str(round(Slopes_average3[0], 2))+r'$\pm$' +
    #                  str(round(Slopeerrs_median3, 2)))
    #         plt.fill_between(xdata4.flatten(),
    #                          func1(xdata4,
    #                                *[Slopes_average3[0]+Slopeerrs_median3,
    #                                  Slopes_average3[1]]).flatten(),
    #                          func1(xdata4,
    #                                *[Slopes_average3[0]-Slopeerrs_median3,
    #                                  Slopes_average3[1]]).flatten(),
    #                          alpha=0.2, color='C3')
    #         plt.plot(xl, (old_df[0], old_df[0]), color='C2',
    #                  linestyle='dashed', label='Dataset2 Channel bw')
    #         plt.plot(xl, (df[0], df[0]), color='C2',
    #                  label='UHF Channel bw')
    #         plt.xlabel('Frequency (MHz)')
    #         plt.ylabel('Scintillation Bandwidth (MHz)')
    #         # ax.legend(fontsize='xx-small')
    #         ax.legend()
    #         # ax.set_xticks([700, 800, 1000, 1200, 1400, 1600, 1800])
    #         # ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    #         # ax.set_yticks([0.02, 0.1, 0.5, 1, 2, 4])
    #         # ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    #         # ax.axis([1, 100000, 1, 100000])
    #         # ax.loglog()
    #         # ax.xaxis.set_tick_params(length=5, width=2)
    #         # ax.yaxis.set_tick_params(length=5, width=2)
    #         # plt.xticks(np.arange(min(freq), max(freq)+1, 1000))
    #         # plt.yticks(np.arange(min(dnu), max(dnu)+1, 1000))
    #         # plt.xlim(min(xdata4), max(xdata4))
    #         plt.ylim(np.min(total_dnu)*0.90, np.max(total_dnu)*1.1)
    #         plt.xlim(np.min(total_freq)*0.95, np.max(total_freq)*1.05)
    #         plt.show()
    #         plt.close()
    #     else:
    #         xdata4 = np.linspace(np.min(freq)*0.75,
    #                              np.max(freq)*1.25, 1000)
    #         plt.scatter(freq, dnu, c='C1', s=Size, alpha=0.7)
    #         plt.errorbar(freq, dnu, yerr=dnuerr, fmt=' ',
    #                      ecolor='k', elinewidth=4, capsize=6, alpha=0.4)
    #         plt.plot(xdata4, func1(xdata4, *Slopes_average1), 'C4',
    #                  label=r'UHF $\alpha$=' +
    #                  str(round(Slopes_average1[0], 2))+r'$\pm$' +
    #                  str(round(Slopeerrs_median1, 2)))
    #         plt.fill_between(xdata4.flatten(),
    #                          func1(xdata4,
    #                                *[Slopes_average1[0]+Slopeerrs_median1,
    #                                  Slopes_average1[1]]).flatten(),
    #                          func1(xdata4,
    #                                *[Slopes_average1[0]-Slopeerrs_median1,
    #                                  Slopes_average1[1]]).flatten(),
    #                          alpha=0.2, color='C4')
    #         plt.plot(xl, (df[0], df[0]), color='C2',
    #                  label='UHF Channel bw')
    #         plt.xlabel('Frequency (MHz)')
    #         plt.ylabel('Scintillation Bandwidth (MHz)')
    #         # ax.legend(fontsize='xx-small')
    #         ax.legend()
    #         ax.set_xticks([650, 700, 800, 850, 900, 1000])
    #         ax.get_xaxis().set_major_formatter(
    #             matplotlib.ticker.ScalarFormatter())
    #         ax.set_yticks([0.01, 0.02, 0.05, 0.1, 0.2, 0.3])
    #         ax.get_yaxis().set_major_formatter(
    #             matplotlib.ticker.ScalarFormatter())
    #         # ax.axis([1, 100000, 1, 100000])
    #         # ax.loglog()
    #         # ax.xaxis.set_tick_params(length=5, width=2)
    #         # ax.yaxis.set_tick_params(length=5, width=2)
    #         # plt.xticks(np.arange(min(freq), max(freq)+1, 1000))
    #         # plt.yticks(np.arange(min(dnu), max(dnu)+1, 1000))
    #         # plt.xlim(min(xdata4), max(xdata4))
    #         plt.ylim(np.min(dnu)*0.90, np.max(dnu)*1.1)
    #         plt.xlim(np.min(freq)*0.95, np.max(freq)*1.05)
    #         plt.savefig('/Users/jacobaskew/Desktop/FreqVdnu.pdf', dpi=400)
    #         plt.savefig('/Users/jacobaskew/Desktop/FreqVdnu.png', dpi=400)
    #         plt.show()
    #         plt.close()

    #     # the same figure but not in log scale
    #     fig = plt.figure(figsize=(20, 10))
    #     ax = fig.add_subplot(1, 1, 1)
    #     fig.subplots_adjust(hspace=0.5, wspace=0.5)
    #     # ax.set_xscale('log')
    #     # ax.set_yscale('log')
    #     if compare:
    #         xdata4 = np.linspace(np.min(total_freq)*0.75,
    #                              np.max(total_freq)*1.25, 1000)
    #         plt.scatter(freq, dnu, c='C1', s=Size, alpha=0.7)
    #         plt.scatter(old_freq, old_dnu, c='C0', s=Size, alpha=0.7)
    #         plt.errorbar(total_freq, total_dnu, yerr=total_dnuerr, fmt=' ',
    #                      ecolor='k', elinewidth=4, capsize=6, alpha=0.4)
    #         # Dataset 1
    #         plt.plot(xdata4, func1(xdata4, *Slopes_average1), 'C1',
    #                  label=r'UHF $\alpha$=' +
    #                  str(round(Slopes_average1[0], 2))+r'$\pm$' +
    #                  str(round(Slopeerrs_median1, 2)))
    #         plt.fill_between(xdata4.flatten(),
    #                          func1(xdata4,
    #                                *[Slopes_average1[0]+Slopeerrs_median1,
    #                                  Slopes_average1[1]]).flatten(),
    #                          func1(xdata4,
    #                                *[Slopes_average1[0]-Slopeerrs_median1,
    #                                  Slopes_average1[1]]).flatten(),
    #                          alpha=0.2, color='C1')
    #         # Dataset 2
    #         plt.plot(xdata4, func1(xdata4, *Slopes_average2), 'C0',
    #                  label=r'Dataset2 $\alpha$=' +
    #                  str(round(Slopes_average2[0], 2))+r'$\pm$' +
    #                  str(round(Slopeerrs_median2, 2)))
    #         plt.fill_between(xdata4.flatten(),
    #                          func1(xdata4,
    #                                *[Slopes_average2[0]+Slopeerrs_median2,
    #                                  Slopes_average2[1]]).flatten(),
    #                          func1(xdata4,
    #                                *[Slopes_average2[0]-Slopeerrs_median2,
    #                                  Slopes_average2[1]]).flatten(),
    #                          alpha=0.2, color='C0')
    #         # Dataset 3
    #         plt.plot(xdata4, func1(xdata4, *Slopes_average3), 'C3',
    #                  label=r'ALL $\alpha$=' +
    #                  str(round(Slopes_average3[0], 2))+r'$\pm$' +
    #                  str(round(Slopeerrs_median3, 2)))
    #         plt.fill_between(xdata4.flatten(),
    #                          func1(xdata4,
    #                                *[Slopes_average3[0]+Slopeerrs_median3,
    #                                  Slopes_average3[1]]).flatten(),
    #                          func1(xdata4,
    #                                *[Slopes_average3[0]-Slopeerrs_median3,
    #                                  Slopes_average3[1]]).flatten(),
    #                          alpha=0.2, color='C3')
    #         plt.plot(xl, (old_df[0], old_df[0]), color='C2',
    #                  linestyle='dashed', label='Dataset2 Channel bw')
    #         plt.plot(xl, (df[0], df[0]), color='C2',
    #                  label='UHF Channel bw')
    #         plt.xlabel('Frequency (MHz)')
    #         plt.ylabel('Scintillation Bandwidth (MHz)')
    #         # ax.legend(fontsize='xx-small')
    #         ax.legend()
    #         # ax.set_xticks([700, 800, 1000, 1200, 1400, 1600, 1800])
    #         # ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    #         # ax.set_yticks([0.02, 0.1, 0.5, 1, 2, 4])
    #         # ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    #         # ax.axis([1, 100000, 1, 100000])
    #         # ax.loglog()
    #         # ax.xaxis.set_tick_params(length=5, width=2)
    #         # ax.yaxis.set_tick_params(length=5, width=2)
    #         # plt.xticks(np.arange(min(freq), max(freq)+1, 1000))
    #         # plt.yticks(np.arange(min(dnu), max(dnu)+1, 1000))
    #         # plt.xlim(min(xdata4), max(xdata4))
    #         plt.ylim(np.min(total_dnu)*0.90, np.max(total_dnu)*1.1)
    #         plt.xlim(np.min(total_freq)*0.95, np.max(total_freq)*1.05)
    #         plt.show()
    #         plt.close()
    #     else:
    #         xdata4 = np.linspace(np.min(freq)*0.75,
    #                              np.max(freq)*1.25, 1000)
    #         plt.scatter(freq, dnu, c='C1', s=Size, alpha=0.7)
    #         plt.errorbar(freq, dnu, yerr=dnuerr, fmt=' ',
    #                      ecolor='k', elinewidth=4, capsize=6, alpha=0.4)
    #         plt.plot(xdata4, func1(xdata4, *Slopes_average1), 'C4',
    #                  label=r'UHF $\alpha$=' +
    #                  str(round(Slopes_average1[0], 2))+r'$\pm$' +
    #                  str(round(Slopeerrs_median1, 2)))
    #         plt.fill_between(xdata4.flatten(),
    #                          func1(xdata4,
    #                                *[Slopes_average1[0]+Slopeerrs_median1,
    #                                  Slopes_average1[1]]).flatten(),
    #                          func1(xdata4,
    #                                *[Slopes_average1[0]-Slopeerrs_median1,
    #                                  Slopes_average1[1]]).flatten(),
    #                          alpha=0.2, color='C4')
    #         plt.plot(xl, (df[0], df[0]), color='C2',
    #                  label='UHF Channel bw')
    #         plt.xlabel('Frequency (MHz)')
    #         plt.ylabel('Scintillation Bandwidth (MHz)')
    #         # ax.legend(fontsize='xx-small')
    #         ax.legend()
    #         # ax.set_xticks([650, 700, 800, 850, 900, 1000])
    #         # ax.get_xaxis().set_major_formatter(
    #         #     matplotlib.ticker.ScalarFormatter())
    #         # ax.set_yticks([0.01, 0.02, 0.05, 0.1, 0.2, 0.3])
    #         # ax.get_yaxis().set_major_formatter(
    #         #     matplotlib.ticker.ScalarFormatter())
    #         # ax.axis([1, 100000, 1, 100000])
    #         # ax.loglog()
    #         # ax.xaxis.set_tick_params(length=5, width=2)
    #         # ax.yaxis.set_tick_params(length=5, width=2)
    #         # plt.xticks(np.arange(min(freq), max(freq)+1, 1000))
    #         # plt.yticks(np.arange(min(dnu), max(dnu)+1, 1000))
    #         # plt.xlim(min(xdata4), max(xdata4))
    #         plt.ylim(np.min(dnu)*0.90, np.max(dnu)*1.1)
    #         plt.xlim(np.min(freq)*0.95, np.max(freq)*1.05)
    #         plt.savefig('/Users/jacobaskew/Desktop/FreqVdnu.pdf', dpi=400)
    #         plt.savefig('/Users/jacobaskew/Desktop/FreqVdnu.png', dpi=400)
    #         plt.show()
    #         plt.close()

    # if modelling_data:
    #     label = 'test'
    #     wd0 = "/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/New/"
    #     outdir = wd0 + "Modelling/"

    #     # if not Anisotropy_Option:
    #     priors = dict(D=bilby.core.prior.Uniform(0, 2, 'D'),
    #                   s=bilby.core.prior.Uniform(0, 1, 's'),
    #                   vism_ra=bilby.core.prior.Uniform(-50, 50, 'vism_ra'),
    #                   vism_dec=bilby.core.prior.Uniform(-50, 50, 'vism_dec'),
    #                   KIN=bilby.core.prior.Uniform(80, 100, 'KIN',
    #                                                boundary='periodic'),
    #                   KOM=bilby.core.prior.Uniform(60, 70, 'KOM',
    #                                                boundary='periodic'))

    #     # if Anisotropy_Option:
    #     #     priors = dict(s=bilby.core.prior.Uniform(0, 1, 's'),
    #     #                   psi=bilby.core.prior.Uniform(0, 180, 'psi',
    #     #                                                boundary='periodic'),
    #     #                   vism_psi=bilby.core.prior.Uniform(-200, 200,
    #     # 'vism_psi'),
    #     #                   # KIN=bilby.core.prior.Uniform(0, 180, 'KIN',
    #     #                   #                              boundary=
    #     # 'periodic'),
    #     #                   KOM=bilby.core.prior.Uniform(0, 360, 'KOM',
    #     #                                                boundary='periodic'),
    #     #                   # KOM=bilby.prior.Gaussian(mu=KOM_mu,
    #     # sigma=KOM_sigma,
    #     #                   #                          name='KOM'),
    #     #                   # OM=bilby.core.prior.Uniform(0, 360, 'OM',
    #     #                   #                             boundary='periodic'),
    #     #                   # T0=bilby.core.prior.Uniform(minT0, maxT0, 'T0'),
    #     #                   efac=bilby.core.prior.Uniform(-2, 2, 'efac'),
    #     #                   equad=bilby.core.prior.Uniform(-2, 2, 'equad'))
    #     # if not Anisotropy_Option:
    #     likelihood = bilby.likelihood.GaussianLikelihood(
    #         x=mjd, y=viss, func=effective_velocity_annual_bilby, sigma=visserr)
    #     # if Anisotropy_Option:
    #     #     likelihood = \
    #     # bilby.likelihood.GaussianLikelihood(
    #     #     x=mjd, func=effective_velocity_annual_anisotropy_bilby,
    #     # sigma=sigma)

    #     # And run sampler
    #     result = bilby.core.sampler.run_sampler(
    #             likelihood, priors=priors, sampler='dynesty', label=label,
    #             nlive=nlive, verbose=True, resume=resume,
    #             outdir=outdir)

    #     font = {'size': 16}
    #     matplotlib.rc('font', **font)
    #     result.plot_corner()


# def arc_analysis(simulation=False, Observations=None, time_bin=30,
#                  freq_bin=30, overwrite=False):
#     psrname = 'J0737-3039A'

#     font = {'size': 28}
#     matplotlib.rc('font', **font)

#     wd0 = '/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/'
#     wd = wd0+'New/'
#     datadir = wd + 'Dynspec/'
#     outdir = wd + 'DataFiles/'
#     par_dir = '/Users/jacobaskew/Desktop/Swinburne_Daniel/ParFiles/'
#     dynspecs = sorted(glob.glob(datadir + '/*.dynspec'))
#     if Observations:
#         observations = [Observations]
#     else:
#         observations = []
#         for i in range(0, len(dynspecs)):
#             observations.append(dynspecs[i].split(datadir)[1].split('-')[0] +
#                                 '-' +
#                                 dynspecs[i].split(datadir)[1].split('-')[1] +
#                                 '-' +
#                                 dynspecs[i].split(datadir)[1].split('-')[2])
#         observations = np.unique(np.asarray(observations))

#     for i in range(0, len(observations)):
#         observation_date = observations[i]+'/'
#         observation_date2 = observations[i]
#         dynspecs = \
#             sorted(glob.glob(datadir + str(observation_date.split('/')[0])
#                              + '*.dynspec'))
#         dynspecfile2 = \
#             outdir+'DynspecPlotFiles/'+observation_date2 + \
#             'Zap_CompleteDynspec.dynspec'
#         filedir = str(outdir)+str(observation_date)
#         try:
#             os.mkdir(filedir)
#         except OSError as error:
#             print(error)
#         outfile = str(filedir)+str(psrname)+'_'+str(observation_date2) + \
#             '_freq'+str(freq_bin)+'_time'+str(time_bin) + \
#             '_ScintillationResults_UHF.txt'
#         if os.path.exists(outfile) and overwrite:
#             os.remove(outfile)

#         if os.path.exists(dynspecfile2):
#             sim = Simulation()
#             dyn = Dynspec(dyn=sim, process=False)
#             dyn.load_file(filename=dynspecfile2)
#         else:
#             continue
#         # Show the dynspec before process
#         dyn.plot_dyn(dpi=400)
#         # Show the sspec before process
#         dyn.plot_sspec(lamsteps=True, maxfdop=15, dpi=400)
#         # Rescale the dynamic spectrum in time, using a velocity model
#         dyn.scale_dyn(scale='lambda+velocity', s=0.7, d=0.9, inc=88.7,
#                       Omega=65, parfile=par_dir+'J0737-3039A.par')
#         # Plot the velocity-rescaled dynamic spectrum
#         dyn.plot_dyn(velocity=True, dpi=400)
#         # plot new sspec
#         dyn.plot_sspec(lamsteps=True, velocity=True, maxfdop=15, prewhite=True,
#                        dpi=400)
#         # Fit the arc using the new spectrum
#         dyn.fit_arc(velocity=True, cutmid=5, plot=True, weighted=False,
#                     lamsteps=True, subtract_artefacts=True, log_parabola=True,
#                     constraint=[100, 1000], high_power_diff=-0.05,
#                     low_power_diff=-4,
#                     etamin=1, startbin=5)
#         # Plot the sspec with the fit
#         dyn.plot_sspec(lamsteps=True, velocity=True, maxfdop=15, plotarc=True,
#                        prewhite=True, dpi=400)

def merge_data(Observations=None, freq_bin=30, time_bin=10,
               weights_2dacf=False, measure_tilt=False):
    psrname = 'J0737-3039A'

    # Pre-define any plotting arrangements
    font = {'size': 28}
    matplotlib.rc('font', **font)

    wd0 = '/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/'
    wd = wd0+'New/'
    datadir = wd + 'Dynspec/'
    # outdir = wd + 'DataFiles/'
    dynspecs = sorted(glob.glob(datadir + '/*.dynspec'))

    dynspecs = sorted(glob.glob(datadir + '/*.dynspec'))
    if Observations:
        observations = [Observations]
    else:
        observations = []
        for i in range(0, len(dynspecs)):
            observations.append((dynspecs[i].split(datadir)[1].split('-')[1] +
                                '-' +
                                dynspecs[i].split(datadir)[1].split('-')[2] +
                                '-' +
                                dynspecs[i].split(datadir)[1].split('-')[3]).split('3039A_')[1])
        observations = np.unique(np.asarray(observations))
    
    wd0 = '/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/'
    wd = wd0+'New/'
    outdir = wd + 'DataFiles/'
    psrname = 'J0737-3039A'

    for i in range(0, len(observations)):
        observation_date = observations[i] + '/'
        observation_date2 = observations[i]
        wd0 = '/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/'
        wd = wd0+'New/'
        outdir = wd + 'DataFiles/'
        filedir = str(outdir)+str(observation_date)
        outfile = \
            str(filedir)+str(psrname)+'_'+str(observation_date2) + \
            '_freq'+str(freq_bin)+'_time'+str(time_bin) + \
            '_ScintillationResults_UHF.txt'
        outfile_total = str(outdir)+str(psrname)+'_freq' + \
            str(freq_bin)+'_time'+str(time_bin) + \
            '_ScintillationResults_UHF_Total.txt'
        if i == 0:
            Data0 = Table.read(outfile, format='ascii.csv')
            if len(observations) == 1:
                Data2 = Data0
        elif i == 1:
            Data1 = Table.read(outfile, format='ascii.csv')
            Data2 = vstack([Data0, Data1])
        else:
            Data3 = Table.read(outfile, format='ascii.csv')
            Data2 = vstack([Data2, Data3])
    np.savetxt(outfile_total, Data2, delimiter=',', fmt='%s')
    DataTest = np.genfromtxt(outfile_total, delimiter=',', dtype=str)
    if weights_2dacf:
        if measure_tilt:
            TitleRow = ['name', 'mjd', 'freq', 'bw', 'tobs', 'dt', 'df',
                        'tau', 'tauerr', 'dnu', 'dnuerr', 'fse_tau',
                        'fse_dnu', 'scint_param_method', 'dnu_est',
                        'nscint', 'acf_tilt', 'acf_tilt_err', 'fse_tilt',
                        'acf_redchisqr', 'acf_model_redchisqr',
                        'phasegrad', 'phasegraderr', 'fse_phasegrad']
            DataTest = np.vstack((TitleRow, DataTest))
        else:
            TitleRow = ['name', 'mjd', 'freq', 'bw', 'tobs', 'dt', 'df',
                        'tau', 'tauerr', 'dnu', 'dnuerr', 'fse_tau',
                        'fse_dnu', 'scint_param_method', 'dnu_est',
                        'nscint', 'acf_redchisqr', 'acf_model_redchisqr',
                        'phasegrad', 'phasegraderr', 'fse_phasegrad']
    else:
        if measure_tilt:
            TitleRow = ['name', 'mjd', 'freq', 'bw', 'tobs', 'dt', 'df',
                        'tau', 'tauerr', 'dnu', 'dnuerr', 'fse_tau',
                        'fse_dnu', 'scint_param_method', 'dnu_est',
                        'nscint', 'acf_tilt', 'acf_tilt_err', 'fse_tilt',
                        'phasegrad', 'phasegraderr', 'fse_phasegrad']
            DataTest = np.vstack((TitleRow, DataTest))
        else:
            TitleRow = ['name', 'mjd', 'freq', 'bw', 'tobs', 'dt', 'df',
                        'tau', 'tauerr', 'dnu', 'dnuerr', 'fse_tau',
                        'fse_dnu', 'scint_param_method', 'dnu_est',
                        'nscint', 'phasegrad', 'phasegraderr',
                        'fse_phasegrad']
    DataTest = np.vstack((TitleRow, DataTest))
    np.savetxt(outfile_total, DataTest, delimiter=',', fmt='%s')


def process_data(overwrite=False, Plot_dynspec=False, Measure_dynspec=False,
                 Plot_data=True, correct_dynspec=False, zap=False,
                 linear=False, mean=True, median=False, freq_bin=30,
                 time_bin=10, Observations=None, measure_tilt=False,
                 SlowACF=False, nscale=5, wfreq=False, display=False,
                 phasewrapper=False, weights_2dacf=None, cutoff=False,
                 dnuscale_ceil=0.4, tauscale_ceil=600, alpha=5/3,
                 merge_data=False, load_data=True, basic_plot=False,
                 compare=False, compare_filename1=None, compare_filename2=None,
                 spectral_index=False, filtered=True, filtered_dnu=10,
                 filtered_tau=10, filtered_acf_redchisqr=np.inf,
                 modelling_data=False, nlive=200, resume=True):
    if Plot_dynspec:
        plot_dynspec(correct_dynspec=correct_dynspec, zap=zap,
                     linear=linear, mean=mean, median=median,
                     overwrite=overwrite,
                     Observations=Observations)
    if Measure_dynspec:
        measure_dynspec(overwrite=overwrite, measure_tilt=measure_tilt,
                        correct_dynspec=correct_dynspec, zap=zap,
                        linear=linear, median=median, mean=mean,
                        time_bin=time_bin, freq_bin=freq_bin,
                        Observations=Observations, SlowACF=SlowACF,
                        nscale=nscale, wfreq=wfreq, display=display,
                        weights_2dacf=weights_2dacf, cutoff=cutoff,
                        dnuscale_ceil=dnuscale_ceil,
                        tauscale_ceil=tauscale_ceil, alpha=alpha,
                        phasewrapper=phasewrapper)
    if Plot_data:
        plot_data(merge_data=merge_data, load_data=load_data,
                  basic_plot=basic_plot, compare=compare,
                  compare_filename1=compare_filename1, overwrite=overwrite,
                  compare_filename2=compare_filename2, time_bin=time_bin,
                  freq_bin=freq_bin, spectral_index=spectral_index,
                  Observations=Observations, filtered=filtered,
                  filtered_dnu=filtered_dnu, filtered_tau=filtered_tau,
                  modelling_data=modelling_data, nlive=nlive, resume=resume,
                  measure_tilt=measure_tilt, weights_2dacf=weights_2dacf,
                  filtered_acf_redchisqr=filtered_acf_redchisqr)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Temp plotting

# S1dir = '/Users/jacobaskew/Desktop/tmp/'
# S16dir = '/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/New/DataFiles/2023-05-16/'
# freqS1 = np.loadtxt(S1dir+'freq.txt', dtype=float)
# freqS16 = np.loadtxt(S16dir+'freq.txt', dtype=float)
# dnuS1 = np.loadtxt(S1dir+'dnu.txt', dtype=float)
# dnuS16 = np.loadtxt(S16dir+'dnu.txt', dtype=float)
# dnuerrS1 = np.loadtxt(S1dir+'dnuerr.txt', dtype=float)
# dnuerrS16 = np.loadtxt(S16dir+'dnuerr.txt', dtype=float)

# UHF16_df = 0.033203125
# S1_df = 0.85449
# S16_df = 0.05341

# desktopdir = '/Users/jacobaskew/Desktop/'
# datadir = desktopdir + 'DoublePulsar_Project/datasets/FullDataStorage/'
# freqUHF = np.loadtxt(datadir + 'Full_FreqData.txt', dtype='float')
# dnuUHF = np.loadtxt(datadir + 'Full_DnuData.txt', dtype='float')
# dnuerrUHF = np.loadtxt(datadir + 'Full_DnuerrData.txt', dtype='float')

# S1nu_i = np.median(dnuS1[np.argwhere((freqS1 < 2300) * (freqS1 > 2100))])
# S16nu_i = np.median(dnuS16[np.argwhere((freqS16 < 2300) * (freqS16 > 2100))])
# UHF16nu_i = np.median(dnuUHF[np.argwhere((freqUHF > 970) * (freqUHF < 1001))])

Size = 80*np.pi  # Determines the size of the datapoints used
font = {'size': 28}
matplotlib.rc('font', **font)

# fig = plt.figure(figsize=(20, 10))
# ax = fig.add_subplot(1, 1, 1)
# fig.subplots_adjust(hspace=0.5, wspace=0.5)
# #
# plt.scatter(freqUHF, dnuUHF, s=Size, c='C0', marker='o', alpha=0.6, edgecolors='k', label='UHF 16k data')
# plt.errorbar(freqUHF, dnuUHF, yerr=dnuerrUHF, fmt=' ', ecolor='C0', elinewidth=3, capsize=3, alpha=0.4)
# #
# plt.scatter(freqS1, dnuS1, s=Size, c='C1', marker='o', alpha=0.6, edgecolors='k', label='Sband 1k data')
# plt.errorbar(freqS1, dnuS1, yerr=dnuerrS1, fmt=' ', ecolor='C1', elinewidth=3, capsize=3, alpha=0.4)
# # #
# plt.scatter(freqS16, dnuS16, s=Size, c='C2', marker='o', alpha=0.6, edgecolors='k', label='Sband 16k data')
# plt.errorbar(freqS16, dnuS16, yerr=dnuerrS16, fmt=' ', ecolor='C2', elinewidth=3, capsize=3, alpha=0.4)
# #
# xl = plt.xlim()
# xrange = np.linspace(xl[0], xl[1], 1000)
# # yrange_K = S1nu_i * (xrange / 2200) ** 4.4
# yrange_K = UHF16nu_i * (xrange / 1000) ** 4.4
# # yrange_UHF = UHF16nu_i * (xrange / 1000) ** 3.536429035411339
# yrange_S = S1nu_i * (xrange / 2200) ** 3.2151391444677264
# #
# plt.plot(xrange, yrange_K, 'k--', label=r'Kolmogorov $\alpha$=4.4')
# # plt.plot(xrange, yrange_UHF, 'C0--', label=r'UHF 16k Fit $\alpha$=3.536')
# # plt.plot(xrange, yrange_S, 'C1--', label=r'Sband 1k Fit $\alpha$=3.215')
# plt.hlines(UHF16_df, xl[0], xl[1], colors='C2', linestyles='dashed', alpha=0.3, label='UHF 16k Channel Bandwidth')
# # plt.hlines(S16_df, xl[0], xl[1], colors='C2', linestyles='dashed', alpha=0.3, label='Sband 16k Channel Bandwidth')
# # plt.hlines(S1_df, xl[0], xl[1], colors='C2', linestyles='dashed', alpha=0.3, label='Sband 1k Channel Bandwidth')
# #
# plt.xlim(xl)
# plt.legend(fontsize='x-small')
# plt.xlabel(r"Frequency, $\nu$ (MHz)")
# plt.ylabel(r"Scintillation Bandwidth, $\Delta\nu_d$ (MHz)")
# #
# plt.savefig("/Users/jacobaskew/Desktop/FreqVDnu.png")
# ax.set_xscale('log')
# ax.set_yscale('log')
# plt.savefig("/Users/jacobaskew/Desktop/FreqVDnu_Log.png")
# #
# plt.show()
# plt.close()

# fig = plt.figure(figsize=(20, 10))
# ax = fig.add_subplot(1, 1, 1)
# fig.subplots_adjust(hspace=0.5, wspace=0.5)
# cm = plt.cm.get_cmap('viridis')
# z = mjd
# plt.scatter(phase, phasegrad, s=Size, c=z, cmap=cm, marker='o', alpha=0.6, edgecolors='k')
# plt.errorbar(phase, phasegrad, yerr=phasegraderr, fmt=' ', ecolor='k', elinewidth=3, capsize=3, alpha=0.4)
# plt.ylabel('Phase Gradient Magnitude (min/MHz)')
# plt.xlabel('Orbital Phase (degrees)')
# plt.savefig("/Users/jacobaskew/Desktop/PhaseVGradient.png")
# plt.show()
# plt.close()

# test = phasegrad * (dnu / tau) * 60

# fig = plt.figure(figsize=(20, 10))
# ax = fig.add_subplot(1, 1, 1)
# fig.subplots_adjust(hspace=0.5, wspace=0.5)
# cm = plt.cm.get_cmap('viridis')
# z = mjd
# plt.scatter(phase, test, s=Size, c=z, cmap=cm, marker='o', alpha=0.6, edgecolors='k')
# # plt.errorbar(phase, test, yerr=phasegraderr, fmt=' ', ecolor='k', elinewidth=3, capsize=3, alpha=0.4)
# plt.ylabel(r'PhaseGradient * ($\Delta\nu_d$ / $\tau_d$)')
# plt.xlabel('Orbital Phase (degrees)')
# plt.savefig("/Users/jacobaskew/Desktop/PhaseVGradient.png")
# plt.show()
# plt.close()
