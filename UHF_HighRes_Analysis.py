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

##############################################################################
# Importing neccessary things
from scintools.dynspec import Dynspec
from scintools.scint_utils import write_results, read_results, read_par, \
        float_array_from_dict, get_ssb_delay, get_earth_velocity, \
        get_true_anomaly, scint_velocity, pars_to_params
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import glob
from copy import deepcopy as cp
import os
from scipy.optimize import curve_fit
from scintools.scint_sim import Simulation
from astropy.table import Table, vstack

###############################################################################
# These functions are used by scipy curve fit


def func1(xdata, c, a):
    return a*(xdata/850)**c


def func2(xdata, c, a):
    return a*(xdata/1300)**c


def func3(xdata, c, a):
    return a*(xdata/1600)**c


##############################################################################
# This function aims to try one method to remove the effect of the eclipse
# this is based on the assumption that there is two eclispes in the groups
# of three observations and they are at the lowest flux in the observation

def remove_eclipse(start_mjd, tobs, dyn, fluxes):
    end_mjd = start_mjd + tobs/86400
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
            dyn.refill()
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
            dyn.refill()
        else:
            print("No Eclispe in dynspec")
    return Eclipse_index


##############################################################################
psrname = 'J0737-3039A'
pulsar = '0737-3039A'

# Pre-define any plotting arrangements
Font = 30
Size = 80*np.pi  # Determines the size of the datapoints used
font = {'size': 28}
matplotlib.rc('font', **font)

HighRes = True
Lband = False

wd0 = '/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/'
eclipsefile = wd0+'Datafiles/Eclipse_mjd.txt'
if HighRes:
    wd = wd0+'New/'
else:
    wd = wd0+'New_LowRes/'
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
if not Lband:
    observations = np.delete(observations, 3)
# Settings #
time_bin = 10
freq_bin = 30
#
plot_dynspec = False
measure = False
merge_data = False
load_data = False
load_all = False
compare = False
arc_analysis = False
overwrite = False
#
measure_tilt = False
correct_dynspec = False
zap = True
linear = False
var = False
median = True
#
if measure:
    acf_residuals = []

for i in range(0, len(observations)):
    observation_date = observations[i]+'/'
    observation_date2 = observations[i]
    spectradir = wd + 'SpectraPlots/' + observation_date
    spectradir2 = wd + 'SpectraPlots/' + observation_date2
    spectrabindir = wd + 'SpectraPlotsBin/' + observation_date
    ACFdir = wd + 'ACFPlots/' + observation_date
    ACFbindir = wd + 'ACFPlotsBin/' + observation_date
    plotdir = wd + 'Plots/' + observation_date
    dynspecs = sorted(glob.glob(datadir + str(observation_date.split('/')[0])
                                + '*.XPp.dynspec'))
    dynspecfile = \
        outdir+'DynspecPlotFiles/'+observation_date2+'_CompleteDynspec.dynspec'
    dynspecfile2 = \
        outdir+'DynspecPlotFiles/'+observation_date2 + \
        'Zap_CompleteDynspec.dynspec'
    filedir = str(outdir)+str(observation_date)
    filedir2 = str(outdir)
    try:
        os.mkdir(filedir)
    except OSError as error:
        print(error)
    outfile = str(filedir)+str(psrname)+'_'+str(observation_date2)+'_freq' + \
        str(freq_bin)+'_time'+str(time_bin)+'_ScintillationResults_UHF.txt'
    outfile_total = str(filedir2)+str(psrname)+'_freq'+str(freq_bin) + \
        '_time'+str(time_bin)+'_ScintillationResults_UHF_Total.txt'
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

    if plot_dynspec:

        if overwrite:
            print("overwriting old dynspec for", observation_date)
        elif os.path.exists(dynspecfile2):
            continue
        else:
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
        dyn.trim_edges()
        Fmax = np.max(dyn.freqs) - 48
        Fmin = np.min(dyn.freqs) + 48
        dyn_crop0 = cp(dyn)
        dyn_crop0.crop_dyn(fmin=Fmin, fmax=Fmax)
        f_min_init = Fmax - freq_bin
        f_max = Fmax
        f_init = int((f_max + f_min_init)/2)
        bw_init = int(f_max - f_min_init)

        # if observation_date2 == '2022-09-18':
        #     if HighRes:
        #         specific_Fmax = len(dyn_crop0.freqs)/2 * dyn_crop0.df + \
        #             np.min(dyn_crop0.freqs)
        #         dyn_crop0.crop_dyn(fmax=specific_Fmax)
        #     dyn_flag1 = cp(dyn_crop0)
        #     # dyn_flag1.crop_dyn(fmin=1630)
        #     dyn_median = np.median(dyn_flag1.dyn)
        #     freq_min_index1 = int((1520 - np.min(dyn_crop0.freqs)) /
        #                           (dyn_crop0.df))
        #     freq_max_index1 = int((1630 - np.min(dyn_crop0.freqs)) /
        #                           (dyn_crop0.df))
        #     dyn_flag1.dyn[freq_min_index1:freq_max_index1, :] = 0
        #     freq_min_index2 = int((1125 - np.min(dyn_crop0.freqs)) /
        #                           (dyn_crop0.df))
        #     freq_max_index2 = int((1300 - np.min(dyn_crop0.freqs)) /
        #                           (dyn_crop0.df))
        #     dyn_flag1.dyn[freq_min_index2:freq_max_index2, :] = 0
        #     freq_min_index3 = int((1080 - np.min(dyn_crop0.freqs)) /
        #                           (dyn_crop0.df))
        #     freq_max_index3 = int((1100 - np.min(dyn_crop0.freqs)) /
        #                           (dyn_crop0.df))
        #     dyn_flag1.dyn[freq_min_index3:freq_max_index3, :] = 0
        #     freq_min_index4 = int((930 - np.min(dyn_crop0.freqs)) /
        #                           (dyn_crop0.df))
        #     freq_max_index4 = int((965 - np.min(dyn_crop0.freqs)) /
        #                           (dyn_crop0.df))
        #     dyn_flag1.dyn[freq_min_index4:freq_max_index4, :] = 0
        #     dyn_flag1.plot_dyn()
        #     dyn_flag1.refill(method='median')
        #     dyn_flag1.plot_dyn()

        # Save the spectra before zapping
        dyn_crop0.plot_dyn(filename=str(spectradir2) +
                           '_Eclipse_FullSpectra.pdf',
                           dpi=400)
        dyn_crop0.write_file(filename=str(dynspecfile))
        dyn_crop = cp(dyn_crop0)
        #
        len_min = dyn_crop.tobs/60
        len_min_chunk = len_min/20

        len_minimum = 0
        len_maximum = int(len_min_chunk)
        try:
            dyn_crop1 = cp(dyn_crop)
            dyn_crop1.crop_dyn(tmin=0, tmax=len_maximum)
            dyn_crop1.zap()
            dyn_crop1.refill()
        except ValueError as e:
            print(e)
        try:
            len_minimum += int(len_min_chunk)
            len_maximum += int(len_min_chunk)
            dyn_crop2 = cp(dyn_crop)
            dyn_crop2.crop_dyn(tmin=len_minimum, tmax=len_maximum)
            dyn_crop2.zap()
            dyn_crop2.refill()
        except ValueError as e:
            print(e)
        try:
            len_minimum += int(len_min_chunk)
            len_maximum += int(len_min_chunk)
            dyn_crop3 = cp(dyn_crop)
            dyn_crop3.crop_dyn(tmin=len_minimum, tmax=len_maximum)
            dyn_crop3.zap()
            dyn_crop3.refill()
        except ValueError as e:
            print(e)
        try:
            len_minimum += int(len_min_chunk)
            len_maximum += int(len_min_chunk)
            dyn_crop4 = cp(dyn_crop)
            dyn_crop4.crop_dyn(tmin=len_minimum, tmax=len_maximum)
            dyn_crop4.zap()
            dyn_crop4.refill()
        except ValueError as e:
            print(e)
        try:
            len_minimum += int(len_min_chunk)
            len_maximum += int(len_min_chunk)
            dyn_crop5 = cp(dyn_crop)
            dyn_crop5.crop_dyn(tmin=len_minimum, tmax=len_maximum)
            dyn_crop5.zap()
            dyn_crop5.refill()
        except ValueError as e:
            print(e)
        try:
            len_minimum += int(len_min_chunk)
            len_maximum += int(len_min_chunk)
            dyn_crop6 = cp(dyn_crop)
            dyn_crop6.crop_dyn(tmin=len_minimum, tmax=len_maximum)
            dyn_crop6.zap()
            dyn_crop6.refill()
        except ValueError as e:
            print(e)
        try:
            len_minimum += int(len_min_chunk)
            len_maximum += int(len_min_chunk)
            dyn_crop7 = cp(dyn_crop)
            dyn_crop7.crop_dyn(tmin=len_minimum, tmax=len_maximum)
            dyn_crop7.zap()
            dyn_crop7.refill()
        except ValueError as e:
            print(e)
        try:
            len_minimum += int(len_min_chunk)
            len_maximum += int(len_min_chunk)
            dyn_crop8 = cp(dyn_crop)
            dyn_crop8.crop_dyn(tmin=len_minimum, tmax=len_maximum)
            dyn_crop8.zap()
            dyn_crop8.refill()
        except ValueError as e:
            print(e)
        try:
            len_minimum += int(len_min_chunk)
            len_maximum += int(len_min_chunk)
            dyn_crop9 = cp(dyn_crop)
            dyn_crop9.crop_dyn(tmin=len_minimum, tmax=len_maximum)
            dyn_crop9.zap()
            dyn_crop9.refill()
        except ValueError as e:
            print(e)
        try:
            len_minimum += int(len_min_chunk)
            len_maximum += int(len_min_chunk)
            dyn_crop10 = cp(dyn_crop)
            dyn_crop10.crop_dyn(tmin=len_minimum, tmax=len_maximum)
            dyn_crop10.zap()
            dyn_crop10.refill()
        except ValueError as e:
            print(e)
        try:
            len_minimum += int(len_min_chunk)
            len_maximum += int(len_min_chunk)
            dyn_crop11 = cp(dyn_crop)
            dyn_crop11.crop_dyn(tmin=len_minimum, tmax=len_maximum)
            dyn_crop11.zap()
            dyn_crop11.refill()
        except ValueError as e:
            print(e)
        try:
            len_minimum += int(len_min_chunk)
            len_maximum += int(len_min_chunk)
            dyn_crop12 = cp(dyn_crop)
            dyn_crop12.crop_dyn(tmin=len_minimum, tmax=len_maximum)
            dyn_crop12.zap()
            dyn_crop12.refill()
        except ValueError as e:
            print(e)
        try:
            len_minimum += int(len_min_chunk)
            len_maximum += int(len_min_chunk)
            dyn_crop13 = cp(dyn_crop)
            dyn_crop13.crop_dyn(tmin=len_minimum, tmax=len_maximum)
            dyn_crop13.zap()
            dyn_crop13.refill()
        except ValueError as e:
            print(e)
        try:
            len_minimum += int(len_min_chunk)
            len_maximum += int(len_min_chunk)
            dyn_crop14 = cp(dyn_crop)
            dyn_crop14.crop_dyn(tmin=len_minimum, tmax=len_maximum)
            dyn_crop14.zap()
            dyn_crop14.refill()
        except ValueError as e:
            print(e)
        try:
            len_minimum += int(len_min_chunk)
            len_maximum += int(len_min_chunk)
            dyn_crop15 = cp(dyn_crop)
            dyn_crop15.crop_dyn(tmin=len_minimum, tmax=len_maximum)
            dyn_crop15.zap()
            dyn_crop15.refill()
        except ValueError as e:
            print(e)
        try:
            len_minimum += int(len_min_chunk)
            len_maximum += int(len_min_chunk)
            dyn_crop16 = cp(dyn_crop)
            dyn_crop16.crop_dyn(tmin=len_minimum, tmax=len_maximum)
            dyn_crop16.zap()
            dyn_crop16.refill()
        except ValueError as e:
            print(e)
        try:
            len_minimum += int(len_min_chunk)
            len_maximum += int(len_min_chunk)
            dyn_crop17 = cp(dyn_crop)
            dyn_crop17.crop_dyn(tmin=len_minimum, tmax=len_maximum)
            dyn_crop17.zap()
            dyn_crop17.refill()
        except ValueError as e:
            print(e)
        try:

            len_minimum += int(len_min_chunk)
            len_maximum += int(len_min_chunk)
            dyn_crop18 = cp(dyn_crop)
            dyn_crop18.crop_dyn(tmin=len_minimum, tmax=len_maximum)
            dyn_crop18.zap()
            dyn_crop18.refill()
        except ValueError as e:
            print(e)
        try:

            len_minimum += int(len_min_chunk)
            len_maximum += int(len_min_chunk)
            dyn_crop19 = cp(dyn_crop)
            dyn_crop19.crop_dyn(tmin=len_minimum, tmax=len_maximum)
            dyn_crop19.zap()
            dyn_crop19.refill()
        except ValueError as e:
            print(e)
        try:

            len_minimum += int(len_min_chunk)
            len_maximum += int(len_min_chunk)
            dyn_crop20 = cp(dyn_crop)
            dyn_crop20.crop_dyn(tmin=len_minimum, tmax=np.inf)
            dyn_crop20.zap()
            dyn_crop20.refill()
        except ValueError as e:
            print(e)
        try:
            dyn_all = dyn_crop1 + dyn_crop2 + dyn_crop3 + dyn_crop4 + \
                dyn_crop5 + dyn_crop6 + dyn_crop7 + dyn_crop8 + dyn_crop9 + \
                dyn_crop10 + dyn_crop11 + dyn_crop12 + dyn_crop13 + \
                dyn_crop14 + dyn_crop15 + dyn_crop16 + dyn_crop17 + \
                dyn_crop18 + dyn_crop19 + dyn_crop20
        except Exception as e:
            print(e)
        if observation_date2 == '2022-06-30':
            dyn_crop1 = cp(dyn_all)
            dyn_crop2 = cp(dyn_all)
            dyn_crop1.crop_dyn(tmin=0, tmax=20)
            dyn_crop2.crop_dyn(tmin=40, tmax=np.inf)
            dyn_all = dyn_crop1 + dyn_crop2
        elif observation_date2 == '2022-08-27':
            dyn_crop1 = cp(dyn_all)
            dyn_crop2 = cp(dyn_all)
            dyn_crop1.crop_dyn(tmin=0, tmax=140)
            dyn_crop2.crop_dyn(tmin=160, tmax=np.inf)
            dyn_all = dyn_crop1 + dyn_crop2
        else:
            dyn_all = cp(dyn_all)

        dyn_all.plot_dyn(filename=str(spectradir2) +
                         '_Zap_Eclipse_FullSpectra.pdf',
                         dpi=400)
        dyn_all.write_file(filename=str(dynspecfile2))

    if arc_analysis:
        if os.path.exists(dynspecfile2):
            sim = Simulation()
            dyn = Dynspec(dyn=sim, process=False)
            dyn.load_file(filename=dynspecfile2)
        else:
            continue
        # Show the dynspec before process
        dyn.plot_dyn(dpi=400)
        # Show the sspec before process
        dyn.plot_sspec(lamsteps=True, maxfdop=15, dpi=400)
        # Rescale the dynamic spectrum in time, using a velocity model
        dyn.scale_dyn(scale='lambda+velocity', s=0.7, d=0.9, inc=88.7,
                      Omega=65, parfile=par_dir+'J0737-3039A.par')
        # Plot the velocity-rescaled dynamic spectrum
        dyn.plot_dyn(velocity=True, dpi=400)
        # plot new sspec
        dyn.plot_sspec(lamsteps=True, velocity=True, maxfdop=15, prewhite=True,
                       dpi=400)
        # Fit the arc using the new spectrum
        dyn.fit_arc(velocity=True, cutmid=5, plot=True, weighted=False,
                    lamsteps=True, subtract_artefacts=True, log_parabola=True,
                    constraint=[100, 1000], high_power_diff=-0.05,
                    low_power_diff=-4,
                    etamin=1, startbin=5)
        # Plot the sspec with the fit
        dyn.plot_sspec(lamsteps=True, velocity=True, maxfdop=15, plotarc=True,
                       prewhite=True, dpi=400)

    if measure:
        if os.path.exists(dynspecfile2):
            sim = Simulation()
            dyn = Dynspec(dyn=sim, process=False)
            dyn.load_file(filename=dynspecfile2)
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
                bad_time_low = 149.5
                bad_time_high = 150.5
                bad_index_time_low = int((bad_time_low*60)/(dyn.dt))
                bad_index_time_high = int((bad_time_high*60)/(dyn.dt))
                dyn.dyn[:, bad_index_time_low:bad_index_time_high] = 0
                bad_freq_low = 946
                bad_freq_high = 949
                bad_index_low = int((bad_freq_low - np.min(dyn.freqs)) /
                                    (dyn.df))
                bad_index_high = int((bad_freq_high - np.min(dyn.freqs)) /
                                     (dyn.df))
                dyn.dyn[bad_index_low:bad_index_high, :] = 0
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
                bad_time_low = 30
                bad_time_high = 30.5
                bad_index_time_low = int((bad_time_low*60)/(dyn.dt))
                bad_index_time_high = int((bad_time_high*60)/(dyn.dt))
                dyn.dyn[:, bad_index_time_low:bad_index_time_high] = 0
                bad_freq_low = 935
                bad_freq_high = 949
                bad_index_low = int((bad_freq_low - np.min(dyn.freqs)) /
                                    (dyn.df))
                bad_index_high = int((bad_freq_high - np.min(dyn.freqs)) /
                                     (dyn.df))
                dyn.dyn[bad_index_low:bad_index_high, :] = 0
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
                dyn.dyn[bad_index_low:bad_index_high, :] = 0
                bad_time_low = 150
                bad_time_high = 151
                bad_index_time_low = int((bad_time_low*60)/(dyn.dt))
                bad_index_time_high = int((bad_time_high*60)/(dyn.dt))
                dyn.dyn[:, bad_index_time_low:bad_index_time_high] = 0
                bad_time_low = 29.5
                bad_time_high = 30.75
                bad_index_time_low = int((bad_time_low*60)/(dyn.dt))
                bad_index_time_high = int((bad_time_high*60)/(dyn.dt))
                dyn.dyn[:, bad_index_time_low:bad_index_time_high] = 0
                bad_time_low = 9.75
                bad_time_high = 10.75
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
                bad_time_low = 150.25
                bad_time_high = 150.75
                bad_index_time_low = int((bad_time_low*60)/(dyn.dt))
                bad_index_time_high = int((bad_time_high*60)/(dyn.dt))
                dyn.dyn[:, bad_index_time_low:bad_index_time_high] = 0
                bad_time_low = 128
                bad_time_high = 132
                bad_index_time_low = int((bad_time_low*60)/(dyn.dt))
                bad_index_time_high = int((bad_time_high*60)/(dyn.dt))
                dyn.dyn[:, bad_index_time_low:bad_index_time_high] = 0
                bad_time_low = 29.75
                bad_time_high = 30.75
                bad_index_time_low = int((bad_time_low*60)/(dyn.dt))
                bad_index_time_high = int((bad_time_high*60)/(dyn.dt))
                dyn.dyn[:, bad_index_time_low:bad_index_time_high] = 0
                bad_time_low = 8.75
                bad_time_high = 9.75
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
                bad_time_low = 150.25
                bad_time_high = 150.75
                bad_index_time_low = int((bad_time_low*60)/(dyn.dt))
                bad_index_time_high = int((bad_time_high*60)/(dyn.dt))
                dyn.dyn[:, bad_index_time_low:bad_index_time_high] = 0
                bad_time_low = 29.75
                bad_time_high = 31
                bad_index_time_low = int((bad_time_low*60)/(dyn.dt))
                bad_index_time_high = int((bad_time_high*60)/(dyn.dt))
                dyn.dyn[:, bad_index_time_low:bad_index_time_high] = 0
            # elif observation_date2 == '2022-12-30':
            #     dyn_diff = cp(dyn)
            #     bad_freq_low = 935
            #     bad_freq_high = 948
            #     bad_index_low = int((bad_freq_low - np.min(dyn_diff.freqs)) /
            #                         (dyn_diff.df))
            #     bad_index_high = int((bad_freq_high - np.min(dyn_diff.freqs)) /
            #                          (dyn_diff.df))
            #     dyn_diff.dyn[bad_index_low:bad_index_high, :] = 0
            #     bad_time_low = 150.25
            #     bad_time_high = 150.75
            #     bad_index_time_low = int((bad_time_low*60)/(dyn_diff.dt))
            #     bad_index_time_high = int((bad_time_high*60)/(dyn_diff.dt))
            #     dyn_diff.dyn[:, bad_index_time_low:bad_index_time_high] = 0
            #     # bad_time_low = 128
            #     # bad_time_high = 132
            #     # bad_index_time_low = int((bad_time_low*60)/(dyn_diff.dt))
            #     # bad_index_time_high = int((bad_time_high*60)/(dyn_diff.dt))
            #     # dyn_diff.dyn[:, bad_index_time_low:bad_index_time_high] = 0
            #     bad_time_low = 29.75
            #     bad_time_high = 31
            #     bad_index_time_low = int((bad_time_low*60)/(dyn_diff.dt))
            #     bad_index_time_high = int((bad_time_high*60)/(dyn_diff.dt))
            #     dyn_diff.dyn[:, bad_index_time_low:bad_index_time_high] = 0
            #     # bad_time_low = 8.75
            #     # bad_time_high = 9.75
            #     # bad_index_time_low = int((bad_time_low*60)/(dyn_diff.dt))
            #     # bad_index_time_high = int((bad_time_high*60)/(dyn_diff.dt))
            #     # dyn_diff.dyn[:, bad_index_time_low:bad_index_time_high] = 0
            #     dyn_diff.plot_dyn(filename='/Users/jacobaskew/Desktop/test.pdf', dpi=400)
            else:
                dyn = cp(dyn)  # This does nothing but is something else loop
        else:
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
            dyn.write_file(filename=str(dynspecfile))
        dyn.trim_edges()
        Fmax = np.max(dyn.freqs) - 48
        Fmin = np.min(dyn.freqs) + 48
        dyn_crop0 = cp(dyn)
        dyn_crop0.crop_dyn(fmin=Fmin, fmax=Fmax)
        f_min_init = Fmax - freq_bin
        f_max = Fmax
        f_init = int((f_max + f_min_init)/2)
        bw_init = int(f_max - f_min_init)

        dyn_crop = cp(dyn_crop0)

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
            if observation_date2 == '2022-06-30':
                if t_max_new >= 31 and t_max_new <= 49 and istart_t >= 31 \
                        and istart_t <= 49:
                    counter_nodynspec += 1
                    continue
                elif t_max_new >= 21 and t_max_new <= 39:
                    dyn_new_time.crop_dyn(tmin=20-time_bin, tmax=20)
                elif istart_t >= 21 and istart_t <= 39:
                    dyn_new_time.crop_dyn(tmin=40, tmax=40+time_bin)
                else:
                    dyn_new_time.crop_dyn(tmin=istart_t, tmax=t_max_new)
            elif observation_date2 == '2022-08-27':
                if t_max_new >= 141 and t_max_new <= 159 \
                        and istart_t >= 141 and istart_t <= 159:
                    counter_nodynspec += 1
                    continue
                elif t_max_new >= 141 and t_max_new <= 159:
                    dyn_new_time.crop_dyn(tmin=istart_t, tmax=140)
                elif istart_t >= 141 and istart_t <= 159:
                    dyn_new_time.crop_dyn(tmin=160, tmax=t_max_new)
                else:
                    dyn_new_time.crop_dyn(tmin=istart_t, tmax=t_max_new)
            else:
                dyn_new_time.crop_dyn(tmin=istart_t, tmax=t_max_new)
            if var:
                freq_bins = 1
            else:
                freq_bins = freq_bin
            for istart_f in range(int(np.max(dyn_crop.freqs)), 0,
                                  -int(freq_bins)):
                try:
                    if var:
                        if istart_f == int(np.max(dyn_crop.freqs)):
                            f_min_new = istart_f - freq_bin
                            bw_new = 0
                        else:
                            f_min_new = istart_f - bw_new
                        bw_new = (f_min_new / f_init)**2 * bw_init
                        f_max_new = f_min_new + bw_new
                    else:
                        f_min_new = istart_f - freq_bin
                        f_max_new = istart_f
                    if f_min_new < Fmin:
                        continue
                    dyn_new_freq = cp(dyn_new_time)
                    # if observation_date2 == '2022-09-18':
                    #     # RFI region in L-Band on this day was 1520-1630
                    #     if f_max_new >= 1521 and f_max_new <= 1629 \
                    #             and f_min_new >= 1521 and f_min_new <= 1629:
                    #         counter_nodynspec += 1
                    #         continue
                    #     elif f_max_new >= 1521 and f_max_new <= 1629:
                    #         dyn_new_freq.crop_dyn(fmin=f_min_new, fmax=1520)
                    #     elif f_min_new >= 1521 and f_min_new <= 1629:
                    #         dyn_new_freq.crop_dyn(fmin=1630, fmax=f_max_new)
                    #     else:
                    #         dyn_new_freq.crop_dyn(fmin=f_min_new,
                    #                               fmax=f_max_new)
                    #     # RFI region in L-Band on this day was 1125-1300
                    #     if f_max_new >= 1126 and f_max_new <= 1299 \
                    #             and f_min_new >= 1126 and f_min_new <= 1299:
                    #         counter_nodynspec += 1
                    #         continue
                    #     elif f_max_new >= 1126 and f_max_new <= 1299:
                    #         dyn_new_freq.crop_dyn(fmin=f_min_new, fmax=1300)
                    #     elif f_min_new >= 1126 and f_min_new <= 1299:
                    #         dyn_new_freq.crop_dyn(fmin=1125, fmax=f_max_new)
                    #     else:
                    #         dyn_new_freq.crop_dyn(fmin=f_min_new,
                    #                               fmax=f_max_new)
                    #     # RFI region in L-Band on this day was 1080-1100
                    #     if f_max_new >= 1081 and f_max_new <= 1099 \
                    #             and f_min_new >= 1081 and f_min_new <= 1099:
                    #         counter_nodynspec += 1
                    #         continue
                    #     elif f_max_new >= 1081 and f_max_new <= 1099:
                    #         dyn_new_freq.crop_dyn(fmin=f_min_new, fmax=1100)
                    #     elif f_min_new >= 1081 and f_min_new <= 1099:
                    #         dyn_new_freq.crop_dyn(fmin=1080, fmax=f_max_new)
                    #     else:
                    #         dyn_new_freq.crop_dyn(fmin=f_min_new,
                    #                               fmax=f_max_new)
                    #     # RFI region in L-Band on this day was 930-965
                    #     if f_max_new >= 931 and f_max_new <= 964 \
                    #             and f_min_new >= 931 and f_min_new <= 964:
                    #         counter_nodynspec += 1
                    #         continue
                    #     elif f_max_new >= 931 and f_max_new <= 964:
                    #         dyn_new_freq.crop_dyn(fmin=f_min_new, fmax=930)
                    #     elif f_min_new >= 931 and f_min_new <= 964:
                    #         dyn_new_freq.crop_dyn(fmin=965, fmax=f_max_new)
                    #     else:
                    #         dyn_new_freq.crop_dyn(fmin=f_min_new,
                    #                               fmax=f_max_new)
                    # For this day 935-960 is dominated by RFI
                    # if observation_date2 == '2022-06-30':  # elif
                    #     if f_max_new >= 931 and f_max_new <= 964 \
                    #             and f_min_new >= 931 and f_min_new <= 964:
                    #         counter_nodynspec += 1
                    #         continue
                    #     elif f_max_new >= 931 and f_max_new <= 964:
                    #         dyn_new_freq.crop_dyn(fmin=f_min_new, fmax=930)
                    #     elif f_min_new >= 931 and f_min_new <= 964:
                    #         dyn_new_freq.crop_dyn(fmin=965, fmax=f_max_new)
                    #     else:
                    #         dyn_new_freq.crop_dyn(fmin=f_min_new,
                    #                               fmax=f_max_new)
                    # elif observation_date2 == '2022-08-27':  # elif
                    #     if f_max_new >= 931 and f_max_new <= 964 \
                    #             and f_min_new >= 931 and f_min_new <= 964:
                    #         counter_nodynspec += 1
                    #         continue
                    #     elif f_max_new >= 931 and f_max_new <= 964:
                    #         dyn_new_freq.crop_dyn(fmin=f_min_new, fmax=930)
                    #     elif f_min_new >= 931 and f_min_new <= 964:
                    #         dyn_new_freq.crop_dyn(fmin=965, fmax=f_max_new)
                    #     else:
                    #         dyn_new_freq.crop_dyn(fmin=f_min_new,
                    #                               fmax=f_max_new)
                    # else:
                    dyn_new_freq.crop_dyn(fmin=f_min_new, fmax=f_max_new)
                    if zap:
                        dyn_new_freq.zap()
                    if linear:
                        dyn_new_freq.refill(method='linear')
                    elif median:
                        dyn_new_freq.refill(method='median')
                    else:
                        dyn_new_freq.refill()
                    if correct_dynspec:
                        dyn_new_freq.correct_dyn()
                    if dyn_new_freq.tobs < 600 or dyn_new_freq.bw < 10:
                        fail_counter += 1
                        try:
                            dyn_new_freq.plot_dyn(filename=str(spectrabindir) +
                                                  str(round(istart_t, 1)) +
                                                  '/dynspec_chunk_' +
                                                  str(round(dyn_new_freq.freq, 2))
                                                  + '.pdf', dpi=400)
                            if measure_tilt:
                                dyn_new_freq.get_acf_tilt(filename=str(ACFbindir) +
                                                          str(round(istart_t, 1)) +
                                                          '/ACFfit_chunk_' +
                                                          str(int(dyn_new_freq.freq)) +
                                                          '.pdf', plot=True,
                                                          display=True)
                            dyn_new_freq.get_scint_params(filename=str(ACFbindir) +
                                                          str(round(istart_t, 1)) +
                                                          '/ACF_chunk_' +
                                                          str(int(dyn_new_freq.freq))+'.pdf',
                                                          method='acf2d_approx',
                                                          plot=True, display=True)
                            dyn_new_freq.get_scint_params(filename=str(ACFbindir) +
                                                          str(round(istart_t, 1)) +
                                                          '/ACF_chunk_' +
                                                          str(int(dyn_new_freq.freq))+'.pdf',
                                                          method='acf1d',
                                                          plot=True, display=True)
                        except Exception as e:
                            print(e)
                            counter_nodynspec += 1
                        continue
                    else:
                        if measure_tilt:
                            dyn_new_freq.get_acf_tilt(filename=str(ACFdir) +
                                                      str(round(istart_t, 1)) +
                                                      '/ACFfit_chunk_' +
                                                      str(int(dyn_new_freq.freq)) +
                                                      '.pdf', plot=True,
                                                      display=True)
                        dyn_new_freq.get_scint_params(filename=str(ACFdir) +
                                                      str(round(istart_t, 1)) +
                                                      '/ACF_chunk_' +
                                                      str(int(dyn_new_freq.freq)) +
                                                      '.pdf',
                                                      method='acf2d_approx',
                                                      plot=True, display=True)
                        dyn_new_freq.get_scint_params(filename=str(ACFbindir) +
                                                      str(round(istart_t, 1)) +
                                                      '/ACF_chunk_' +
                                                      str(int(dyn_new_freq.freq))+'.pdf',
                                                      method='acf1d',
                                                      plot=True, display=True)
                        # if dyn.scint_param_method != 'acf2d_approx':
                        #     test = 1/0
                        write_results(outfile, dyn=dyn_new_freq)
                        acf_residuals.append(np.std(dyn_new_freq.acf_residuals))
                        good_counter += 1
                        dyn_new_freq.plot_dyn(filename=str(spectradir) +
                                              str(round(istart_t, 1)) +
                                              '/dynspec_chunk_' +
                                              str(round(dyn_new_freq.freq,
                                                        2))+'.pdf', dpi=400)
                except Exception as e:
                    print(e)
                    fail_counter += 1
                    try:
                        dyn_new_freq.plot_dyn(filename=str(spectrabindir) +
                                              str(round(istart_t, 1)) +
                                              '/dynspec_chunk_' +
                                              str(round(dyn_new_freq.freq, 2))
                                              + '.pdf', dpi=400)
                        if measure_tilt:
                            dyn_new_freq.get_acf_tilt(
                                filename=str(ACFbindir) +
                                str(round(istart_t, 1))+'/ACFfit_chunk_' +
                                str(int(dyn_new_freq.freq))
                                + '.pdf', plot=True, display=True)
                        dyn_new_freq.get_scint_params(filename=str(ACFbindir) +
                                                      str(round(istart_t, 1)) +
                                                      '/ACF_chunk_' +
                                                      str(int(dyn_new_freq.freq))+'.pdf',
                                                      method='acf2d_approx',
                                                      plot=True, display=True)
                        dyn_new_freq.get_scint_params(filename=str(ACFbindir) +
                                                      str(round(istart_t, 1)) +
                                                      '/ACF_chunk_' +
                                                      str(int(dyn_new_freq.freq))+'.pdf',
                                                      method='acf1d',
                                                      plot=True, display=True)
                    except Exception as e:
                        print(e)
                        counter_nodynspec += 1
                    continue
if merge_data:
    # data_len = 0
    # for i in range(0, len(observations)):
    #     observation_date = observations[i] + '/'
    #     observation_date2 = observations[i]
    #     filedir = str(outdir)+str(observation_date)
    #     outfile = \
    #         str(filedir)+str(psrname)+'_'+str(observation_date2)+'_freq' + \
    #         str(freq_bin)+'_time'+str(time_bin)+'_ScintillationResults_UHF.txt'
    #     if i == 0:
            # Data0 = np.genfromtxt(outfile, delimiter=',',
            #                       dtype=str)  # np.loadtxt()
            # TitleRow = Data0[0, 1:]
    #         Data0 = np.genfromtxt(outfile, delimiter=',', dtype=str)[1:, 1:]
    #         data_len += np.shape(Data0)[0]
    #         Data0 = np.vstack((TitleRow, Data0))
    #     elif i == 1:
    #         Data1 = np.genfromtxt(outfile, delimiter=',', dtype=str)[1:, 1:]
    #         data_len += np.shape(Data1)[0]
    #         Data2 = np.vstack((Data0, Data1))
    #     else:
    #         Data3 = np.genfromtxt(outfile, delimiter=',', dtype=str)[1:, 1:]
    #         data_len += np.shape(Data3)[0]
    #         Data2 = np.vstack((Data2, Data3))
    # np.savetxt(outfile_total, Data2, delimiter=',', fmt='%s')
    # print("This is the length of the data", data_len, " against what we see",
    #       np.shape(Data2)[0])

    for i in range(0, len(observations)):
        observation_date = observations[i] + '/'
        observation_date2 = observations[i]
        filedir = str(outdir)+str(observation_date)
        outfile = \
            str(filedir)+str(psrname)+'_'+str(observation_date2)+'_freq' + \
            str(freq_bin)+'_time'+str(time_bin)+'_ScintillationResults_UHF.txt'
        if i == 0:
            Data0 = Table.read(outfile, format='ascii.csv')
            # Data0.remove_column('name')
            # Data0.remove_column('scint_param_method')
            # Data0.remove_column('fse_tau')
            # Data0.remove_column('fse_dnu')
            # Data0.remove_column('fse_phasegrad')
        elif i == 1:
            Data1 = Table.read(outfile, format='ascii.csv')
            # Data1.remove_column('name')
            # Data1.remove_column('scint_param_method')
            # Data1.remove_column('fse_tau')
            # Data1.remove_column('fse_dnu')
            # Data1.remove_column('fse_phasegrad')
            Data2 = vstack([Data0, Data1])
        else:
            Data3 = Table.read(outfile, format='ascii.csv')
            # Data3.remove_column('name')
            # Data3.remove_column('scint_param_method')
            # Data3.remove_column('fse_tau')
            # Data3.remove_column('fse_dnu')
            # Data3.remove_column('fse_phasegrad')
            Data2 = vstack([Data2, Data3])
    np.savetxt(outfile_total, Data2, delimiter=',', fmt='%s')
    DataTest = np.genfromtxt(outfile_total, delimiter=',', dtype=str)
    try:
        TitleRow = ['name', 'mjd', 'freq', 'bw', 'tobs', 'dt', 'df', 'tau',
                    'tauerr', 'dnu', 'dnuerr', 'fse_tau', 'fse_dnu',
                    'scint_param_method', 'dnu_est', 'nscint', 'acf_tilt',
                    'acf_tilt_err', 'fse_tilt', 'phasegrad',
                    'phasegraderr', 'fse_phasegrad']
    except ValueError as e:
        TitleRow = ['name', 'mjd', 'freq', 'bw', 'tobs', 'dt', 'df', 'tau',
                    'tauerr', 'dnu', 'dnuerr', 'fse_tau', 'fse_dnu',
                    'scint_param_method', 'dnu_est', 'nscint']
        print(e)
    DataTest = np.vstack((TitleRow, DataTest))
    np.savetxt(outfile_total, DataTest, delimiter=',', fmt='%s')

if load_data:
    if load_all:
        file_option1 = "/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/New/DataFiles/Base_UHF_3months_noRFI.txt"
        file_option2 = "/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/New/DataFiles/Base_UHF_3months_RFI.txt"
        file_option3 = "/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/New/DataFiles/All_Scintillation_results.txt"
        file_option4 = "/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/New/DataFiles/Base_UHF_3months_RFI copy.txt"
        params = read_results(file_option3)
    else:
        params = read_results(outfile_total)

    pars = read_par(str(par_dir) + str(psrname) + '.par')

    # Read in arrays
    mjd = float_array_from_dict(params, 'mjd')  # MJD for observation start
    df = float_array_from_dict(params, 'df')  # channel bandwidth
    dnu = float_array_from_dict(params, 'dnu')  # scint bandwidth
    # dnu_est = float_array_from_dict(params, 'dnu_est')  # estimated bandwidth
    dnuerr = float_array_from_dict(params, 'dnuerr')
    tau = float_array_from_dict(params, 'tau')
    tauerr = float_array_from_dict(params, 'tauerr')
    freq = float_array_from_dict(params, 'freq')
    bw = float_array_from_dict(params, 'bw')
    name = np.asarray(params['name'])
    # scintle_num = float_array_from_dict(params, 'scintle_num')
    tobs = float_array_from_dict(params, 'tobs')  # tobs in second
    # rcvrs = np.array([rcvr[0] for rcvr in params['name']])

    # Sort by MJD
    sort_ind = np.argsort(mjd)

    df = np.array(df[sort_ind]).squeeze()
    dnu = np.array(dnu[sort_ind]).squeeze()
    # dnu_est = np.array(dnu_est[sort_ind]).squeeze()
    dnuerr = np.array(dnuerr[sort_ind]).squeeze()
    tau = np.array(tau[sort_ind]).squeeze()
    tauerr = np.array(tauerr[sort_ind]).squeeze()
    mjd = np.array(mjd[sort_ind]).squeeze()
    # rcvrs = np.array(rcvrs[sort_ind]).squeeze()
    freq = np.array(freq[sort_ind]).squeeze()
    tobs = np.array(tobs[sort_ind]).squeeze()
    name = np.array(name[sort_ind]).squeeze()
    # scintle_num = np.array(scintle_num[sort_ind]).squeeze()
    bw = np.array(bw[sort_ind]).squeeze()

    # acf_residuals = np.asarray(acf_residuals[sort_ind]).squeeze()

    # BEFORE filtering make a plot
    # acf_residuals_norm = (acf_residuals - np.min(acf_residuals)) / \
    #     (np.max(acf_residuals) - np.min(acf_residuals))
    # dnu_norm = (dnu - np.min(dnu)) / (np.max(dnu) - np.min(dnu))
    # fig = plt.figure(figsize=(20, 10))
    # fig.subplots_adjust(hspace=0.5, wspace=0.5)
    # ax = fig.add_subplot(1, 1, 1)
    # plt.plot(freq, acf_residuals_norm, color='C1', alpha=0.7)
    # plt.scatter(freq, dnu_norm, c='C0', marker='o', s=Size, alpha=0.6)
    # xl = plt.xlim()
    # plt.xlabel('Observational Frequency (MHz)')
    # plt.ylabel(r'Normalised $\Delta\nu$ and ACF $\sigma$ (arb)')
    # plt.show()
    # plt.close()

    # Used to filter the data
    indicies = np.argwhere((tauerr < 2*tau) * (dnuerr < 2*dnu))
    # * (tau > 38)

    df = df[indicies].squeeze()
    dnu = dnu[indicies].squeeze()
    # dnu_est = dnu_est[indicies].squeeze()
    dnuerr = dnuerr[indicies].squeeze()
    tau = tau[indicies].squeeze()
    tauerr = tauerr[indicies].squeeze()
    mjd = mjd[indicies].squeeze()
    # rcvrs = rcvrs[indicies].squeeze()
    freq = freq[indicies].squeeze()
    tobs = tobs[indicies].squeeze()
    name = name[indicies].squeeze()
    # scintle_num = scintle_num[indicies].squeeze()
    bw = bw[indicies].squeeze()
    # acf_residuals = acf_residuals[indicies].squeeze()

    # AFTER filtering make a plot
    # acf_residuals_norm = (acf_residuals - np.min(acf_residuals)) / \
    #     (np.max(acf_residuals) - np.min(acf_residuals))
    # dnu_norm = (dnu - np.min(dnu)) / (np.max(dnu) - np.min(dnu))
    # fig = plt.figure(figsize=(20, 10))
    # fig.subplots_adjust(hspace=0.5, wspace=0.5)
    # ax = fig.add_subplot(1, 1, 1)
    # plt.plot(freq, acf_residuals_norm, color='C1', alpha=0.7)
    # plt.scatter(freq, dnu_norm, c='C0', marker='o', s=Size, alpha=0.6)
    # xl = plt.xlim()
    # plt.xlabel('Observational Frequency (MHz)')
    # plt.ylabel(r'Normalised $\Delta\nu$ and ACF $\sigma$ (arb)')
    # plt.show()
    # plt.close()

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
    U = get_true_anomaly(mjd, pars)

    true_anomaly = U.squeeze()
    vearth_ra = vearth_ra.squeeze()
    vearth_dec = vearth_dec.squeeze()

    om = pars['OM'] + pars['OMDOT']*(mjd - pars['T0'])/365.2425
    # compute orbital phase
    phase = U*180/np.pi + om
    phase = phase % 360
    # PHASE and observation day #
    name_num = []
    for i in range(0, len(name)):
        for ii in range(0, len(np.unique(name))):
            if name[i] == np.unique(name)[ii]:
                name_num.append(ii)
    name_num = np.asarray(name_num)
    # Calculate Viss

    Aiss = 2.78*10**4  # thin screen, table 2 of Cordes & Rickett (1998)
    D = 0.735  # kpc
    pars = read_par(str(par_dir) + str(psrname) + '.par')
    params = pars_to_params(pars)
    params.add('d', value=D, vary=False)
    params.add('derr', value=0.060, vary=False)
    params.add('s', value=0.7, vary=False)
    params.add('serr', value=0.03, vary=False)
    viss, visserr = scint_velocity(None, dnu, tau, freq, dnuerr, tauerr,
                                   a=Aiss)

    # ORBITAL phase against scintillation bandwidth for each 'observation run'
    fig = plt.figure(figsize=(20, 10))
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    cm = plt.cm.get_cmap('viridis')
    z = mjd
    sc = plt.scatter(phase, dnu, c=z, cmap=cm, s=Size, alpha=0.6)
    # plt.colorbar(sc)
    plt.errorbar(phase, dnu, yerr=dnuerr, fmt=' ', ecolor='k',
                 elinewidth=2, capsize=3, alpha=0.55)
    xl = plt.xlim()
    plt.plot(xl, (df[0], df[0]), color='C2')
    plt.xlabel('Orbital Phase (degrees)')
    plt.ylabel('Scintillation Bandwidth (MHz)')
    plt.title('Orbital Phase and "observation run"')
    plt.xlim(xl)
    plt.savefig("/Users/jacobaskew/Desktop/Dnu_Orbital_Freq.png")
    plt.savefig(plotdir+"Dnu_Orbital_Freq.pdf", dpi=400)
    plt.show()
    plt.close()

    # A plot showing the annual modulation if any?! ANNUAL #
    fig = plt.figure(figsize=(20, 10))
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    ax = fig.add_subplot(1, 1, 1)
    cm = plt.cm.get_cmap('viridis')
    z = phase
    sc = plt.scatter(mjd_annual, dnu, c=z, cmap=cm, s=Size, alpha=0.4)
    plt.colorbar(sc)
    plt.errorbar(mjd_annual, dnu, yerr=dnuerr, fmt=' ', ecolor='k',
                 elinewidth=2, capsize=3, alpha=0.55)
    xl = plt.xlim()
    plt.plot(xl, (0.0332, 0.0332), color='C2')
    # plt.plot(xl, (0.83594, 0.83594), color='C2', linestyle='dashed')
    plt.xlabel('Annual Phase (days)')
    plt.ylabel('Scintillation Bandwidth (MHz)')
    plt.title('Annual Phase and "Orbital Phase"')
    plt.xlim(xl)
    plt.savefig("/Users/jacobaskew/Desktop/Dnu_Freq_Observation.png")
    plt.savefig(plotdir+"Dnu_Freq_Observation.pdf", dpi=400)
    # ax.set_xscale('log')
    # ax.set_yscale('log')
    plt.show()
    plt.close()

    # A temporary plot about the observations less than a channel and high freq
    fig = plt.figure(figsize=(20, 10))
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    ax = fig.add_subplot(1, 1, 1)
    cm = plt.cm.get_cmap('viridis')
    z = name_num
    sc = plt.scatter(freq, dnu, c=z, cmap=cm, s=Size, alpha=0.9)
    # plt.colorbar(sc)
    plt.errorbar(freq, dnu, yerr=dnuerr, fmt=' ', ecolor='k',
                 elinewidth=2, capsize=3, alpha=0.55)
    xl = plt.xlim()
    plt.plot(xl, (df[0], df[0]), color='C2')
    # plt.plot(xl, (0.83594, 0.83594), color='C2', linestyle='dashed')
    plt.xlabel('Observational Frequency (MHz)')
    plt.ylabel('Scintillation Bandwidth (MHz)')
    plt.title('Bandwidth v Frequency "observation run"')
    plt.xlim(xl)
    # plt.ylim(0, 0.2)
    plt.savefig("/Users/jacobaskew/Desktop/Dnu_Freq_Observation.png")
    plt.savefig(plotdir+"Dnu_Freq_Observation.pdf", dpi=400)
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.show()
    plt.close()

    # ORBITAL phase against timescale
    fig = plt.figure(figsize=(20, 10))
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    cm = plt.cm.get_cmap('viridis')
    z = name_num
    sc = plt.scatter(phase, tau, c=z, cmap=cm, s=Size, alpha=0.6)
    # plt.colorbar(sc)
    plt.errorbar(phase, tau, yerr=tauerr, fmt=' ', ecolor='k',
                 elinewidth=2, capsize=3, alpha=0.55)
    xl = plt.xlim()
    plt.plot(xl, (df[0], df[0]), color='C2')
    plt.xlabel('Orbital Phase (degrees)')
    plt.ylabel('Scintillation Timescale (s)')
    plt.title('Timescale and "observation run"')
    plt.xlim(xl)
    plt.savefig("/Users/jacobaskew/Desktop/Dnu_Orbital_Freq.png")
    plt.savefig(plotdir+"Dnu_Orbital_Freq.pdf", dpi=400)
    plt.show()
    plt.close()

    # A plot showing the annual modulation if any?! ANNUAL #
    fig = plt.figure(figsize=(20, 10))
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    ax = fig.add_subplot(1, 1, 1)
    cm = plt.cm.get_cmap('viridis')
    z = name_num
    sc = plt.scatter(mjd_annual, tau, c=z, cmap=cm, s=Size, alpha=0.6)
    # plt.colorbar(sc)
    plt.errorbar(mjd_annual, tau, yerr=tauerr, fmt=' ', ecolor='k',
                 elinewidth=2, capsize=3, alpha=0.55)
    xl = plt.xlim()
    plt.plot(xl, (0.0332, 0.0332), color='C2')
    # plt.plot(xl, (0.83594, 0.83594), color='C2', linestyle='dashed')
    plt.xlabel('Annual Phase (days)')
    plt.ylabel('Scintillation Timescale (s)')
    plt.title('Annual Phase and "observation run"')
    plt.xlim(xl)
    plt.savefig("/Users/jacobaskew/Desktop/Dnu_Freq_Observation.png")
    plt.savefig(plotdir+"Dnu_Freq_Observation.pdf", dpi=400)
    # ax.set_xscale('log')
    # ax.set_yscale('log')
    plt.show()
    plt.close()

    # A temporary plot about the observations less than a channel and high freq
    fig = plt.figure(figsize=(20, 10))
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    ax = fig.add_subplot(1, 1, 1)
    cm = plt.cm.get_cmap('viridis')
    z = name_num
    sc = plt.scatter(freq, tau, c=z, cmap=cm, s=Size, alpha=0.6)
    # plt.colorbar(sc)
    plt.errorbar(freq, tau, yerr=tauerr, fmt=' ', ecolor='k',
                 elinewidth=2, capsize=3, alpha=0.55)
    xl = plt.xlim()
    plt.plot(xl, (df[0], df[0]), color='C2')
    # plt.plot(xl, (0.83594, 0.83594), color='C2', linestyle='dashed')
    plt.xlabel('Observational Frequency (MHz)')
    plt.ylabel('Scintillation Timescale (s)')
    plt.title('Frequency and "observation run"')
    plt.xlim(xl)
    # plt.ylim(0, 0.2)
    plt.savefig("/Users/jacobaskew/Desktop/Dnu_Freq_Observation.png")
    plt.savefig(plotdir+"Dnu_Freq_Observation.pdf", dpi=400)
    # ax.set_xscale('log')
    # ax.set_yscale('log')
    plt.show()
    plt.close()

    # Viss against orbital phase
    fig = plt.figure(figsize=(20, 10))
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    ax = fig.add_subplot(1, 1, 1)
    cm = plt.cm.get_cmap('viridis')
    z = name_num
    sc = plt.scatter(phase, viss, c=z, cmap=cm, s=Size, alpha=0.6)
    # plt.colorbar(sc)
    plt.errorbar(phase, viss, yerr=visserr, fmt=' ', ecolor='k',
                 elinewidth=2, capsize=3, alpha=0.55)
    xl = plt.xlim()
    plt.xlabel('Orbital Phase (degrees)')
    plt.ylabel(r'Scintillation Velocity (km$\,$s$^{-1}$)')
    plt.title('Velocity and "observation run"')
    plt.xlim(xl)
    # plt.ylim(0, 0.2)
    plt.savefig("/Users/jacobaskew/Desktop/Viss_Observation.png")
    plt.savefig(plotdir+"Viss_Observation.pdf", dpi=400)
    # ax.set_xscale('log')
    # ax.set_yscale('log')
    plt.show()
    plt.close()

###############################################################################
if compare:
    # Here we are going to combine the two datasets and attempt to fit the
    # powerlaw

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
    old_U = get_true_anomaly(old_mjd, pars)

    old_true_anomaly = old_U.squeeze()
    old_vearth_ra = old_vearth_ra.squeeze()
    old_vearth_dec = old_vearth_dec.squeeze()

    old_om = pars['OM'] + pars['OMDOT']*(old_mjd - pars['T0'])/365.2425
    # compute orbital phase
    old_phase = old_U*180/np.pi + old_om
    old_phase = old_phase % 360
###############################################################################
    # Combine the data
    freq = np.concatenate((freq, old_freq))
    dnu = np.concatenate((dnu, old_dnu))
    dnuerr = np.concatenate((dnuerr, old_dnuerr))
    phase = np.concatenate((phase, old_phase))
    df = np.concatenate((df, old_df))
###############################################################################
    # Here i want to explore all the possible fits to the entire data at diff
    # phase
    phase_list = []
    dnu_list = []
    freq_list = []
    dnuerr_list = []
    df_list = []
    for i in range(0, 18):
        dnu_list.append(np.asarray(dnu[np.argwhere((phase > i*20) *
                                                   (phase < (i+1)*20))]))
        dnuerr_list.append(np.asarray(dnuerr[np.argwhere((phase > i*20) *
                                                         (phase < (i+1)*20))]))
        phase_list.append(np.asarray(phase[np.argwhere((phase > i*20) *
                                                       (phase < (i+1)*20))]))
        freq_list.append(np.asarray(freq[np.argwhere((phase > i*20) *
                                                     (phase < (i+1)*20))]))
        df_list.append(np.asarray(df[np.argwhere((phase > i*20) *
                                                 (phase < (i+1)*20))]))
    dnu_array = np.asarray(dnu_list)
    phase_array = np.asarray(phase_list)
    freq_array = np.asarray(freq_list)
    dnuerr_array = np.asarray(dnuerr_list)
    df_array = np.asarray(df_list)
    Slopes1 = []
    Slopeerrs1 = []
    Slopes2 = []
    Slopeerrs2 = []
    Slopes3 = []
    Slopeerrs3 = []
    for i in range(0, 18):
        # Old method I hope the new one works
        # #
        # xdata = freq_array[i][np.argwhere((freq_array[i] > 800) *
        #                                   (freq_array[i] < 1200))].flatten()
        # ydata = dnu_array[i][np.argwhere((freq_array[i] > 800) *
        #                                  (freq_array[i] < 1200))].flatten()
        # ydataerr = dnuerr_array[i][np.argwhere((freq_array[i] > 800) *
        #                                        (freq_array[i] < 1200))].flatten()
        # #
        # xdata1 = freq_array[i][np.argwhere(freq_array[i] < 1200)].flatten()
        # ydata1 = dnu_array[i][np.argwhere(freq_array[i] < 1200)].flatten()
        # ydataerr1 = dnuerr_array[i][np.argwhere(freq_array[i] < 1200)].flatten()
        # #
        # xdata2 = freq_array[i][np.argwhere((freq_array[i] > 800) *
        #                                    (freq_array[i] > 1200))].flatten()
        # ydata2 = dnu_array[i][np.argwhere((freq_array[i] > 800) *
        #                                   (freq_array[i] > 1200))].flatten()
        # ydataerr2 = dnuerr_array[i][np.argwhere((freq_array[i] > 800) *
        #                                         (freq_array[i] > 1200))].flatten()
        # #
        # xdata3 = freq_array[i][np.argwhere(freq_array[i] > 800)].flatten()
        # ydata3 = dnu_array[i][np.argwhere(freq_array[i] > 800)].flatten()
        # ydataerr3 = dnuerr_array[i][np.argwhere(freq_array[i] > 800)].flatten()
        # #
        # new method
        #
        xdata1 = freq_array[i][np.argwhere((freq_array[i] > 800) *
                                           (df_array[i] < 0.5))].flatten()
        ydata1 = dnu_array[i][np.argwhere((freq_array[i] > 800) *
                                          (df_array[i] < 0.5))].flatten()
        ydataerr1 = dnuerr_array[i][np.argwhere((freq_array[i] > 800) *
                                                (df_array[i] < 0.5))].flatten()
        #
        xdata2 = freq_array[i][np.argwhere((freq_array[i] > 800) *
                                           (df_array[i] > 0.5))].flatten()
        ydata2 = dnu_array[i][np.argwhere((freq_array[i] > 800) *
                                          (df_array[i] > 0.5))].flatten()
        ydataerr2 = dnuerr_array[i][np.argwhere((freq_array[i] > 800) *
                                                (df_array[i] > 0.5))].flatten()
        #
        xdata3 = freq_array[i][np.argwhere((freq_array[i] > 800))].flatten()
        ydata3 = dnu_array[i][np.argwhere((freq_array[i] > 800))].flatten()
        ydataerr3 = \
            dnuerr_array[i][np.argwhere((freq_array[i] > 800))].flatten()
        #

        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(1, 1, 1)
        fig.subplots_adjust(hspace=0.5, wspace=0.5)
        xdata4 = np.linspace(np.min(freq_array[1])*0.75,
                             np.max(freq_array[1])*1.25,
                             1000)
# xdata the highly channelised data
        if xdata1.size > 0:
            plt.scatter(xdata1, ydata1, c='C0', s=Size/4)
            plt.errorbar(xdata1, ydata1, yerr=ydataerr1,
                         fmt=' ', ecolor='k',
                         elinewidth=2, capsize=3, alpha=0.55)
            xl = plt.xlim()
            popt, pcov = curve_fit(func1, xdata1, ydata1)
            perr = np.sqrt(np.diag(pcov))
            plt.plot(xdata4, func1(xdata4, *popt), 'C1', label=r'UHF $\alpha$=' +
                     str(round(popt[0], 2))+r'$\pm$'+str(round(perr[0], 2)))
            plt.fill_between(xdata4.flatten(),
                             func1(xdata4, *[popt[0]+perr[0], popt[1]]).flatten(),
                             func1(xdata4, *[popt[0]-perr[0], popt[1]]).flatten(),
                             alpha=0.5, color='C1')
            Slopes1.append(popt)
            Slopeerrs1.append(perr[0])
# xdata2 the low (1024) channelised data
        if xdata2.size > 0:
            plt.scatter(xdata2, ydata2, c='C0', s=Size/4)
            plt.errorbar(xdata2, ydata2, yerr=ydataerr2, fmt=' ', ecolor='k',
                         elinewidth=2, capsize=3, alpha=0.55)
            xl = plt.xlim()
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
# xdata3 all the data
        popt3, pcov3 = curve_fit(func3, xdata3, ydata3)
        perr3 = np.sqrt(np.diag(pcov3))
        plt.plot(xdata4, func3(xdata4, *popt3), 'C4', label=r'ALL $\alpha$=' +
                 str(round(popt3[0], 2))+r'$\pm$'+str(round(perr3[0], 2)))
        xl = plt.xlim()
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
        plt.xlim(min(xdata4), max(xdata4))
        plt.ylim(0, np.max(dnu)*1.05)
        plt.show()
# Final Plot
    Slopes_average1 = np.mean(Slopes1, axis=0)
    Slopeerrs_median1 = np.median(Slopeerrs1)
    Slopes_average2 = np.mean(Slopes2, axis=0)
    Slopeerrs_median2 = np.median(Slopeerrs2)
    Slopes_average3 = np.mean(Slopes3, axis=0)
    Slopeerrs_median3 = np.median(Slopeerrs3)

    new_sort = np.argwhere(df < 0.5)
    lowres_sort = np.argwhere(df > 0.5)

    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(1, 1, 1)
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    ax.set_xscale('log')
    ax.set_yscale('log')
    xdata4 = np.linspace(np.min(freq)*0.75,
                         np.max(freq)*1.25,
                         1000)
    plt.scatter(freq[new_sort], dnu[new_sort], c='C1', marker='^', s=Size)
    plt.scatter(freq[lowres_sort], dnu[lowres_sort], c='C0', marker='v',
                s=Size)
    plt.scatter(old_freq, old_dnu, c='C0', s=Size)
    plt.errorbar(freq, dnu, yerr=dnuerr, fmt=' ', ecolor='k',
                 elinewidth=4, capsize=6, alpha=0.55)
    plt.plot(xdata4, func1(xdata4, *Slopes_average1), 'C1', label=r'16k Data $\alpha$=' +
             str(round(Slopes_average1[0], 2))+r'$\pm$'+str(round(Slopeerrs_median1, 2)))
    xl = plt.xlim()
    plt.fill_between(xdata4.flatten(),
                     func1(xdata4, *[Slopes_average1[0]+Slopeerrs_median1,
                                     Slopes_average1[1]]).flatten(),
                     func1(xdata4, *[Slopes_average1[0]-Slopeerrs_median1,
                                     Slopes_average1[1]]).flatten(),
                     alpha=0.2, color='C1')
    # plt.plot(xdata4, func2(xdata4, *Slopes_average2), 'C3', label=r'1k Data $\alpha$=' +
    #          str(round(Slopes_average2[0], 2))+r'$\pm$'+str(round(Slopeerrs_median2, 2)))
    # plt.fill_between(xdata4.flatten(),
    #                  func2(xdata4, *[Slopes_average2[0]+Slopeerrs_median2,
    #                                  Slopes_average2[1]]).flatten(),
    #                  func2(xdata4, *[Slopes_average2[0]-Slopeerrs_median2,
    #                                  Slopes_average2[1]]).flatten(),
    #                  alpha=0.2, color='C3')
    # plt.plot(xdata4, func3(xdata4, *Slopes_average3), 'C4', label=r'All Data $\alpha$=' +
    #          str(round(Slopes_average3[0], 2))+r'$\pm$'+str(round(Slopeerrs_median3, 2)))
    # plt.fill_between(xdata4.flatten(),
    #                  func3(xdata4, *[Slopes_average3[0]+Slopeerrs_median3,
    #                                  Slopes_average3[1]]).flatten(),
    #                  func3(xdata4, *[Slopes_average3[0]-Slopeerrs_median3,
    #                                  Slopes_average3[1]]).flatten(),
    #                  alpha=0.2, color='C4')
    # plt.plot(xl, (old_df[0], old_df[0]), color='C2', linestyle='dashed',
    #          label='LowRes df='+str(round(old_df[0], 2))+'MHz')
    plt.plot(xl, (old_df[0], old_df[0]), color='C2', linestyle='dashed',
             label='1k L-Band Channel bw')
    # plt.plot(xl, (df[0], df[0]), color='C2',
    #          label='HighRes df='+str(round(df[0], 2))+'MHz')
    plt.plot(xl, (df[0], df[0]), color='C2',
             label='16k UHF Channel bw')
    plt.plot(xl, (np.unique(df)[1], np.unique(df)[1]), color='C0',
             label='16k L-Band Channel bw')
    plt.plot(xl, (np.unique(df)[1]*2, np.unique(df)[1]*2), color='C0',
             linestyle='dashed', label='8k L-Band Channel bw')
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Scintillation Bandwidth (MHz)')
    # ax.legend(fontsize='xx-small')
    ax.legend()
    ax.set_xticks([700, 800, 1000, 1200, 1400, 1600, 1800])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_yticks([0.02, 0.1, 0.5, 1, 2, 4])
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    # ax.axis([1, 100000, 1, 100000])
    # ax.loglog()
    # ax.xaxis.set_tick_params(length=5, width=2)
    # ax.yaxis.set_tick_params(length=5, width=2)
    # plt.xticks(np.arange(min(freq), max(freq)+1, 1000))
    # plt.yticks(np.arange(min(dnu), max(dnu)+1, 1000))
    # plt.xlim(min(xdata4), max(xdata4))
    plt.ylim(np.min(dnu)*0.90, np.max(dnu)*1.1)
    plt.xlim(np.min(freq)*0.95, np.max(freq)*1.05)
    plt.show()
    plt.close()

    # This plot shows the freq v dnu for all high res UHF
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(1, 1, 1)
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    ax.set_xscale('log')
    ax.set_yscale('log')
    xdata4 = np.linspace(np.min(freq)*0.75,
                         np.max(freq)*1.25,
                         1000)
    plt.scatter(freq[new_sort], dnu[new_sort], c='C1', marker='^', s=Size)
    plt.errorbar(freq[new_sort], dnu[new_sort],
                 yerr=dnuerr[new_sort.flatten()], fmt=' ',
                 ecolor='k', elinewidth=4, capsize=6, alpha=0.55)
    plt.plot(xdata4, func1(xdata4, *Slopes_average1), 'C1',
             label=r'16k Data $\alpha$=' + str(round(Slopes_average1[0], 2)) +
             r'$\pm$'+str(round(Slopeerrs_median1, 2)))
    xl = plt.xlim()
    plt.fill_between(xdata4.flatten(),
                     func1(xdata4, *[Slopes_average1[0]+Slopeerrs_median1,
                                     Slopes_average1[1]]).flatten(),
                     func1(xdata4, *[Slopes_average1[0]-Slopeerrs_median1,
                                     Slopes_average1[1]]).flatten(),
                     alpha=0.2, color='C1')
    plt.plot(xl, (df[0], df[0]), color='C2',
             label='16k UHF Channel bw')
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Scintillation Bandwidth (MHz)')
    # ax.legend(fontsize='xx-small')
    ax.legend()
    ax.set_xticks([700, 800, 900, 1000, 1400, 1600, 1800])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_yticks([0.04, 0.06, 0.1, 0.2, 2, 4])
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    plt.ylim(np.min(dnu[new_sort])*0.90, np.max(dnu[new_sort])*1.1)
    plt.xlim(np.min(freq[new_sort])*0.95, np.max(freq[new_sort])*1.05)
    plt.show()
    plt.close()

    # This plot shows the phase v dnu for all high res UHF
    # fig = plt.figure(figsize=(20, 10))
    # ax = fig.add_subplot(1, 1, 1)
    # fig.subplots_adjust(hspace=0.5, wspace=0.5)
    # xdata4 = np.linspace(np.min(phase),
    #                      np.max(phase),
    #                      3600)
    # plt.scatter(phase[new_sort], dnu[new_sort], c='C1', marker='^', s=Size)
    # plt.errorbar(phase[new_sort], dnu[new_sort],
    #              yerr=dnuerr[new_sort.flatten()], fmt=' ',
    #              ecolor='k', elinewidth=4, capsize=6, alpha=0.55)
    # xl = plt.xlim()
    # plt.xlabel('Frequency (MHz)')
    # plt.ylabel('Scintillation Bandwidth (MHz)')
    # # ax.legend(fontsize='xx-small')
    # ax.legend()
    # plt.ylim(np.min(dnu[new_sort])*0.90, np.max(dnu[new_sort])*1.1)
    # plt.xlim(np.min(phase[new_sort]), np.max(phase[new_sort]))
    # plt.show()
    # plt.close()

###############################################################################
# This section is only for testing some observations, the problem has been solved
# pars = read_par(str(par_dir) + str(psrname) + '.par')

# for i in range(0, len(observations)):
#     observation_date2 = observations[i]
#     dynspecfile2 = \
#         outdir+'DynspecPlotFiles/'+observation_date2 + \
#         'Zap_CompleteDynspec.dynspec'
#     if os.path.exists(dynspecfile2):
#         sim = Simulation()
#         dyn = Dynspec(dyn=sim, process=False)
#         dyn.load_file(filename=dynspecfile2)
#         if observation_date2 == '2022-09-18':
#             dyn.mjd = 59840.29919
#             mjd = np.linspace(59840.29919, 59840.29919+np.max(dyn.times)/86400,
#                               1000)
#         else:
#             mjd = np.linspace(dyn.mjd, dyn.mjd+np.max(dyn.times)/86400, 1000)
#             ssb_delays = get_ssb_delay(mjd, pars['RAJ'], pars['DECJ'])
#             mjd += np.divide(ssb_delays, 86400)  # add ssb delay
#             U = get_true_anomaly(mjd, pars)
#             om = pars['OM'] + pars['OMDOT']*(mjd - pars['T0'])/365.2425
#             # compute orbital phase
#             





#             phase[phase > 360] = phase[phase > 360] - 360
#             Tmin = int(((mjd[np.argmin(abs(phase - 240))] - dyn.mjd) * 86400) / 60)
#             Tmax = Tmin + 30
#             crop_dyn = cp(dyn)
#             crop_dyn.crop_dyn(fmin=990, fmax=1020, tmin=Tmin, tmax=Tmax)
#             spec = cp(crop_dyn)
#             correctspec = cp(crop_dyn)
#             spec.plot_dyn(filename='/Users/jacobaskew/Desktop/Dynspec.png',
#                           figsize=(20, 10))
#             spec.get_scint_params(filename='/Users/jacobaskew/Desktop/ACF.png',
#                                   method='acf2d_approx', plot=True, display=True)
#             correctspec.correct_dyn()
#             correctspec.plot_dyn(
#                 filename='/Users/jacobaskew/Desktop/Correct_Dynspec.png',
#                           figsize=(20, 10))
#             correctspec.get_scint_params(
#                 filename='/Users/jacobaskew/Desktop/Correct_ACF.png',
#                 method='acf2d_approx', plot=True, display=True)
    
#         image1A = plt.imread("/Users/jacobaskew/Desktop/Dynspec.png")
#         image1B = plt.imread("/Users/jacobaskew/Desktop/Correct_Dynspec.png")
#         image2A = plt.imread("/Users/jacobaskew/Desktop/ACF_2Dfit.png")
#         image2B = plt.imread("/Users/jacobaskew/Desktop/Correct_ACF_2Dfit.png")
    
#         fig, axes = plt.subplots(1, 2)
#         for column in [0, 1]:
#             ax = axes[column]
#             ax.axis('off')
#             print(column)
#             if column == 0:
#                 ax.imshow(image1A)
#             if column == 1:
#                 ax.imshow(image1B)
#         plt.tight_layout()
#         plt.savefig("/Users/jacobaskew/Desktop/" + str(observations[i]) +
#                     "Fig1.pdf", dpi=300)
#         plt.show()
    
#         fig, axes = plt.subplots(1, 2)
#         for column in [0, 1]:
#             ax = axes[column]
#             ax.axis('off')
#             print(column)
#             if column == 0:
#                 ax.imshow(image2A)
#             if column == 1:
#                 ax.imshow(image2B)
#         plt.tight_layout()
#         plt.savefig("/Users/jacobaskew/Desktop/" + str(observations[i]) +
#                     "Fig2.pdf", dpi=300)
#         plt.show()
