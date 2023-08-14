#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 14:43:39 2021

@author: jacobaskew
"""

# A new script that is suitable for OzStar and does what needs to be done!
# Measure scint bandwidth and timescale
# Measure in blocks of frequency and time, large enough to maximise # of scintles and cover the scintillation timescale

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
import os

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
datadir = '/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/RawData/'
spectradir = '/Users/jacobaskew/Desktop/DoublePulsar_Project/' + str(pulsar) + '/Spectra/'
par_dir = '/Users/jacobaskew/Desktop/Swinburne_Daniel/ParFiles/'
plotdir = '/Users/jacobaskew/Desktop/DoublePulsar_Project/' + str(pulsar) + '/Plots/'
eclipsefile = '/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/Datafiles/Eclipse_mjd.txt'
outdir = '/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/Datafiles/' 
HighFreqDir = '/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/HighFreq/Plots/'
##############################################################################
# Also Common #
freq_bin = 40
time_bin = 30
dynspecs = sorted(glob.glob(datadir + '/*ar.dynspec'))
outfile = str(outdir) + str(psrname) + '_ScintillationResults_HighFreq_' + str(freq_bin) + 'MHz_' + str(time_bin) + 'min.txt'
# outfile = str(outdir) + str(psrname) + '_ScintillationResults_HighFreq3.txt'

##############################################################################
# Manual Inputs #
measure = False
model = True
zap = False
linear = False
##############################################################################
def SearchEclipse(start_mjd, tobs):
    end_mjd = start_mjd + tobs/86400
    Eclipse_mjd = np.genfromtxt(eclipsefile, delimiter=','
                                ,encoding=None, dtype=float)
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

        File1=dynspec.split(str(datadir))[1]
        Filename=str(File1.split('.')[0])
                
        try:
            dyn = Dynspec(filename=dynspec, process=False)
            dyn.trim_edges()
            if dyn.freq < 1000:  # Ignore the UHF band
                continue
            start_mjd=dyn.mjd
            tobs = dyn.tobs
            Eclipse_index = SearchEclipse(start_mjd, tobs)
            if Eclipse_index != None:
                dyn.dyn[:,Eclipse_index-3:Eclipse_index+3] = 0
            # dyn.plot_dyn(filename=str(spectradir) + str(Filename) + '_Spectra.png')
            # dyn.plot_dyn(filename=str(HighFreqDir) + str(Filename) + '/' + str(Filename) + '_Spectra.png')
        except Exception as e:
            print(e)
            continue
        if dyn.freq > 1000:

            try: 
                os.mkdir(str(HighFreqDir)+str(Filename))
            except OSError as error: 
                print(error)   
            try:
                os.mkdir(str(HighFreqDir) + 'Spectra_' + str(freq_bin) + 'MHz_' + str(time_bin) + 'min.txt')
            except OSError as error: 
                print(error)   

            dyn_crop = cp(dyn)
            dyn_crop.crop_dyn(fmin=1300, fmax=1500)
            f_min_init = 1630
            f_max = 1670
            f_init = int((f_max + f_min_init)/2)
            bw_init = int(f_max - f_min_init)
            for istart_t in range(0, int(dyn.tobs/60), int(time_bin)):
                try:
                    dyn_new = cp(dyn_crop)
                    dyn_new.crop_dyn(fmin=f_min_init, fmax=f_min_init+bw_init, tmin=istart_t, tmax=istart_t + time_bin)  #fmin=istart_f, fmax=istart_f + int(bw_top),
                    dyn_new.trim_edges()
                    if dyn_new.tobs/60 < (time_bin - 5):
                        print("Spectra rejected: Not enough time!")
                        print(str(dyn_new.tobs/60) + " < " + str (time_bin - 5))
                        continue
                    if (np.max(dyn_new.freqs) - np.min(dyn_new.freqs)) < freq_bin - 5:
                        print("Spectra rejected: Not enough frequency!")
                        print(str((np.max(dyn_new.freqs) - np.min(dyn_new.freqs))) + " < " + str (freq_bin - 5))
                        continue
                    print("Spectra Time accepeted: ")
                    print(str(dyn_new.tobs/60) + " > " + str (time_bin - 5))
                    print("Spectra Frequency accepeted: ")
                    print(str((np.max(dyn_new.freqs) - np.min(dyn_new.freqs))) + " > " + str (freq_bin - 5))
                    if zap:
                        dyn_new.zap()
                    if linear:
                        dyn_new.refill(linear=True)
                    else:
                        dyn_new.refill(linear=False)
                    dyn_new.get_acf_tilt(plot=False, display=False)
                    dyn_new.get_scint_params(method='acf2d_approx',
                                                 flux_estimate=True,
                                                 plot=False, display=False)
                    dyn_new.plot_dyn(filename=str(HighFreqDir) + str(Filename) + '/' + str(Filename) + '_' + str(freq_bin) + 'MHz_' + str(time_bin) + '_fmin' + str(f_min_init) + '_tmin' + str(istart_t) + '_Spectra.png')  #
                    dyn_new.plot_dyn(filename=str(HighFreqDir) + 'Spectra_' + str(freq_bin) + 'MHz_' + str(time_bin) + 'min.txt' + '/' + str(Filename) + '_' + str(freq_bin) + 'MHz_' + str(time_bin) + '_fmin' + str(f_min_init) + '_tmin' + str(istart_t) + '_Spectra.png')  #
                    write_results(outfile, dyn=dyn_new)
                except Exception as e:
                    print(e)
                    continue
