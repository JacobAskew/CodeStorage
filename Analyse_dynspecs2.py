#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 18:38:56 2021

@author: dreardon
"""

from scintools.dynspec import Dynspec
from scintools.scint_utils import write_results, read_results, read_par, \
        float_array_from_dict, get_ssb_delay, get_earth_velocity, \
        get_true_anomaly, pars_to_params, scint_velocity
from scintools.scint_models import veff_thin_screen, effective_velocity_annual
import corner
from lmfit import Parameters, Minimizer
import matplotlib.pyplot as plt
import numpy as np
import glob
from copy import deepcopy as cp
import bilby
from bilby.core import result
import pdb
import itertools as it

datadir = '/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/Data/'
psrname = 'J0737-3039A'
pulsar = '0737-3039A'
spectradir = '/Users/jacobaskew/Desktop/DoublePulsar_Project/' + str(pulsar) + '/Spectra/'
#psrname = 'J1757-5322'

#dynspecs = sorted(glob.glob(datadir + psrname + '/*2019-12-14-19:44*.dynspec'))
dynspecs = sorted(glob.glob(datadir + '/*ar.dynspec'))

uhf = False
measure = 1 # 0 = No measure, 1 = Daniel measure, 2 = Jacob measure
if measure == 1:
    outdir = '/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/DanielsOutput/' 
    outfile = '/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/DanielsOutput/Results.txt' 
if measure == 2 or measure == 0:
    outdir = '/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/Datafiles/' 
    outfile = str(outdir) + 'J' + str(pulsar) + '_ScintillationResults.txt'

model = True
######
def SearchEclipse(start_mjd, tobs):
    end_mjd = start_mjd + tobs/86400
    Eclipse_mjd = np.genfromtxt('/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/Datafiles/Eclipse_mjd.txt', delimiter=','
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

#
bw_top = 50
f_top = 1675
f_low = 1025
f_mid = 1400
f_uhf = 600
bw_mid = (f_mid / f_top)**2 * bw_top
num_mid = int(200/bw_mid)
bw_low = (f_low / f_top)**2 * bw_top
num_low = int(100/bw_low)
bw_uhf = (f_uhf / f_top)**2 * bw_top
num_uhf = int(371.875/bw_uhf)

###################################
# Mucking around with dnu  and tau #

# f_uhf = 925 - 575
# f_high = 1675
# bw_high = 50
# bw_uhf = (f_uhf / f_high)**2 * bw_high

# plt.scatter(phase, tau)

# tau_new = (tau-np.min(tau))/(np.max(tau)-np.min(tau))
# phase_new = (phase-np.min(phase))/(np.max(phase)-np.min(phase))

# plt.scatter(phase_new, tau_new)

# tau_new2 = tau_new - np.median(tau_new)

# plt.scatter(phase_new, tau_new2)


# outfile = '/Users/jacobaskew/Desktop/results.txt'
# # L-Band #
# dyn = Dynspec(filename='/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/Data/J0737-3039A_2019-07-20-06:52:56_ch5.0_sub5.0.ar.dynspec', process=False)
# dyn.trim_edges()
# dyn.crop_dyn(fmin=1650, fmax=1700)
# dyn.get_acf_tilt()
# dyn.get_scint_params(method='acf2d_approx',
#                               flux_estimate=True)
# write_results(outfile, dyn=dyn)
# dyn.plot_dyn()

# dyn = Dynspec(filename='/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/Data/J0737-3039A_2019-07-20-06:52:56_ch5.0_sub5.0.ar.dynspec', process=False)
# dyn.trim_edges()
# dyn.crop_dyn(fmin=1666, fmax=1682)
# dyn.get_acf_tilt()
# dyn.get_scint_params(method='acf2d_approx',
#                               flux_estimate=True)
# write_results(outfile, dyn=dyn)
# dyn.plot_dyn()

# dyn = Dynspec(filename='/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/Data/J0737-3039A_2019-07-20-06:52:56_ch5.0_sub5.0.ar.dynspec', process=False)
# dyn.trim_edges()
# dyn.crop_dyn(fmin=1650, fmax=1666)
# dyn.get_acf_tilt()
# dyn.get_scint_params(method='acf2d_approx',
#                               flux_estimate=True)
# write_results(outfile, dyn=dyn)
# dyn.plot_dyn()

# # UHF #

# dyn = Dynspec(filename='/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/Data/J0737-3039A_2020-03-28-12:39:12_ch5.0_sub5.0.ar.dynspec', process=False)
# dyn.trim_edges()
# dyn.crop_dyn(fmin=700, fmax=900)
# dyn.get_acf_tilt()
# dyn.get_scint_params(method='acf2d_approx',
#                               flux_estimate=True)
# write_results(outfile, dyn=dyn)
# dyn.plot_dyn()

# dyn = Dynspec(filename='/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/Data/J0737-3039A_2020-03-28-12:39:12_ch5.0_sub5.0.ar.dynspec', process=False)
# dyn.trim_edges()
# dyn.crop_dyn(fmin=600, fmax=700)
# dyn.get_acf_tilt()
# dyn.get_scint_params(method='acf2d_approx',
#                               flux_estimate=True)
# write_results(outfile, dyn=dyn)
# dyn.plot_dyn()

# Data = read_results(outfile)
# df = float_array_from_dict(Data, 'df')  # channel bandwidth
# dnu = float_array_from_dict(Data, 'dnu')  # scint bandwidth
# dnu_est = float_array_from_dict(Data, 'dnu_est')  # estimated bandwidth
# dnuerr = float_array_from_dict(Data, 'dnuerr')
# freq = float_array_from_dict(Data, 'freq')

# freqs = freq
# Experimental = dnu
# Experimental_est = dnu_est
# Observational_Bandwidth = df
# Theoretical = []
# for x in range(0, len(Experimental)):
#     Theoretical.append((((float(freqs[x]))/(1660.59))**2)*float(Experimental[x]))
# Theoretical = np.asarray(Theoretical)
# X = [min(Theoretical),max(Theoretical)]
# Y = X

# fig = plt.figure(figsize=(5, 5))
# ax = fig.add_subplot(1, 1, 1)
# plt.scatter(Theoretical,Experimental, label='Measured dnu', c='r')
# plt.errorbar(Theoretical,Experimental, yerr=dnuerr, fmt=' ', ecolor='r')
# plt.plot(Theoretical,Experimental, c='r')
# plt.scatter(Theoretical,Experimental_est, label='Estimated dnu', c='b')
# plt.errorbar(Theoretical,Experimental_est, yerr=dnuerr, fmt=' ', ecolor='b')
# plt.plot(Theoretical,Experimental_est, c='b')
# plt.scatter(Theoretical,Observational_Bandwidth, label='Channel Bandwidth', c='g')
# plt.plot(Theoretical,Observational_Bandwidth, c='g')
# plt.scatter(X,Y, label='Expected Values', c='k')
# plt.plot(X,Y, c='k')
# ax.legend()
# plt.xlabel("Theoretical Scint Bandwidth")
# plt.ylabel("Measured Scint Bandwidth")

###################################
# plt.plot(X,Y)

# plt.plot(Y,X)

# dyn = Dynspec(filename='/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/Data/J0737-3039A_2020-03-28-12:39:12_ch5.0_sub5.0.ar.dynspec', process=False)
# dyn.trim_edges()
# dyn.plot_dyn()


# dyn_crop = cp(dyn)
# dyn_crop.crop_dyn(fmin=1300, fmax=1500)
# for istart_f in range(1300, 1500, int(bw_mid)):
#     for istart_t in range(0, int(dyn.tobs/60), 10):
#         try:
#             FreqRange = 'Mid'
    
#             dyn_new = cp(dyn_crop)
#             dyn_new.crop_dyn(fmin=istart_f, fmax=istart_f + int(bw_mid)
#                              , tmin=istart_t, tmax=istart_t + 10) 
#             dyn_new.trim_edges()
#             if dyn_new.tobs <=5 or dyn_new.bw < bw_mid*0.9:  # if only 5 minutes remaining
#                 continue
#             dyn_new.zap()
#             dyn_new.refill(linear=False)
#             dyn_new.get_acf_tilt()
#             dyn_new.get_scint_params(method='acf2d_approx',
#                                          flux_estimate=True)
#             dyn_new.plot_dyn() # filename=str(spectradir) + str(Filename) + '_' + str(FreqRange) + '_Spectra.png'
#             dyn_new.plot_acf(fit=True) # filename=str(plotdir) + str(Filename) + '_' + str(FreqRange) + '_ACF.png',
#             write_results(outfile, dyn=dyn_new)
#         except Exception as e:
#             print(e)
#             continue

# Data = read_results(outfile)
# print(len(Data))
# print()
# print(Bandwidth)

#

# ######
# dyn = Dynspec(filename='/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/Data/J0737-3039A_2019-07-20-06:22:48_ch5.0_sub5.0.ar.dynspec', process=False)

# dyn.get_acf_tilt()
# dyn.get_scint_params(method='acf2d_approx',
#                          flux_estimate=True)


# Eclipse_index = SearchEclipse(dyn.mjd, dyn.tobs)

# # Eclipse_minute = dyn.times/60
# Eclipse_minute_before = int((dyn.times[Eclipse_index-3]/60))
# Eclipse_minute_after = int(np.ceil((dyn.times[Eclipse_index+3]/60)))

# for istart_t in it.chain(range(0, Eclipse_minute_before, Eclipse_minute_before)
#                          , range(Eclipse_minute_after, int(dyn.tobs/60), Eclipse_minute_before)):
    
#     print(istart_t)
#     # dyn = Dynspec(filename='/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/Data/J0737-3039A_2019-07-20-06:22:48_ch5.0_sub5.0.ar.dynspec', process=False)
#     # dyn.crop_dyn(tmin=istart_t, tmax=istart_t + Eclipse_minute_before-1)
#     # dyn.plot_dyn(filename='/Users/jacobaskew/Desktop/Spectra.png')
#     # if dyn.tobs/60 < dyn.tau/60:
#     #     print("Failed")
#     #     continue

# # dyn_crop = cp(dyn)
# # dyn_crop.crop_dyn(fmin=975, fmax=1075)
# if len(SearchEclipse(dyn.mjd, dyn.tobs)) == 1:
#     Eclipse_index = SearchEclipse(dyn.mjd, dyn.tobs)
# start1_t = 0
# end1_t = int(dyn.times[Eclipse_index-3]/60)
# start2_t = int(dyn.times[Eclipse_index+3]/60)
# end2_t = int(dyn.tobs/60)
    
    
# for istart_t in it.chain(range(start1_t, end1_t, 10), range(start2_t, end2_t, 10)):

#     print(istart_t)
#     dyn = Dynspec(filename='/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/Data/J0737-3039A_2019-07-20-06:22:48_ch5.0_sub5.0.ar.dynspec', process=False)
#     dyn.plot_dyn(filename='/Users/jacobaskew/Desktop/Spectra.png')
#     dyn.crop_dyn(tmin=istart_t, tmax=istart_t + 10)
#     print("Passed")
#     if dyn.tobs/60 < 10:
#         print("Failed")
#         continue

# ######

if measure == 1:
    for dynspec in dynspecs:
        
        File1=dynspec.split(str(datadir))[1]
        Filename=str(File1.split('.')[0])

        try:
            dyn = Dynspec(filename=dynspec, process=False)
            
            # start_mjd=dyn.mjd
            # tobs = dyn.tobs
            # Eclipse_index = SearchEclipse(start_mjd, tobs)
            # dyn.dyn[:,Eclipse_index-3:Eclipse_index+3] = 0

            if not uhf and dyn.freq < 1000:
                continue
            elif uhf and dyn.freq > 1000:
                continue
            #dyn.refill()
            dyn.plot_dyn(filename=str(outdir) + str(Filename) + '_Spectra.png',
                         display=True)
            #dyn.plot_sspec(lamsteps=True,
            #               filename=datadir+psrname+'/plots/'+dyn.name+\
            #                   '_sspec.png', display=True)
            
            # Crop this region in L-band
            if uhf:
                dyn.crop_dyn(fmin=600, fmax=900)
            else:
                dyn.crop_dyn(fmin=1300, fmax=1500)
            dyn.plot_dyn()
        except Exception as e:
            print(e)
            continue
        
        
        dyn_crop = cp(dyn)
        if uhf:
            dyn_crop.crop_dyn(fmin=600, fmax=700)  # 600 to 700
            print('Measuring scint params for 600MHz to 700MHz...')
        else:
            dyn_crop.crop_dyn(fmax=1400)  # 1300 to 1400
            print('Measuring scint params for 1300MHz to 1400MHz...')
        for istart_t in range(0, int(dyn.tobs/60), 10):
            try:
                dyn_new = cp(dyn_crop)
                dyn_new.crop_dyn(tmin=istart_t, tmax=istart_t + 10)
                if dyn_new.tobs <=5:  # if only 5 minutes remaining
                    continue
                # dyn_new.plot_dyn()
                # dyn_new.plot_sspec(lamsteps=True)
                dyn_new.get_acf_tilt()
                dyn_new.get_scint_params(method='acf2d_approx',
                                         flux_estimate=True)
                print(dyn_new.dnu, dyn_new.dnu_est)
                write_results(outfile, dyn=dyn_new)
            except Exception as e:
                print(e)
                continue
                
        dyn_crop = cp(dyn)
        if uhf:
            dyn_crop.crop_dyn(fmin=700, fmax=800)  # 700 to 800
            print('Measuring scint params for 700MHz to 800MHz...')
        else:
            dyn_crop.crop_dyn(fmin=1400)  # 1400 to 1500
            print('Measuring scint params for 1400MHz to 1500MHz...')
        for istart_t in range(0, int(dyn.tobs/60), 10):
            try:
                dyn_new = cp(dyn_crop)
                dyn_new.crop_dyn(tmin=istart_t, tmax=istart_t + 10)
                # dyn_new.plot_dyn()
                # dyn_new.plot_sspec(lamsteps=True)
                dyn_new.get_acf_tilt()
                dyn_new.get_scint_params(method='acf2d_approx',
                                         flux_estimate=True)
                print(dyn_new.dnu, dyn_new.dnu_est)
                write_results(outfile, dyn=dyn_new)
            except Exception as e:
                print(e)
                continue
            
        dyn_crop = cp(dyn)
        if uhf:
            dyn_crop.crop_dyn(fmin=800, fmax=900)  # 800 to 900
            print('Measuring scint params for 800MHz to 800MHz...')
        else:
            continue
        for istart_t in range(0, int(dyn.tobs/60), 10):
            try:
                dyn_new = cp(dyn_crop)
                dyn_new.crop_dyn(tmin=istart_t, tmax=istart_t + 10)
                # dyn_new.plot_dyn()
                # dyn_new.plot_sspec(lamsteps=True)
                dyn_new.get_acf_tilt()
                dyn_new.get_scint_params(method='acf2d_approx',
                                         flux_estimate=True)
                print(dyn_new.dnu, dyn_new.dnu_est)
                write_results(outfile, dyn=dyn_new)
            except Exception as e:
                print(e)
                continue
            
if measure == 2:
    for dynspec in dynspecs:
        
        File1=dynspec.split(str(datadir))[1]
        Filename=str(File1.split('.')[0])
        
        try:
            dyn = Dynspec(filename=dynspec, process=False)
            dyn.trim_edges()
            start_mjd=dyn.mjd
            tobs = dyn.tobs
            Eclipse_index = SearchEclipse(start_mjd, tobs)
            if Eclipse_index != None:
                dyn.dyn[:,Eclipse_index-3:Eclipse_index+3] = 0
            dyn.plot_dyn(filename=str(spectradir) + str(Filename) + '_Spectra.png')
        except Exception as e:
            print(e)
            continue
        
        if dyn.freq > 1000:
            
            dyn_crop = cp(dyn)
            dyn_crop.crop_dyn(fmin=1650, fmax=1700)
            bw_high = 50/3
            f_min = 1650
            f_max = 1700
            for istart_f in range(f_min, f_max, int(bw_high)):
                for istart_t in range(0, int(dyn.tobs/60), 10):
                    try:
                        FreqRange = 'High'
                        dyn_new = cp(dyn_crop)
                        dyn_new.crop_dyn(fmin=istart_f, fmax=istart_f + int(bw_high)
                                         , tmin=istart_t, tmax=istart_t + 10) 
                        dyn_new.trim_edges()
                        if dyn_new.tobs <=5 or dyn_new.bw < bw_high*0.9:  # if only 5 minutes remaining
                            continue
                        dyn_new.zap()
                        dyn_new.refill(linear=False)
                        dyn_new.get_acf_tilt()
                        dyn_new.get_scint_params(method='acf2d_approx',
                                                     flux_estimate=True)
                        # print(dyn_new.dnu, dyn_new.dnu_est)
                        dyn_new.plot_dyn(filename=str(spectradir) + str(Filename) + '_' + str(FreqRange) + '_Spectra.png')
                        dyn_new.plot_acf(fit=True) # filename=str(plotdir) + str(Filename) + '_' + str(FreqRange) + '_ACF.png',
                        write_results(outfile, dyn=dyn_new)
                    except Exception as e:
                        print(e)
                        continue
                
            dyn_crop = cp(dyn)    
            dyn_crop.crop_dyn(fmin=1300, fmax=1500)
            for istart_f in range(1300, 1500, int(bw_mid)):
                for istart_t in range(0, int(dyn.tobs/60), 10):
                    try:
                        FreqRange = 'Mid'
                        dyn_new = cp(dyn_crop)
                        dyn_new.crop_dyn(fmin=istart_f, fmax=istart_f + int(bw_mid)
                                         , tmin=istart_t, tmax=istart_t + 10) 
                        dyn_new.trim_edges()
                        if dyn_new.tobs <=5 or dyn_new.bw < bw_mid*0.9:  # if only 5 minutes remaining
                            continue
                        dyn_new.zap()
                        dyn_new.refill(linear=False)
                        dyn_new.get_acf_tilt()
                        dyn_new.get_scint_params(method='acf2d_approx',
                                                     flux_estimate=True)
                        dyn_new.plot_dyn() # filename=str(spectradir) + str(Filename) + '_' + str(FreqRange) + '_Spectra.png'
                        dyn_new.plot_acf(fit=True) # filename=str(plotdir) + str(Filename) + '_' + str(FreqRange) + '_ACF.png',
                        write_results(outfile, dyn=dyn_new)
                    except Exception as e:
                        print(e)
                        continue

            # for istart_t in range(0, int(dyn.tobs/60), 10):
            #     try:
            #         FreqRange = 'Mid'
            #         Bandwidth = 200
            #         dyn_new = cp(dyn_crop)
            #         dyn_new.crop_dyn(tmin=istart_t, tmax=istart_t + 10)
            #         dyn_new.trim_edges()
            #         dyn_new.zap()
            #         dyn_new.refill(linear=False)
            #         dyn_new.get_acf_tilt()
            #         dyn_new.get_scint_params(method='acf2d_approx',
            #                                      flux_estimate=True)
            #         # print(dyn_new.dnu, dyn_new.dnu_est)
            #         dyn_new.plot_dyn(filename=str(spectradir) + str(Filename) + '_' + str(FreqRange) + '_' + str(istart_t) + '_Spectra.png')
            #         dyn_new.plot_acf(fit=True) # filename=str(plotdir) + str(Filename) + '_' + str(FreqRange) + '_ACF.png',
            #         write_results(outfile, dyn=dyn_new)
            #     except Exception as e:
            #         print(e)
            #         continue
                
            dyn_crop = cp(dyn)
            dyn_crop.crop_dyn(fmin=975, fmax=1075)
            for istart_f in range(975, 1075, int(bw_low)):
                for istart_t in range(0, int(dyn.tobs/60), 10):
                    try:
                        FreqRange = 'Low'
                        dyn_new = cp(dyn_crop)
                        dyn_new.crop_dyn(fmin=istart_f, fmax=istart_f + int(bw_low)
                                         , tmin=istart_t, tmax=istart_t + 10) 
                        dyn_new.trim_edges()
                        if dyn_new.tobs <=5 or dyn_new.bw < bw_low*0.9:  # if only 5 minutes remaining
                            continue
                        dyn_new.zap()
                        dyn_new.refill(linear=False)
                        dyn_new.get_acf_tilt()
                        dyn_new.get_scint_params(method='acf2d_approx',
                                                     flux_estimate=True)
                        dyn_new.plot_dyn() # filename=str(spectradir) + str(Filename) + '_' + str(FreqRange) + '_Spectra.png'
                        dyn_new.plot_acf(fit=True) # filename=str(plotdir) + str(Filename) + '_' + str(FreqRange) + '_ACF.png',
                        write_results(outfile, dyn=dyn_new)
                    except Exception as e:
                        print(e)
                        continue          
                    
            # for istart_t in range(0, int(dyn.tobs/60), 10):
            #     try:
            #         FreqRange = 'Low'
            #         Bandwidth = 100
            #         dyn_new = cp(dyn_crop)
            #         dyn_new.crop_dyn(tmin=istart_t, tmax=istart_t + 10)
            #         dyn_new.trim_edges()
            #         dyn_new.zap()
            #         dyn_new.refill(linear=False)
            #         dyn_new.get_acf_tilt()
            #         dyn_new.get_scint_params(method='acf2d_approx',
            #                                      flux_estimate=True)
            #         # print(dyn_new.dnu, dyn_new.dnu_est)
            #         dyn_new.plot_dyn(filename=str(spectradir) + str(Filename) + '_' + str(FreqRange) + '_' + str(istart_t) + '_Spectra.png')
            #         dyn_new.plot_acf(fit=True) # filename=str(plotdir) + str(Filename) + '_' + str(FreqRange) + '_ACF.png',
            #         write_results(outfile, dyn=dyn_new)
            #     except Exception as e:
            #         print(e)
            #         continue
                
            # if dyn.freq < 1000:
                
            #     for istart_t in range(0, int(dyn.tobs/60), 10):
            #         try:
            #             FreqRange = 'UHF'
            #             # Bandwidth = 100
            #             dyn_new = cp(dyn_crop)
            #             dyn_new.crop_dyn(tmin=istart_t, tmax=istart_t + 10)
            #             dyn_new.trim_edges()
            #             dyn_new.zap()
            #             dyn_new.refill(linear=False)
            #             dyn_new.get_acf_tilt()
            #             dyn_new.get_scint_params(method='acf2d_approx',
            #                                           flux_estimate=True)
            #             # print(dyn_new.dnu, dyn_new.dnu_est)
            #             dyn_new.plot_dyn(filename=str(spectradir) + str(Filename) + '_' + str(FreqRange) + '_' + str(istart_t) + '_Spectra.png')
            #             dyn_new.plot_acf(fit=True) # filename=str(plotdir) + str(Filename) + '_' + str(FreqRange) + '_ACF.png',
            #             write_results(outfile, dyn=dyn_new)
            #         except Exception as e:
            #             print(e)
            #             continue

            
        #     try:
        #         dyn = Dynspec(filename=dynspec, process=False)
        #         FreqRange = 'High'
        #         Bandwidth = 50
        #         dyn.crop_dyn(fmin=1650, fmax=1700)
        #         dyn.trim_edges()
        #         dyn.zap
        #         dyn.refill(linear=False)
        #         dyn.get_acf_tilt()
        #         dyn.get_scint_params(method='acf2d_approx',
        #                                      flux_estimate=True)
        #         # dyn.get_scint_params(method='acf2d_approx')
        #         # dyn.get_acf_tilt()
        #         # dyn.calc_acf(method='acf2d_approx')
        #         dyn.plot_dyn(filename=str(spectradir) + str(Filename) + '_' + str(FreqRange) + '_Spectra.png')
        #         dyn.plot_acf(fit=True) # filename=str(plotdir) + str(Filename) + '_' + str(FreqRange) + '_ACF.png',
        #         write_results(outfile, dyn=dyn)
                
        #     except Exception as e:
        #         print(e)
        #         continue

            # try:

            #     dyn = Dynspec(filename=dynspec, process=False)
            #     FreqRange = 'Mid'
            #     Bandwidth = 200
                
            #     dyn.crop_dyn(fmin=1300, fmax=1500)
            #     dyn.trim_edges()
            #     dyn.zap
            #     dyn.refill(linear=False)
            #     dyn.get_acf_tilt()
            #     dyn.get_scint_params(method='acf2d_approx',
            #                                  flux_estimate=True)
            #     # dyn.get_scint_params(method='acf2d_approx')
            #     # dyn.get_acf_tilt()
            #     # dyn.calc_acf(method='acf2d_approx')
            #     dyn.plot_dyn(filename=str(spectradir) + str(Filename) + '_' + str(FreqRange) + '_Spectra.png')
            #     dyn.plot_acf(fit=True) # filename=str(plotdir) + str(Filename) + '_' + str(FreqRange) + '_ACF.png',
            #     write_results(outfile, dyn=dyn)
                
            # except Exception as e:
            #     print(e)
            #     continue            
            
            # try:
                
            #     dyn = Dynspec(filename=dynspec, process=False)
            #     FreqRange = 'Low'
            #     Bandwidth = 100
            #     dyn.crop_dyn(fmin=975, fmax=1075)
            #     dyn.trim_edges()
            #     dyn.zap
            #     dyn.refill(linear=False)
            #     dyn.get_acf_tilt()
            #     dyn.get_scint_params(method='acf2d_approx',
            #                                  flux_estimate=True)
            #     # dyn.get_scint_params(method='acf2d_approx')
            #     # dyn.get_acf_tilt()
            #     # dyn.calc_acf(method='acf2d_approx')
            #     dyn.plot_dyn(filename=str(spectradir) + str(Filename) + '_' + str(FreqRange) + '_Spectra.png')
            #     dyn.plot_acf(fit=True) # filename=str(plotdir) + str(Filename) + '_' + str(FreqRange) + '_ACF.png',
            #     write_results(outfile, dyn=dyn)
                
            # except Exception as e:
            #     print(e)
            #     continue
        
if model:
    par_dir = '/Users/jacobaskew/Desktop/Swinburne_Daniel/ParFiles/'
    results_dir = outdir
    params = read_results(outfile)
    
    # pars = read_par(par_dir + psrname + '.par')
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
    
    """
    Do corrections!
    """
    
    indicies = np.argwhere((tauerr < 0.2*tau) * (tau < 1200) * (np.sqrt(dnu)/tau < 0.01) * (dnuerr < 0.2*dnu))
    
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
    
    # Make MJD centre of observation, instead of start
    mjd = mjd + tobs/86400/2
    
    # Form Viss from the data
    Aiss = 2.78*10**4  # thin screen, table 2 of Cordes & Rickett (1998)
    D = 1  # kpc
    # viss = Aiss * np.sqrt(D * dnu ) / (freq/1000 * tau)
    # visserr = 0.2*viss  # temporary
    # viss = 1/tau
    # visserr = 0.2*viss#viss*(1/tauerr**2)
    ind_low = np.argwhere((freq < 1100))
    viss_low, visserr_low = scint_velocity(None, dnu_est[ind_low], tau[ind_low], freq[ind_low], dnuerr[ind_low], 
                                                  tauerr[ind_low], a=Aiss)

    viss, visserr = scint_velocity(None, dnu, tau, freq, dnuerr, 
                                                  tauerr, a=Aiss)
    #visserr *= 15
    
    #
    # Experiment #
    
    # Theoretical = []
    # for x in range(0, len(Experimental)):
    #     Theoretical.append((((float(freqs[x]))/(1660.59))**2)*float(Experimental[x]))
    # Theoretical = np.asarray(Theoretical)
    
    dnu_tmp1 = []
    dnu_tmp2 = []
    dnu_tmp3 = []
    dnu_est_tmp1 = []
    dnu_est_tmp2 = []
    dnu_est_tmp3 = []
    dnuerr_tmp1 = []
    dnuerr_tmp2 = []
    dnuerr_tmp3 = []
    freq1 = []
    freq2 = []
    freq3 = []
    
    for x in range(0, len(dnu)):
        if freq[x] < 1100:
            dnu_tmp1.append(dnu[x])
            dnu_est_tmp1.append(dnu_est[x])
            dnuerr_tmp1.append(dnuerr[x])
            freq1.append(freq[x])
        if freq[x] < 1500 and freq[x] > 1200:
            dnu_tmp2.append(dnu[x])
            dnu_est_tmp2.append(dnu_est[x])
            dnuerr_tmp2.append(dnuerr[x])
            freq2.append(freq[x])
        if freq[x] > 1600:
            dnu_tmp3.append(dnu[x])
            dnu_est_tmp3.append(dnu_est[x])
            dnuerr_tmp3.append(dnuerr[x])
            freq3.append(freq[x])
            
    dnu_avg = [np.mean(dnu_tmp1),np.mean(dnu_tmp2),np.mean(dnu_tmp3)]
    dnu_est_avg = [np.mean(dnu_est_tmp1),np.mean(dnu_est_tmp2),np.mean(dnu_est_tmp3)]
    dnuerr_avg = [np.mean(dnuerr_tmp1),np.mean(dnuerr_tmp2),np.mean(dnuerr_tmp3)]
    
    freq1_avg = np.mean(freq1)
    freq2_avg = np.mean(freq2)
    freq3_avg = np.mean(freq3)
    
    freqs = [freq1_avg,freq2_avg,freq3_avg]
    
    dfs = [0.83594,0.83594,0.83594]
    
    Expected = [(((freqs[0])/(1661.8240476190472))**2)*dnu_avg[0],(((freqs[1])/(1661.8240476190472))**2)*dnu_avg[1],(((freqs[2])/(1661.8240476190472))**2)*dnu_avg[2]]
    
    # dnu_avg_ind = np.argsort(dnu_avg)
    # dnu_est_avg_ind = np.argsort(dnu_est_avg)
    # dnuerr_avg_ind = np.argsort(dnuerr_avg)
    
    # X = np.linspace(np.min(freq[dnu_avg_ind]),np.max(freq[dnu_avg_ind]))
    # Y = Expected

    
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(1, 1, 1)
    plt.scatter(freqs,dnu_avg, label='Measured dnu', c='r', alpha=0.7)
    plt.errorbar(freqs,dnu_avg, yerr=dnuerr_avg, fmt=' ', ecolor='r', alpha=0.5)
    plt.scatter(freqs,dnu_est_avg, label='Estimated dnu', c='b', alpha=0.7)
    plt.errorbar(freqs,dnu_est_avg, yerr=dnuerr_avg, fmt=' ', ecolor='b', alpha=0.5)
    plt.plot(freqs,dfs, c='g', alpha=0.7, label='Channel Bandwidth')
    plt.scatter(freqs,Expected, label='Expected Values', c='k', alpha=0.5)
    plt.plot(freqs,Expected, c='k', alpha=0.5)
    ax.legend(fontsize="xx-small")
    plt.xlabel("Frequency")
    plt.ylabel("Measured Scint Bandwidth")

    
    #
    
    # Plotting Begins here #
    
    fig = plt.figure(figsize=(15, 10))
    # plt.figure(figsize=(12,6))
    plt.errorbar(dnu_est, dnu, yerr=dnuerr, fmt='o', alpha=0.8, label='outliers')
    ax = fig.add_subplot(1, 1, 1)
    inds = np.argwhere((dnu < df))
    plt.errorbar(dnu_est[inds], dnu[inds], yerr=dnuerr[inds].squeeze(), fmt='o', alpha=0.8, label='dnu<df')
    #plt.yscale('log')
    #plt.xscale('log')
    plt.grid()
    plt.xlim([0, 1])
    plt.ylim([0, 1.5])
    plt.plot([0, 1], [0, 1], 'k', zorder=3)
    plt.ylabel('Measured scint bandwidth (MHz)')
    plt.xlabel('Estimated scint bandwidth (MHz)')
    ax.legend(fontsize='xx-small')
    plt.show()
    
    
    plt.figure(figsize=(12,6))
    plt.errorbar(freq, dnu, yerr=dnuerr, fmt='o', alpha=0.8)
    inds = np.argwhere((dnu < df))
    plt.errorbar(freq[inds], dnu[inds], yerr=dnuerr[inds].squeeze(), fmt='o', alpha=0.8)
    plt.yscale('log')
    plt.xscale('log')
    plt.grid()
    plt.ylim([0, 1.5])
    plt.ylabel('Measured scint bandwidth (MHz)')
    plt.xlabel('Observing frequency (MHz)')
    plt.show()
    
    #import sys
    #sys.exit()
    
    plt.figure(figsize=(12,6))
    inds = np.argwhere((freq > 1300))
    plt.errorbar(mjd[inds], viss[inds], yerr=visserr[inds].squeeze()*15, fmt='o', )
    inds = np.argwhere((freq < 1300))
    plt.errorbar(mjd[inds], viss[inds], yerr=visserr[inds].squeeze()*15, fmt='o', )
    inds = np.argwhere((dnu < 0.5*df))
    #plt.errorbar(mjd[inds], viss[inds], yerr=visserr[inds].squeeze()*15, fmt='x', color='k')
    plt.xlabel('MJD')
    plt.ylabel('Viss (km/s)')
    # plt.ylim(0, 200)
    plt.title(psrname)
    plt.show()
    
    #import sys
    #sys.exit()
    
    mjd_annual = mjd % 365.2425
    plt.errorbar(mjd_annual, viss, yerr=visserr, fmt='o', )
    plt.xlabel('Annual phase (arb)')
    plt.ylabel('Viss')
    # plt.ylim(0, 200)
    plt.title(psrname)
    plt.show()
    
    
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
    phase[phase>360] = phase[phase>360] - 360
    
    
    
    ind_high = np.argwhere(freq > 1600)
    ind_mid = np.argwhere((freq > 1100) * (freq < 1600))
    ind_low = np.argwhere((freq < 1100))

    plt.figure(figsize=(8,4))
    plt.scatter(phase[ind_high].flatten(), viss[ind_high].flatten(), c='red', alpha=0.6)
    plt.scatter(phase[ind_mid].flatten(), viss[ind_mid].flatten(), c='blue', alpha=0.6)
    plt.scatter(phase[ind_low].flatten(), viss_low.flatten(), c='green', alpha=0.6)
    plt.errorbar(phase[ind_high].flatten(), viss[ind_high].flatten(), yerr=visserr[ind_high].flatten(), fmt=' ', ecolor='red', alpha=0.4)
    plt.errorbar(phase[ind_mid].flatten(), viss[ind_mid].flatten(), yerr=visserr[ind_mid].flatten(), fmt=' ', ecolor='blue', alpha=0.4)
    plt.errorbar(phase[ind_low].flatten(), viss_low.flatten(), yerr=visserr_low.flatten(), fmt=' ', ecolor='green', alpha=0.4)
    plt.xlabel('Orbital phase (degrees)')
    plt.ylabel('Viss (km/s)')
    plt.xlim(0, 360)
    plt.title(psrname + ' Scintillation velocity')
    plt.show()
    plt.close()
    
    #import sys
    #sys.exit()
    # """
    # Old Fitting routine
    # """
    
    # refit = True
    # nitr = 50
    # chisqr = np.Inf
    
    # print('Doing fit')
    # if refit:
    #     for itr in range(0, nitr):
    #         print(itr)
    
    #         params = pars_to_params(pars)
    #         params.add('s', value=np.random.uniform(low=0, high=1), min=0.0, max=1.0)
    #         #params.add('d', value=np.random.normal(loc=1, scale=0.2), vary=True, min=0, max=100)  # psr distance in kpc
    #         params.add('d', value=0.8, vary=False)  # psr distance in kpc
    #         params.add('vism_ra', value=np.random.normal(loc=0, scale=20), vary=True, min=-100, max=100)
    #         params.add('vism_dec', value=np.random.normal(loc=0, scale=20), vary=True, min=-100, max=100)
    #         params.add('R', value=np.random.uniform(low=0, high=1), vary=True, min=0, max=1)
    #         params.add('psi', value=np.random.uniform(low=0, high=180), vary=True, min=0, max=180)
    #         params.add('kappa', value=np.random.uniform(low=0.2, high=5), vary=True, min=0, max=5)
            
    #         # Pulsar binary params
    #         #params.add('KIN', value=np.random.uniform(low=0, high=180), vary=True, min=0, max=180)
    #         params.add('sense', value=np.random.uniform(low=0, high=1), vary=True, min=0, max=1)
    #         params.add('SINI', value=0.999932, vary=False)
    #         params.add('KOM', value=np.random.uniform(low=0, high=360), vary=True, min=0, max=360)
    #         params['OMDOT'].vary = True
    
    #         func = Minimizer(veff_thin_screen, params,
    #                          fcn_args=(viss, 1/visserr, U,
    #                                    vearth_ra, vearth_dec, mjd))
    
    #         results = func.minimize()
    #         if results.chisqr < chisqr:
    #             chisqr = results.chisqr
    #             results_new = results
    
    # results = results_new
    # print(results.params)
    
    
    # results = results_new
    # func = Minimizer(veff_thin_screen, results.params,
    #                  fcn_args=(viss, 1/visserr, U,
    #                            vearth_ra, vearth_dec, mjd))
    # print('Doing mcmc posterior sample')
    # mcmc_results = func.emcee(steps=1000, burn=200, nwalkers=100, is_weighted=False)
    # truths = []
    # for var in mcmc_results.var_names:
    #     truths.append(mcmc_results.params[var].value)
        
    # # Make latex not break when it sees an underscore in the name
    # mcmc_results.var_names = [a.replace('_', '-') for a in mcmc_results.var_names]
    
    # corner.corner(mcmc_results.flatchain,
    #               labels=mcmc_results.var_names,
    #               truths=truths)
    # plt.show()
    # results = mcmc_results
    
    # #KIN = results.params['KIN'].value
    # #KINerr = results.params['KIN'].stderr
    # sense = results.params['sense'].value
    # senseerr = results.params['sense'].stderr
    # #kappa = results.params['kappa'].value
    # #kappaerr = results.params['kappa'].stderr
    # D = results.params['d'].value
    # Derr = results.params['d'].stderr
    # R = results.params['R'].value
    # Rerr = results.params['R'].stderr
    # psi = results.params['psi'].value
    # psierr = results.params['psi'].stderr
    # OMDOT = results.params['OMDOT'].value
    # OMDOTerr = results.params['OMDOT'].stderr
    # KOM = results.params['KOM'].value
    # KOMerr = results.params['KOM'].stderr
    # s = results.params['s'].value
    # serr = results.params['s'].stderr
    # vism_ra = results.params['vism_ra'].value
    # vism_raerr = results.params['vism_ra'].stderr
    # vism_dec = results.params['vism_dec'].value
    # vism_decerr = results.params['vism_dec'].stderr
    
    # coeff = np.sqrt(2 * D * (1 - s) / s)
    
    # print('=====================================')
    # print('sense', sense, senseerr)
    # print('OMDOT', OMDOT, OMDOTerr)
    # #print('kappa', kappa, kappaerr)
    # print('D', D, Derr)
    # print('R', R, Rerr)
    # print('psi', psi, psierr)
    # print('KOM', KOM, KOMerr)
    # print('vism (RA)', vism_ra, vism_raerr)
    # print('vism (DEC)', vism_dec, vism_decerr)
    # print('s', s, serr)
    # print('=====================================')
    
    # model = (viss - results.residual) * coeff
    # viss_scaled, visserr_scaled = scint_velocity(results.params, dnu, tau,
    #                                              freq, dnuerr, tauerr, a=Aiss)
    
    # plt.errorbar(U, viss_scaled, yerr=visserr_scaled*15, fmt='o', )
    # plt.scatter(U, model, marker='x', c='k')
    # plt.xlabel('True anomaly (rad)')
    # plt.ylabel('Viss')
    # #plt.ylim(0, 200)
    # plt.title(psrname)
    # plt.show()
    
    """
    Bilby fitting routine
    """
    # import bilby
    # from bilby.core import result
    params = pars_to_params(pars)
    params.add('SINI', value=0.999932, vary=False)
    
    label = 'Scintillation'
    
    """
    ############# Bilby-compatible model #############
    """
    def veff_thin_screen_bilby(xdata, s, sense, KOM, d, vism_ra, vism_dec, 
                               R_1, psi_1, R_2, psi_2, R_3, psi_3, R_4, psi_4, 
                               R_5, psi_5, R_6, psi_6, R_7, psi_7, R_8, psi_8, 
                               R_9, psi_9, R_10, psi_10):
        """
        Effective velocity thin screen model.
        Uses Eq. 4 from Rickett et al. (2014) for anisotropy coefficients.
        """ 
            
    
        params_ = dict(params)

        params_['s'] = s
        params_['sense'] = sense
        params_['KOM'] = KOM
        params_['d'] = d
        params_['vism_ra'] = vism_ra
        params_['vism_dec'] = vism_dec
        params_['R_1'] = R_1
        params_['psi_1'] = psi_1
        params_['R_2'] = R_2
        params_['psi_2'] = psi_2
        params_['R_3'] = R_3
        params_['psi_3'] = psi_3
        params_['R_4'] = R_4
        params_['psi_4'] = psi_4
        params_['R_5'] = R_5
        params_['psi_5'] = psi_5
        params_['R_6'] = R_6
        params_['psi_6'] = psi_6
        params_['R_7'] = R_7
        params_['psi_7'] = psi_7
        params_['R_8'] = R_8
        params_['psi_8'] = psi_8
        params_['R_9'] = R_9
        params_['psi_9'] = psi_9
        params_['R_10'] = R_10
        params_['psi_10'] = psi_10
        kappa = 1
    
        veff_ra, veff_dec, vp_ra, vp_dec = \
            effective_velocity_annual(params_, true_anomaly,
                                      vearth_ra, vearth_dec, mjd=mjd)
    
        veff_ra -= vism_ra
        veff_dec -= vism_dec
    
        model = np.zeros(np.shape(mjd))
        
        stepsize = np.ptp(mjd)/nvary
        psi_array = [psi_1, psi_2, psi_3, psi_4, psi_5, psi_6, psi_7, psi_8, psi_9, psi_10]
        R_array = [R_1, R_2, R_3, R_4, R_5, R_6, R_7, R_8, R_9, R_10]
        
        for ii in range(0, nvary):
            
            indices = np.argwhere((mjd > np.min(mjd)+ii*stepsize) * 
                                  (mjd < np.min(mjd)+(ii+1)*stepsize))
    
            psi = psi_array[ii]
            R = R_array[ii]
            
            psi *= np.pi / 180  # anisotropy angle
    
            gamma = psi
            cosa = np.cos(2 * gamma)
            sina = np.sin(2 * gamma)
    
            # quadratic coefficients
            a = (1 - R * cosa) / np.sqrt(1 - R**2)
            b = (1 + R * cosa) / np.sqrt(1 - R**2)
            c = -2 * R * sina / np.sqrt(1 - R**2)
    
            # coefficient to match model with data
            coeff = 1 / np.sqrt(2 * d * (1 - s) / s)
            
            veff = kappa * (np.sqrt(a*veff_dec[indices]**2 + b*veff_ra[indices]**2 +
                                    c*veff_ra[indices]*veff_dec[indices]))
            model[indices] = coeff * veff / s
    
        return model
    
    """
    ############# End Bilby-compatible model #############
    """
    
    likelihood = bilby.likelihood.GaussianLikelihood(mjd, viss, 
                                                 veff_thin_screen_bilby, visserr*15)
    injection_parameters = None


    priors = dict()
    priors['sense'] = bilby.core.prior.Uniform(0, 1, 'sense')
    priors['KOM'] = bilby.core.prior.Uniform(0, 360, 'KOM', boundary='periodic')
    priors['s'] = bilby.core.prior.Uniform(0, 1, 's')
    priors['d'] = bilby.core.prior.Uniform(0, 10, 'd')
    priors['vism_ra'] = bilby.core.prior.Uniform(-1000, 1000, 'vism_ra')
    priors['vism_dec'] = bilby.core.prior.Uniform(-1000, 1000, 'vism_dec')
    priors['R_1'] = bilby.core.prior.Uniform(0, 1, 'R_1')
    priors['psi_1'] = bilby.core.prior.Uniform(0, 180, 'psi_1', boundary='periodic')
    priors['R_2'] = bilby.core.prior.Uniform(0, 1, 'R_2')
    priors['psi_2'] = bilby.core.prior.Uniform(0, 180, 'psi_2', boundary='periodic')
    priors['R_3'] = bilby.core.prior.Uniform(0, 1, 'R_3')
    priors['psi_3'] = bilby.core.prior.Uniform(0, 180, 'psi_3', boundary='periodic')
    priors['R_4'] = bilby.core.prior.Uniform(0, 1, 'R_4')
    priors['psi_4'] = bilby.core.prior.Uniform(0, 180, 'psi_4', boundary='periodic')
    priors['R_5'] = bilby.core.prior.Uniform(0, 1, 'R_5')
    priors['psi_5'] = bilby.core.prior.Uniform(0, 180, 'psi_5', boundary='periodic')
    priors['R_6'] = bilby.core.prior.Uniform(0, 1, 'R_6')
    priors['psi_6'] = bilby.core.prior.Uniform(0, 180, 'psi_6', boundary='periodic')
    priors['R_7'] = bilby.core.prior.Uniform(0, 1, 'R_7')
    priors['psi_7'] = bilby.core.prior.Uniform(0, 180, 'psi_7', boundary='periodic')
    priors['R_8'] = bilby.core.prior.Uniform(0, 1, 'R_8')
    priors['psi_8'] = bilby.core.prior.Uniform(0, 180, 'psi_8', boundary='periodic')
    priors['R_9'] = bilby.core.prior.Uniform(0, 1, 'R_9')
    priors['psi_9'] = bilby.core.prior.Uniform(0, 180, 'psi_9', boundary='periodic')
    priors['R_10'] = bilby.core.prior.Uniform(0, 1, 'R_10')
    priors['psi_10'] = bilby.core.prior.Uniform(0, 180, 'psi_10', boundary='periodic')
    nvary = 10
    
    # import pickle
    # results = pickle.load(open("/Users/dreardon/Desktop/double_pulsar_scint/results10.pkl", "rb"))
    # results.plot_with_data(veff_thin_screen_bilby, mjd, viss)
    # plt.show()
    
    import sys
    sys.exit()
    
    results = bilby.core.sampler.run_sampler(
        likelihood, priors=priors, sampler='dynesty', label=label,
        nlive=500, verbose=True, resume=False, 
        outdir=outdir)
    
    results.plot_corner()
    print(np.shape(mjd), np.shape(viss))
    results.plot_with_data(veff_thin_screen_bilby, mjd, viss, ndraws=100, 
                           xlabel='mjd', ylabel='Viss')
    
    
    
    
    
    
    
    
    