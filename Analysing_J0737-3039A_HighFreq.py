#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 18:38:56 2021

@author: dreardon
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
datadir = '/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/DataTmp2/'
spectradir = '/Users/jacobaskew/Desktop/DoublePulsar_Project/' + str(pulsar) + '/Spectra/'
par_dir = '/Users/jacobaskew/Desktop/Swinburne_Daniel/ParFiles/'
plotdir = '/Users/jacobaskew/Desktop/DoublePulsar_Project/' + str(pulsar) + '/Plots/'
eclipsefile = '/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/Datafiles/Eclipse_mjd.txt'
outdir = '/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/Datafiles/' 
HighFreqDir = '/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/HighFreq/Plots/'
##############################################################################
##### Single Observation Testing ####
# observation = 'J0737-3039A_2020-12-22-03:15:28_ch5.0_sub5.0.ar.dynspec'
# obs = str(datadir) + str(observation)
# dyn = Dynspec(filename=obs, process=False)
# dyn.crop_dyn(fmin=1600, tmin=120, tmax=150)
# dyn.trim_edges()
# dyn.refill(linear=False)
# dyn.calc_acf()
# dyn.calc_sspec(window_frac=0.1, prewhite=False, lamsteps=True)
# dyn.plot_dyn(filename='/Users/jacobaskew/Desktop/Spectra.png')
# dyn.plot_acf(filename='/Users/jacobaskew/Desktop/ACF.png', fit=True)
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
            dyn.plot_dyn(filename=str(spectradir) + str(Filename) + '_Spectra.png')
            # dyn.plot_dyn(filename=str(HighFreqDir) + str(Filename) + '/' + str(Filename) + '_Spectra.png')
        except Exception as e:
            print(e)
            continue
        if dyn.freq > 1000:
            fmin = 1600
            try:
                dyn = Dynspec(filename=dynspec, process=False)
                dyn.trim_edges()
                dyn.crop_dyn(fmin=fmin)
                start_mjd=dyn.mjd
                tobs = dyn.tobs
                Eclipse_index = SearchEclipse(start_mjd, tobs)
                if Eclipse_index != None:
                    dyn.dyn[:,Eclipse_index-3:Eclipse_index+3] = 0
                # dyn.plot_dyn(filename=str(spectradir) + str(Filename) + '_Spectra.png')
                dyn.plot_dyn(filename=str(HighFreqDir) + 'Spectra/' + str(Filename) + '_HighFreqSpectra.png')
            except Exception as e:
                print(e)
                continue

            # print("LOOK AT ME")
            # print(str(HighFreqDir)+str(Filename))
            # print("LOOK AT ME")

            try: 
                os.mkdir(str(HighFreqDir)+str(Filename))
                # print("SUCCESSSSSSSSSSSSSS")
                # print(str(HighFreqDir)+str(Filename))
# '/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/HighFreq/Plots/'
# '/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/HighFreq/Plots/J0737-3039A_2019-04-22-12:06:24_ch5/J0737-3039A_2019-04-22-12:06:24_ch5_Spectra.png'
            except OSError as error: 
                print(error)   
            try:
                os.mkdir(str(HighFreqDir) + 'Spectra_' + str(freq_bin) + 'MHz_' + str(time_bin) + 'min.txt')
            except OSError as error: 
                print(error)   
                # print(str(HighFreqDir)+str(Filename))
            
            dyn_crop = cp(dyn)
            dyn_crop.crop_dyn(fmin=fmin)
            f_max = int(round(np.max(dyn.freqs), -1))
            f_high = int((f_max + fmin)/2)
            bw_high = int(f_max - fmin)
            for istart_f in range(fmin, f_max, int(freq_bin)):
                for istart_t in range(0, int(dyn.tobs/60), int(time_bin)):
                    try:
                        FreqRange = 'High'
                        dyn_new = cp(dyn_crop)
                        dyn_new.crop_dyn(fmin=istart_f, fmax=istart_f+freq_bin, tmin=istart_t, tmax=istart_t + time_bin)  #fmin=istart_f, fmax=istart_f + int(bw_top),
                        dyn_new.trim_edges()
                        # print((np.max(dyn_new.freqs) - np.min(dyn_new.freqs)))
                        # print((np.max(dyn_new.freqs) - np.min(dyn_new.freqs)))
                        # print((np.max(dyn_new.freqs) - np.min(dyn_new.freqs)))
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
                        # dyn_new.get_scint_params(filename=str(HighFreqDir) + str(Filename) + '/' + str(Filename) + '_' + str(FreqRange) + '_fmin' + str(istart_f) + '_tmin' + str(istart_t) + '_ACFittingPlots.png', plot=True, method='acf1d')
                        dyn_new.get_acf_tilt(plot=False, display=False)
                        dyn_new.get_scint_params(method='acf2d_approx',
                                                     flux_estimate=True,
                                                     plot=False, display=False)
                        # print(dyn_new.dnu, dyn_new.dnu_est)
                        dyn_new.plot_dyn(filename=str(HighFreqDir) + str(Filename) + '/' + str(Filename) + '_' + str(FreqRange) + '_' + str(freq_bin) + 'MHz_' + str(time_bin) + '_fmin' + str(istart_f) + '_tmin' + str(istart_t) + '_Spectra.png')  #
                        dyn_new.plot_dyn(filename=str(HighFreqDir) + 'Spectra_' + str(freq_bin) + 'MHz_' + str(time_bin) + 'min.txt' + '/' + str(Filename) + '_' + str(FreqRange) + '_' + str(freq_bin) + 'MHz_' + str(time_bin) + '_fmin' + str(istart_f) + '_tmin' + str(istart_t) + '_Spectra.png')  #
                        # dyn_new.plot_acf(filename=str(HighFreqDir) + str(Filename) + '/' + str(Filename) + '_' + str(FreqRange) + '_' + str(freq_bin) + 'MHz_' + str(time_bin) + '_fmin' + str(istart_f) + '_tmin' + str(istart_t) + '_ACF.png', fit=True)  #
                        write_results(outfile, dyn=dyn_new)
                    except Exception as e:
                        print(e)
                        continue
            
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
    scintle_num = float_array_from_dict(params, 'scintle_num')
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
    scintle_num = np.array(scintle_num[sort_ind]).squeeze()
    bw = np.array(bw[sort_ind]).squeeze()
    
    """
    Do corrections!
    """
    print()
    print(len(mjd))
    print()
        
    indicies = np.argwhere((tauerr < 0.5*tau) * (dnuerr < 0.5*dnu)) # * (scintle_num > 25))  # * (tau < 1200)) # * (mjd > 58650) * (mjd < 59100) * * (tau < 1200) * (np.sqrt(dnu)/tau < 0.01)* (mjd > mjd_test[z]-9) * (mjd < mjd_test[z]+9) 
    
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
        
    # Make MJD centre of observation, instead of start
    mjd = mjd + tobs/86400/2
    
    print()
    print(len(mjd))
    print()
    # Form Viss from the data
    Aiss = 2.78*10**4  # thin screen, table 2 of Cordes & Rickett (1998)
    D = 1  # kpc
    ind_low = np.argwhere((freq < 1100))

    viss, visserr = scint_velocity(None, dnu, tau, freq, dnuerr, 
                                                  tauerr, a=Aiss)
        
    
    
    mjd_test = []
    for x in range(0, len(mjd)):
        mjd_test.append(round(mjd[x], -1))
    mjd_test = np.unique(mjd_test)
    
    # ind_high = np.argwhere(freq > 1600)
    # ind_mid = np.argwhere((freq > 1100) * (freq < 1600))
    # ind_low = np.argwhere((freq < 1100) * (freq > 950))
    # ind_uhf = np.argwhere(freq < 950)    
    
    # cm = plt.cm.get_cmap('coolwarm')
    # z = freq[ind_uhf]
    # sc = plt.scatter(phase[ind_uhf].flatten(), viss[ind_uhf].flatten(), c=z, cmap=cm, alpha=0.6)
    # plt.colorbar(sc)
    # plt.xlabel('Orbital phase (degrees)')
    # plt.ylabel('Viss (km/s)')
    # plt.xlim(0, 360)
    # plt.title(psrname + ' Scintillation velocity')
    # plt.savefig(plotdir + str(psrname) + "_Viss_OrbitalPhase_UHF.png")
    # plt.show()
    
    # cm = plt.cm.get_cmap('coolwarm')
    # z = freq
    # sc = plt.scatter(phase.flatten(), viss.flatten(), c=z, cmap=cm, alpha=0.6)
    # plt.colorbar(sc)
    # plt.xlabel('Orbital phase (degrees)')
    # plt.ylabel('Viss (km/s)')
    # plt.xlim(0, 360)
    # plt.title(psrname + ' Scintillation velocity')
    # plt.savefig(plotdir + str(psrname) + "_Viss_OrbitalPhase_Freq.png")
    # plt.show()

    # cm = plt.cm.get_cmap('coolwarm')
    # z = freq
    # sc = plt.scatter(phase.flatten(), dnu.flatten(), c=z, cmap=cm, alpha=0.6)
    # plt.colorbar(sc)
    # plt.xlabel('Orbital phase (degrees)')
    # plt.ylabel('Scintillation Bandwidth (MHz)')
    # plt.xlim(0, 360)
    # plt.title(psrname + ' Scintillation velocity')
    # plt.savefig(plotdir + str(psrname) + "_Dnu_OrbitalPhase_Freq.png")
    # plt.show()
    mjd_annual = mjd % 365.2425
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
    phase[phase>360] = phase[phase>360] - 360
         
    Font = 35
    Size = 80*np.pi #Determines the size of the datapoints used
    font = {'size'   : 32}
    matplotlib.rc('font', **font)
    
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(1, 1, 1)
    plt.scatter(phase, viss, c='C0', alpha=0.6, s=Size)
    plt.errorbar(phase, viss, yerr=visserr, fmt=' ', ecolor='C0', alpha=0.4, elinewidth=5)
    plt.xlabel('Orbital phase (degrees)')
    plt.ylabel('Viss (km/s)')
    plt.xlim(0, 360)
    plt.title(psrname + ' Scintillation velocity')
    ax.legend(fontsize='xx-small')
    plt.savefig(plotdir + str(psrname) + "_Viss_OrbitalPhase.png")
    plt.show()
    plt.close()
    
    ind_low_freq = np.argwhere(freq == 1664.77).flatten()
    ind_high_freq = np.argwhere(freq == 1654.74).flatten()
    
    # Orbital Phase #
    
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(1, 1, 1)
    plt.scatter(phase[ind_low_freq], dnu[ind_low_freq], c='C1', alpha=0.6, s=Size)
    plt.errorbar(phase[ind_low_freq], dnu[ind_low_freq], yerr=dnuerr[ind_low_freq], fmt=' ', ecolor='C1', alpha=0.4, elinewidth=5)
    plt.scatter(phase[ind_high_freq], dnu[ind_high_freq], c='C2', alpha=0.6, s=Size)
    plt.errorbar(phase[ind_high_freq], dnu[ind_high_freq], yerr=dnuerr[ind_high_freq], fmt=' ', ecolor='C2', alpha=0.4, elinewidth=5)
    plt.xlabel('Orbital phase (degrees)')
    plt.ylabel('Scintillation Bandwidth (MHz)')
    plt.xlim(0, 360)
    # plt.ylim(0, 2)
    plt.title(psrname + ' Scintillation Bandwidth')
    ax.legend(fontsize='xx-small')
    plt.savefig(plotdir + str(psrname) + "_Viss_OrbitalPhase.png")
    plt.show()
    plt.close()
    
    # MJD #
    
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(1, 1, 1)
    plt.scatter(mjd[ind_low_freq], dnu[ind_low_freq], c='C1', alpha=0.6, s=Size)
    plt.errorbar(mjd[ind_low_freq], dnu[ind_low_freq], yerr=dnuerr[ind_low_freq], fmt=' ', ecolor='C1', alpha=0.4, elinewidth=5)
    plt.scatter(mjd[ind_high_freq], dnu[ind_high_freq], c='C2', alpha=0.6, s=Size)
    plt.errorbar(mjd[ind_high_freq], dnu[ind_high_freq], yerr=dnuerr[ind_high_freq], fmt=' ', ecolor='C2', alpha=0.4, elinewidth=5)
    plt.xlabel('MJD')
    plt.ylabel('Scintillation Bandwidth (MHz)')
    # plt.xlim(0, 360)
    plt.title(psrname + ' Scintillation Bandwidth')
    # plt.ylim(0, 2)
    ax.legend(fontsize='xx-small')
    plt.savefig(plotdir + str(psrname) + "_Dnu_MJD.png")
    plt.show()
    plt.close()

    
    # Theoretical = []
    # for x in range(0, len(dnu)):
    #     Theoretical.append((((float(freq[x]))/(1660.59))**2)*float(dnu[x]))
    # Theoretical = np.asarray(Theoretical)
    # Y = [min(Theoretical),max(Theoretical)]
    # X = [min(freq), max(freq)]
    
    # ind_freq = np.argsort(freq)
    # freq[ind_freq]

    # fig = plt.figure(figsize=(15, 15))
    # ax = fig.add_subplot(1, 1, 1)
    # plt.scatter(freq,dnu, label='Measured dnu', c='C2', s=Size)
    # plt.errorbar(freq,dnu, yerr=dnuerr, fmt=' ', ecolor='C2')
    # # plt.plot(Theoretical,dnu, c='C2')
    # # plt.scatter(Theoretical,dnu_est, label='Estimated dnu', c='C5', s=Size)
    # # plt.errorbar(Theoretical,dnu_est, yerr=dnuerr, fmt=' ', ecolor='C5')
    # # plt.plot(Theoretical,dnu_est, c='C5')
    # plt.scatter(freq,df, label='Channel Bandwidth', c='C4', s=Size)
    # # plt.plot(freq,df, c='C4', label='Channel Bandwidth')
    # # plt.scatter(X,Y, label='Expected Values', c='C3', s=Size)
    # # plt.plot(X,Y, c='C3')
    # ax.legend(fontsize='xx-small')
    # plt.xlabel("Frequency")
    # plt.ylabel("Measured Scint Bandwidth")
    # plt.title("")
    # plt.show()
    # plt.close()
        
    # fig = plt.figure(figsize=(15, 10))
    # ax = fig.add_subplot(1, 1, 1)
    # cm = plt.cm.get_cmap('tab20')
    # z = mjd
    # sc = plt.scatter(phase, dnu, c=z, cmap=cm, alpha=0.6, s=Size)
    # plt.colorbar(sc)
    # plt.xlabel('Orbital phase (degrees)')
    # plt.ylabel('Scintillation Bandwidth (MHz)')
    # plt.xlim(0, 360)
    # plt.ylim(0, 0.7)
    # plt.title(psrname + ' Scintillation Bandwidth')
    # plt.savefig(plotdir + str(psrname) + "_Viss_OrbitalPhase_Freq.png")
    # plt.show()
    
    # fig = plt.figure(figsize=(15, 10))
    # ax = fig.add_subplot(1, 1, 1)
    # cm = plt.cm.get_cmap('tab20')
    # z = mjd
    # sc = plt.scatter(mjd, dnu, c=z, cmap=cm, alpha=0.6, s=Size)
    # plt.colorbar(sc)
    # plt.xlabel('Orbital phase (degrees)')
    # plt.ylabel('MJD')
    # # plt.xlim(0, 360)
    # plt.ylim(0, 0.7)
    # plt.title(psrname + ' Scintillation Bandwidth')
    # plt.savefig(plotdir + str(psrname) + "_Viss_OrbitalPhase_Freq.png")
    # plt.show()


    # mjd_test = []
    # for x in range(0, len(mjd)):
    #     mjd_test.append(round(mjd[x], -1))
    # mjd_test = np.unique(mjd_test)
    
    # ArrayDnu = np.zeros([len(mjd_test),1200])
    # # print(ArrayDnu.shape)
    # for x in range(0, len(dnu)):
    #     for y in range(0, len(mjd_test)):
    #         if mjd[x] < mjd_test[y] + 11 and mjd[x] > mjd_test[y] - 11:
    #             ArrayDnu[y,x] = dnu[x]
    #             # print(dnu[x])
    #             # print("YES")
    # print(ArrayDnu)
    # # ArrayDnu = np.ma.masked_equal(ArrayDnu,0)
    # for x in range(0, ArrayDnu.shape[0]):
    #     print()
    #     ArrayDnu[x,:][ArrayDnu[x,:] != 0]
    # print(ArrayDnu)
    # Dnu = ArrayDnu

    # ind_Dnu = np.argsort(Dnu)

    # plt.scatter(phase[ind_Dnu],Dnu)
    
    plt.hist(dnu, bins=20, alpha=0.4, color='C3')
    yl = plt.ylim()
    plt.plot([df[0], df[0]], yl, 'C2--')
    plt.xlabel("Measured Scint Bandwidth")
    plt.title("High Frequency 'dnu'")
    plt.show()
    plt.close()
    
    plt.hist(scintle_num, bins=20, alpha=0.4, color='C6')
    yl = plt.ylim()
    plt.plot([25, 25], yl, 'C0--')
    plt.xlabel("Number of Estimated Scintles")
    plt.title("High Frequency 'N'")
    plt.show()
    plt.close()
    
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(1, 1, 1)
    plt.scatter(freq, dnu, c='red', alpha=0.6, s=Size)  # , label=r'$f_{c} =$ 1675'
    plt.errorbar(freq, dnu, yerr=dnuerr, fmt=' ', ecolor='red',
                 alpha=0.4, elinewidth=5)
    plt.plot([np.min(freq), np.max(freq)], [df[0], df[0]], c='C0')
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Scintillation Bandwidth (MHz)')
    # plt.xlim(0, 360)
    plt.title(psrname + ' Scintillation Bandwidth')
    # ax.legend(fontsize='xx-small')
    plt.savefig(plotdir + str(psrname) + "Dnu_Frequency_Comparison.png")
    plt.show()
    plt.close()

    
    # for Loop in range(0, len(mjd_test)):    
    
    #     results_dir = outdir
    #     params = read_results(outfile)
        
    #     pars = read_par(str(par_dir) + str(psrname) + '.par')
        
    #     # Read in arrays
    #     mjd = float_array_from_dict(params, 'mjd')  # MJD for observation start
    #     df = float_array_from_dict(params, 'df')  # channel bandwidth
    #     dnu = float_array_from_dict(params, 'dnu')  # scint bandwidth
    #     dnu_est = float_array_from_dict(params, 'dnu_est')  # estimated bandwidth
    #     dnuerr = float_array_from_dict(params, 'dnuerr')
    #     tau = float_array_from_dict(params, 'tau')
    #     tauerr = float_array_from_dict(params, 'tauerr')
    #     freq = float_array_from_dict(params, 'freq')
    #     bw = float_array_from_dict(params, 'bw')
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
    #     bw = np.array(bw[sort_ind]).squeeze()
        
    #     """
    #     Do corrections!
    #     """
    
            
    #     try:
    #         indicies = np.argwhere((mjd > mjd_test[Loop]-9) * (mjd < mjd_test[Loop]+9) * (tauerr < 0.3*tau) * (dnuerr < 0.3*dnu))  # * (tau < 1200)) #(tauerr < 0.2*tau) * (dnuerr < 0.2*dnu) * (dnu<1) * * (tau < 1200) * (np.sqrt(dnu)/tau < 0.01)
            
    #         df = df[indicies].squeeze()
    #         dnu = dnu[indicies].squeeze()
    #         dnu_est = dnu_est[indicies].squeeze()
    #         dnuerr = dnuerr[indicies].squeeze()
    #         tau = tau[indicies].squeeze()
    #         tauerr = tauerr[indicies].squeeze()
    #         mjd = mjd[indicies].squeeze()
    #         rcvrs = rcvrs[indicies].squeeze()
    #         freq = freq[indicies].squeeze()
    #         tobs = tobs[indicies].squeeze()
    #         bw = bw[indicies].squeeze()
            
    #         # Make MJD centre of observation, instead of start
    #         mjd = mjd + tobs/86400/2
            
    #         # Form Viss from the data
    #         Aiss = 2.78*10**4  # thin screen, table 2 of Cordes & Rickett (1998)
    #         D = 1  # kpc
    #         ind_low = np.argwhere((freq < 1100))
        
    #         viss, visserr = scint_velocity(None, dnu, tau, freq, dnuerr, 
    #                                                       tauerr, a=Aiss)
            
    #     except Exception as e:
    #         print(e)
    #         continue

    #     #visserr *= 15
            
    #     # Plotting Begins here #
        
    #     # fig = plt.figure(figsize=(15, 10))
    #     # plt.errorbar(dnu_est, dnu, yerr=dnuerr, fmt='o', alpha=0.8, label='outliers')
    #     # ax = fig.add_subplot(1, 1, 1)
    #     # inds = np.argwhere((dnu < df))
    #     # plt.errorbar(dnu_est[inds], dnu[inds], yerr=dnuerr[inds].squeeze(), fmt='o', alpha=0.8, label='dnu<df')
    #     # plt.grid()
    #     # plt.xlim([0, 1])
    #     # plt.ylim([0, 1.5])
    #     # plt.plot([0, 1], [0, 1], 'k', zorder=3)
    #     # plt.ylabel('Measured scint bandwidth (MHz)')
    #     # plt.xlabel('Estimated scint bandwidth (MHz)')
    #     # ax.legend(fontsize='xx-small')
    #     # plt.savefig(plotdir + str(psrname) + "_EsitmatedScintBandwidth.png")
    #     # plt.show()
        
    #     # plt.figure(figsize=(12,6))
    #     # plt.errorbar(freq, dnu, yerr=dnuerr, fmt='o', alpha=0.8)
    #     # inds = np.argwhere((dnu < df))
    #     # plt.errorbar(freq[inds], dnu[inds], yerr=dnuerr[inds].squeeze(), fmt='o', alpha=0.8)
    #     # plt.yscale('log')
    #     # plt.xscale('log')
    #     # plt.grid()
    #     # plt.ylim([0, 1.5])
    #     # plt.ylabel('Measured scint bandwidth (MHz)')
    #     # plt.xlabel('Observing frequency (MHz)')
    #     # plt.savefig(plotdir + str(psrname) + "_ScintBandwidth.png")
    #     # plt.show()
        
    #     # plt.figure(figsize=(12,6))
    #     # inds = np.argwhere((freq > 1300))
    #     # plt.errorbar(mjd[inds], viss[inds], yerr=visserr[inds].squeeze()*15, fmt='o', )
    #     # inds = np.argwhere((freq < 1300))
    #     # plt.errorbar(mjd[inds], viss[inds], yerr=visserr[inds].squeeze()*15, fmt='o', )
    #     # inds = np.argwhere((dnu < 0.5*df))
    #     # plt.xlabel('MJD')
    #     # plt.ylabel('Viss (km/s)')
    #     # plt.title(psrname)
    #     # plt.savefig(plotdir + str(psrname) + "_Viss_MJD.png")
    #     # plt.show()
            
    #     mjd_annual = mjd % 365.2425
    #     # plt.errorbar(mjd_annual, viss, yerr=visserr, fmt='o', )
    #     # plt.xlabel('Annual phase (arb)')
    #     # plt.ylabel('Viss')
    #     # plt.title(psrname)
    #     # plt.savefig(plotdir + str(psrname) + "_Viss_AnnualPhase.png")
    #     # plt.show()
    #     try:
    #         print('Getting SSB delays')
    #         ssb_delays = get_ssb_delay(mjd, pars['RAJ'], pars['DECJ'])
    #         mjd += np.divide(ssb_delays, 86400)  # add ssb delay
            
    #         """
    #         Model Viss
    #         """
    #         print('Getting Earth velocity')
    #         vearth_ra, vearth_dec = get_earth_velocity(mjd, pars['RAJ'], pars['DECJ'])
    #         print('Getting true anomaly')
    #         pars['PBDOT'] *= 10**-12  # add in factor dropped by tempo
    #         U = get_true_anomaly(mjd, pars)
            
    #         true_anomaly = U.squeeze()
    #         vearth_ra = vearth_ra.squeeze()
    #         vearth_dec = vearth_dec.squeeze()
            
    #         om = pars['OM'] + pars['OMDOT']*(mjd - pars['T0'])/365.2425
    #         # compute orbital phase
    #         phase = U*180/np.pi + om
    #         phase[phase>360] = phase[phase>360] - 360
    #     except Exception as e:
    #         print(e)
    #         continue
        
    #     ind_low_freq = np.argwhere(freq == 1664.77).flatten()
    #     ind_high_freq = np.argwhere(freq == 1654.74).flatten()

        
    #     # Orbital Phase #

    #     fig = plt.figure(figsize=(15, 10))
    #     ax = fig.add_subplot(1, 1, 1)
    #     plt.scatter(phase[ind_low_freq], dnu[ind_low_freq], c='C1', alpha=0.6, s=Size)
    #     plt.errorbar(phase[ind_low_freq], dnu[ind_low_freq], yerr=dnuerr[ind_low_freq], fmt=' ', ecolor='C1', alpha=0.4, elinewidth=5)
    #     plt.scatter(phase[ind_high_freq], dnu[ind_high_freq], c='C2', alpha=0.6, s=Size)
    #     plt.errorbar(phase[ind_high_freq], dnu[ind_high_freq], yerr=dnuerr[ind_high_freq], fmt=' ', ecolor='C2', alpha=0.4, elinewidth=5)
    #     plt.xlabel('Orbital phase (degrees)')
    #     plt.ylabel('Scintillation Bandwidth (MHz)')
    #     plt.xlim(0, 360)
    #     plt.ylim(0, 2)
    #     plt.title(psrname + ' Scintillation Bandwidth')
    #     ax.legend(fontsize='xx-small')
    #     plt.savefig(plotdir + str(psrname) + "_Viss_OrbitalPhase_"+str(Loop)+".png")
    #     plt.show()
    #     plt.close()
        


    