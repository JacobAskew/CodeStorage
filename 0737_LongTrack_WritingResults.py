#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 17:34:31 2021

@author: jacobaskew
"""

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
        float_array_from_dict, get_ssb_delay, get_earth_velocity, \
        get_true_anomaly, scint_velocity, pars_to_params
from scintools.scint_models import effective_velocity_annual
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import glob
from copy import deepcopy as cp
import os
import bilby

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
Spectradir = "/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/LongTrack/Spectra/"
HighFreqSpectradir = "/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/LongTrack/HighFreqSpectra/"
ACFtiltdir = "/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/LongTrack/ACFtilt/"
Dnudir = "/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/LongTrack/Dnu/"
NormTimeSeriesdir = "/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/LongTrack/NormTimeSeries/"
Phasegraddir = "/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/LongTrack/Phasegrad/"
Taudir = "/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/LongTrack/Tau/"
desktopdir = '/Users/jacobaskew/Desktop/'
##############################################################################
# Also Common #
dynspecs = sorted(glob.glob(datadir + '/*ar.dynspec'))
##############################################################################
# Manual Inputs #
freq_bin = 40
time_bin = 10
Nlive = 200
measure = True
model = False
zap = False
linear = False
plot = False
anisotropy = False
# sense = True  # True == Flipped, False == NOT Flipped # MIGHT NEED HELP HELP
Resume = False
outfile = outdir + str(freq_bin) + str(time_bin) + '_LongTrackResult.txt'
##############################################################################


def SearchEclipse(start_mjd, tobs, times):
    end_mjd = start_mjd + tobs/86400
    Eclipse_mjd = np.genfromtxt(eclipsefile, delimiter=',', encoding=None,
                                dtype=float)
    Eclipse_events = np.array(np.where((Eclipse_mjd > start_mjd) *
                              (Eclipse_mjd < end_mjd)))
    print(Eclipse_events.shape[1])
    print(Eclipse_events)
    print(Eclipse_events.shape)
    if Eclipse_events.size == 0:
        Eclipse_index = None
        print("No Eclispe in dynspec")
    elif Eclipse_events.shape[1] == 1:
        Eclipse_events_mjd = Eclipse_mjd[Eclipse_events]
        mjds = start_mjd + times/86400
        Eclipse_index = np.argmin(abs(mjds - Eclipse_events_mjd))
    elif Eclipse_events.shape[1] > 1:
        Eclipse_index = []
        for i in range(0, Eclipse_events.size):
            Eclipse_events_mjd = Eclipse_mjd[Eclipse_events]
            mjds = start_mjd + times/86400
            Eclipse_index.append(np.argmin(abs(mjds - Eclipse_events_mjd[:, i])))
    return Eclipse_index


##############################################################################


def measure_dynspec(observations, FreqFloor, Fmin, Tmin, Fmax, Tmax):
    test = 5

    return test


##############################################################################


def veff_thin_screen_bilby(KIN, d, kappa, KOM, s, vism_ra, vism_dec, efac,
                           equad):
    """
    Effective velocity thin screen model.
    Uses Eq. 4 from Rickett et al. (2014) for anisotropy coefficients.

        ydata: arc curvature

    """

    params_ = dict(params)

    params_['KIN'] = KIN
    params_['d'] = d
    params_['KOM'] = KOM
    params_['kappa'] = kappa
    params_['s'] = s
    params_['vism_ra'] = vism_ra
    params_['vism_dec'] = vism_dec

    veff_ra, veff_dec, vp_ra, vp_dec = \
        effective_velocity_annual(params_, true_anomaly,
                                  vearth_ra, vearth_dec, mjd=mjd)

    veff_ra -= vism_ra
    veff_dec -= vism_dec

    a, b, c = 1, 1, 0

    # coefficient to match model with data
    coeff = 1 / np.sqrt(2 * d * (1 - s) / s)

    veff = kappa * (np.sqrt(a*veff_dec**2 + b*veff_ra**2 +
                            c*veff_ra*veff_dec))
    model = coeff * veff / s

    return model


##############################################################################


def veff_thin_screen_bilby_anisotropy(KIN, d, KOM, s, psi, R, efac,
                                      equad):

    params_ = dict(params)

    params_['KIN'] = KIN
    params_['d'] = d
    params_['KOM'] = KOM
    params_['s'] = s
    params_['psi'] = psi
    params_['R'] = R

    veff_ra, veff_dec, vp_ra, vp_dec = \
        effective_velocity_annual(params_, true_anomaly,
                                  vearth_ra, vearth_dec, mjd=mjd)

    # HELP DANIEL/RYAN HELP
    # psi = params['psi'] * np.pi / 180  # anisotropy angle

    gamma = psi
    cosa = np.cos(2 * gamma)
    sina = np.sin(2 * gamma)

    # quadratic coefficients
    a = (1 - R * cosa) / np.sqrt(1 - R**2)
    b = (1 + R * cosa) / np.sqrt(1 - R**2)
    c = -2 * R * sina / np.sqrt(1 - R**2)

    # coefficient to match model with data
    coeff = 1 / np.sqrt(2 * d * (1 - s) / s)

    veff = (np.sqrt(a*veff_dec**2 + b*veff_ra**2 + c * veff_ra * veff_dec))
    model = coeff * veff / s

    return model


##############################################################################



def modelling(outfile, Name):
    results = read_results(outfile)
    results_dir = outdir
    
    pars = read_par(str(par_dir) + str(psrname) + '.par')

    params = pars_to_params(pars)

    # Read in arrays
    mjd = float_array_from_dict(results, 'mjd')  # MJD for observation start
    df = float_array_from_dict(results, 'df')  # channel bandwidth
    dnu = float_array_from_dict(results, 'dnu')  # scint bandwidth
    dnu_est = float_array_from_dict(results, 'dnu_est')  # estimated bandwidth
    dnuerr = float_array_from_dict(results, 'dnuerr')
    tau = float_array_from_dict(results, 'tau')
    tauerr = float_array_from_dict(results, 'tauerr')
    freq = float_array_from_dict(results, 'freq')
    bw = float_array_from_dict(results, 'bw')
    scintle_num = float_array_from_dict(results, 'scintle_num')
    tobs = float_array_from_dict(results, 'tobs')  # tobs in second
    rcvrs = np.array([rcvr[0] for rcvr in results['name']])
    acf_tilt = float_array_from_dict(results, 'acf_tilt')
    acf_tilt_err = float_array_from_dict(results, 'acf_tilt_err')
    phasegrad = float_array_from_dict(results, 'phasegrad')
    phasegraderr = float_array_from_dict(results, 'phasegraderr')

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
    mjd_min = (mjd*(60*24))
    mjd_min = mjd_min - mjd_min[0]
        
    # Form Viss from the data
    Aiss = 2.78*10**4  # thin screen, table 2 of Cordes & Rickett (1998)
    # D = 1  # kpc
    viss, visserr = scint_velocity(None, dnu, tau, freq, dnuerr, 
                                                  tauerr, a=Aiss)
    
    

    mjd_test = []
    for x in range(0, len(mjd)):
        mjd_test.append(round(mjd[x], -1))
    mjd_test = np.unique(mjd_test)
    
    mjd_annual = mjd % 365.2425
    print('Getting SSB delays')
    ssb_delays = get_ssb_delay(mjd, params['RAJ'].value, params['DECJ'].value)
    mjd += np.divide(ssb_delays, 86400)  # add ssb delay
    
    """
    Model Viss
    """
    print('Getting Earth velocity')
    vearth_ra, vearth_dec = get_earth_velocity(mjd, params['RAJ'].value, params['DECJ'].value)
    print('Getting true anomaly')
    params['PBDOT'].value *= 10**-12  # add in factor dropped by tempo
    U = get_true_anomaly(mjd, params)
    
    true_anomaly = U.squeeze()
    vearth_ra = vearth_ra.squeeze()
    vearth_dec = vearth_dec.squeeze()
    
    om = params['OM'].value + params['OMDOT'].value*(mjd - params['T0'].value)/365.2425
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
    plt.xlabel('Orbital Phase (deg)')
    plt.ylabel('Viss (km/s)')
    plt.xlim(0, 360)
    plt.title(psrname + ' Scintillation velocity')
    ax.legend(fontsize='xx-small')
    plt.savefig(plotdir + Name + "_Viss_OrbitalPhase.png")
    plt.show()
    plt.close()

    # Tau #
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(1, 1, 1)
    plt.scatter(mjd, tau, c='C3', alpha=0.6, s=Size)
    plt.errorbar(mjd, tau, yerr=tauerr, fmt=' ', ecolor='red',
                 alpha=0.4, elinewidth=5)
    plt.xlabel('MJD')
    plt.ylabel('Scintillation Timescale (mins)')
    plt.title(psrname + ' Timescale v Time')
    plt.grid()
    plt.ylim(0, 800)
    plt.savefig(plotdir + str(Name) + "_Tau_TimeSeries.png")
    plt.show()
    plt.close()

    # Dnu #
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
    plt.ylim(0, 8)
    plt.title(psrname + ' Scintillation Bandwidth')
    plt.savefig(plotdir + str(Name) + "_Dnu_TimeSeries.png")
    plt.show()
    plt.close()

    # ACF Tilt #
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(1, 1, 1)
    plt.scatter(mjd, acf_tilt, c='C4', alpha=0.6, s=Size)
    plt.errorbar(mjd, acf_tilt, yerr=acf_tilt_err, fmt=' ',
                 ecolor='C4', alpha=0.4, elinewidth=5)
    plt.xlabel('MJD')
    plt.ylabel('ACF Tilt (?)')
    plt.ylim(-10, 10)
    plt.grid()
    plt.title(psrname + ' ACF Tilt')
    plt.savefig(plotdir + str(Name) + "_Tilt_Timeseries.png")
    plt.show()
    plt.close()

    # Phase Gradient #
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(1, 1, 1)
    plt.scatter(mjd, phasegrad, c='C7', alpha=0.6, s=Size)
    plt.errorbar(mjd, phasegrad, yerr=phasegraderr, fmt=' ',
                 ecolor='C7', alpha=0.4, elinewidth=5)
    plt.xlabel('MJD')
    plt.ylabel('Phase Gradient (?)')
    plt.ylim(-2, 5)
    plt.title(psrname + ' Phase Gradient')
    plt.savefig(str(plotdir) + str(Name) + "_PhaseGrad_Timeseries.png")
    plt.show()
    plt.grid()
    plt.close()
    
    vissscatter = np.sqrt(dnu)/(tau)
    vissscattererr = np.sqrt(dnuerr)/(tauerr)
    norm_vissscatter = (vissscatter-np.min(vissscatter))/(np.max(vissscatter) - np.min(vissscatter))
    norm_inversetau = (1/tau-np.min(1/tau))/(np.max(1/tau) - np.min(1/tau))
    norm_viss = (viss-np.min(viss))/(np.max(viss) - np.min(viss))
    
    constant = viss * tau/(np.sqrt(dnu))
    
    
    # Testing the residuals #
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(1, 1, 1)
    plt.scatter(phase, norm_viss - norm_vissscatter, c='C9', alpha=0.6, s=Size, label='sqrt(dnu)/tau')
    plt.scatter(phase, norm_viss - norm_inversetau, c='C3', alpha=0.6, s=Size, label='1/tau')
    # plt.errorbar(mjd, phasegrad, yerr=phasegraderr, fmt=' ',
    #               ecolor='C7', alpha=0.4, elinewidth=5)
    # plt.errorbar(mjd, phasegrad, yerr=phasegraderr, fmt=' ',
    #               ecolor='C7', alpha=0.4, elinewidth=5)
    plt.xlabel('MJD')
    plt.ylabel('Residuals')
    # plt.ylim(-2, 5)
    plt.title(psrname + ' Scintillation Bandwidth')
    ax.legend(fontsize='xx-small')
    plt.savefig(str(plotdir) + str(Name) + "_ScatterComparison.png")
    plt.show()
    plt.grid()

    # Testing the scatter #
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(1, 1, 1)
    plt.scatter(phase, viss, c='C0', alpha=0.6, s=Size, label='viss')
    plt.scatter(phase, constant/tau, c='C3', alpha=0.6, s=Size, label='1/tau')
    plt.errorbar(phase, viss, yerr=visserr, fmt=' ',
                  ecolor='C0', alpha=0.4, elinewidth=5)
    plt.errorbar(phase, constant/tau, yerr=tau/tauerr, fmt=' ',
                  ecolor='C3', alpha=0.4, elinewidth=5)
    plt.xlabel('Orbital Phase (deg)')
    plt.ylabel('Scintillation Velocity (km/s)')
    # plt.ylim(-2, 5)
    plt.title(psrname + ' Scintillation Velocity')
    ax.legend(fontsize='xx-small')
    plt.savefig(str(plotdir) + str(Name) + "_VissResiduals.png")
    plt.show()
    plt.grid()
    plt.close()
    
    dnu_mean = np.mean(dnu)
    dnu_median = np.median(dnu)
    tau_mean = np.mean(tau)
    tau_median = np.median(tau)
    outfilename = outfile.split(outdir)[1]
    
    # Testing the DNU #
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(1, 1, 1)
    plt.scatter(mjd, dnu, c='C0', alpha=0.7, s=Size, label='DNU')
    plt.plot([np.min(mjd), np.max(mjd)], [dnu_mean, dnu_mean], c='C4', alpha=0.6, label='MEAN')
    plt.plot([np.min(mjd), np.max(mjd)], [dnu_median, dnu_median], c='C5', alpha=0.6, label='MEDIAN')
    plt.plot([np.min(mjd), np.max(mjd)], [df[0], df[0]], c='C2', alpha=0.6, label='CHANNEL BANDWIDTH')
    plt.errorbar(mjd, dnu, yerr=dnuerr, fmt=' ',
                 ecolor='C0', alpha=0.5, elinewidth=5)
    plt.xlabel('MJD')
    plt.ylabel('Scintillation Bandwidth (MHz)')
    plt.title(psrname + ' Scintillation Bandwidth')
    plt.ylim(0, 5)
    ax.legend(fontsize='xx-small')
    plt.savefig(str(desktopdir) + str(outfilename) + "_DnuMEANnMedian.png")
    plt.show()
    plt.grid()
    plt.close()
    
    # Testing the TAU #
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(1, 1, 1)
    plt.scatter(mjd, tau, c='C3', alpha=0.7, s=Size, label='TAU')
    plt.plot([np.min(mjd), np.max(mjd)], [tau_mean, tau_mean], c='C4', alpha=0.6, label='MEAN')
    plt.plot([np.min(mjd), np.max(mjd)], [tau_median, tau_median], c='C5', alpha=0.6, label='MEDIAN')
    plt.errorbar(mjd, tau, yerr=tauerr, fmt=' ',
                 ecolor='C3', alpha=0.5, elinewidth=5)
    plt.xlabel('MJD')
    plt.ylabel('Scintillation Timescale (mins)')
    plt.title(psrname + ' Scintillation Timescale')
    plt.ylim(0, 500)
    ax.legend(fontsize='xx-small')
    plt.savefig(str(desktopdir) + str(outfilename) + "_TauMEANnMedian.png")
    plt.show()
    plt.grid()
    plt.close()
    


##############################################################################

if measure:

    # Group 1 # dyn3 is a loner! # This is not writing results for now
    # dyn1 = Dynspec(filename="/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/RawData/J0737-3039A_2019-04-22-11:43:04_ch5.0_sub5.0.ar.dynspec", process=False)#, lamsteps=True)
    # dyn2 = Dynspec(filename="/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/RawData/J0737-3039A_2019-04-22-12:06:24_ch5.0_sub5.0.ar.dynspec", process=False)#, lamsteps=True)
    # # dyn3 = Dynspec(filename="/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/RawData/J0737-3039A_2019-05-20-17:12:56_ch5.0_sub5.0.ar.dynspec", process=False)

    # Fmax = np.inf
    # Fmin = 0
    # dyn1.crop_dyn(fmin=Fmin, fmax=Fmax)
    # dyn1.refill(linear=False)
    # dyn2.crop_dyn(fmin=Fmin, fmax=Fmax)
    # dyn2.refill(linear=False)

    # dyn_tot = dyn1 + dyn2

    # # Removing any eclispe
    # start_mjd = dyn_tot.mjd
    # tobs = dyn_tot.tobs
    # times = dyn_tot.times
    # Eclipse_index = SearchEclipse(start_mjd, tobs, times)
    # if Eclipse_index is not None and Eclipse_index.size > 1:
    #     for i in range(0, len(Eclipse_index)):
    #         dyn_tot.dyn[:, Eclipse_index[i]-3:Eclipse_index[i]+3] = 0
    # elif Eclipse_index is not None:
    #     dyn_tot.dyn[:, Eclipse_index-3:Eclipse_index+3] = 0
    # # Plotting and Saving the Dynamic Spectra for this group
    # dyn_tot.refill(linear=True)
    # dyn_tot.plot_dyn(filename=str(Spectradir) + str(dyn_tot.name.split('.')[0]) + '_Dynspec.png')

    # # Processing the spectra
    # FreqFloor = 1580
    # time_len = int((round(dyn_tot.tobs/60, 0) - time_bin))
    # max_freq = int(np.max(dyn_tot.freqs))
    # dyn_tot.crop_dyn(fmin=FreqFloor, fmax=np.inf)
    # dyn_tot.plot_dyn(filename=str(HighFreqSpectradir) + str(dyn_tot.name.split('.')[0]) + '_CroppedDynspec')
    # outfile2 = outdir + dyn_tot.name + '.txt'
    # for time_step in range(0, time_len, time_bin):
    #     for freq_step in range(0, max_freq, freq_bin):
    #         try:
    #             dyn = cp(dyn_tot)
    #             dyn.crop_dyn(fmin=FreqFloor + freq_step, fmax=FreqFloor + freq_bin + freq_step, tmin=0+time_step, tmax=10+time_step)
    #             if zap:
    #                 dyn.zap()
    #             if linear:
    #                 dyn.refill(linear=True)
    #             else:
    #                 dyn.refill(linear=False)
    #             if dyn.tobs/60 < time_bin - 5:
    #                 continue
    #             if np.max(dyn.freqs) - np.min(dyn.freqs) < freq_bin - 5:
    #                 continue
    #             dyn.get_acf_tilt(plot=False, display=False)
    #             dyn.get_scint_params(method='acf2d_approx',
    #                                  flux_estimate=True, plot=False,
    #                                  display=False)
    #             dyn.plot_dyn()
    #             # write_results(outfile, dyn=dyn)
    #             # write_results(outfile2, dyn=dyn)

    #         except Exception as e:
    #             print(e)
    #             print("THIS FILE DIDN'T WORK")
    #             continue

    # modelling(outfile=outfile2, Name=dyn_tot.name)

    # # Group 2 # This is not writing results for now
    # dyn1 = Dynspec(filename="/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/RawData/J0737-3039A_2019-07-20-06:22:48_ch5.0_sub5.0.ar.dynspec", process=False)#, lamsteps=True)
    # dyn2 = Dynspec(filename="/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/RawData/J0737-3039A_2019-07-20-06:52:56_ch5.0_sub5.0.ar.dynspec", process=False)#, lamsteps=True)
    # dyn3 = Dynspec(filename="/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/RawData/J0737-3039A_2019-07-20-08:08:08_ch5.0_sub5.0.ar.dynspec", process=False)#, lamsteps=True)
    # dyn4 = Dynspec(filename="/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/RawData/J0737-3039A_2019-07-20-08:38:56_ch5.0_sub5.0.ar.dynspec", process=False)#, lamsteps=True)
    # dyn5 = Dynspec(filename="/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/RawData/J0737-3039A_2019-07-20-09:09:12_ch5.0_sub5.0.ar.dynspec", process=False)#, lamsteps=True)

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
    # dyn4.crop_dyn(fmin=Fmin, fmax=Fmax)
    # # dyn2.zap()
    # dyn4.refill(linear=False)
    # dyn5.crop_dyn(fmin=Fmin, fmax=Fmax)
    # # dyn2.zap()
    # dyn5.refill(linear=False)
    
    # dyn_tot = dyn1 + dyn2 + dyn3 + dyn4 + dyn5
    
    # # Removing any eclispe
    # start_mjd = dyn_tot.mjd
    # tobs = dyn_tot.tobs
    # times = dyn_tot.times
    # Eclipse_index = SearchEclipse(start_mjd, tobs, times)
    # if Eclipse_index is not None and len(Eclipse_index) > 1:
    #     for i in range(0, len(Eclipse_index)):
    #         dyn_tot.dyn[:, Eclipse_index[i]-3:Eclipse_index[i]+3] = 0
    # elif Eclipse_index is not None:
    #     dyn_tot.dyn[:, Eclipse_index-3:Eclipse_index+3] = 0
    # # Plotting and Saving the Dynamic Spectra for this group
    # dyn_tot.refill(linear=True)
    # dyn_tot.plot_dyn(filename=str(Spectradir) + str(dyn_tot.name.split('.')[0]) + '_Dynspec.png')
    
    # # Processing the spectra
    # FreqFloor = 1580
    # time_len = int((round(dyn_tot.tobs/60, 0) - time_bin))
    # max_freq = int(np.max(dyn_tot.freqs))
    # dyn_tot.crop_dyn(fmin=FreqFloor, fmax=np.inf)
    # dyn_tot.plot_dyn(filename=str(HighFreqSpectradir) + str(dyn_tot.name.split('.')[0]) + '_CroppedDynspec')
    # outfile2 = outdir + dyn_tot.name + '.txt'
    # for time_step in range(0, time_len, time_bin):
    #     for freq_step in range(0, max_freq, freq_bin):
    #         try:
    #             dyn = cp(dyn_tot)
    #             dyn.crop_dyn(fmin=FreqFloor + freq_step, fmax=FreqFloor + freq_bin + freq_step, tmin=0+time_step, tmax=10+time_step)
    #             if zap:
    #                 dyn.zap()
    #             if linear:
    #                 dyn.refill(linear=True)
    #             else:
    #                 dyn.refill(linear=False)
    #             if dyn.tobs/60 < time_bin - 5:
    #                 continue
    #             if np.max(dyn.freqs) - np.min(dyn.freqs) < freq_bin - 5:
    #                 continue
    #             dyn.get_acf_tilt(plot=False, display=False)
    #             dyn.get_scint_params(method='acf2d_approx',
    #                                  flux_estimate=True, plot=False,
    #                                  display=False)
    #             dyn.plot_dyn()
    #             # write_results(outfile, dyn=dyn)
    #             # write_results(outfile2, dyn=dyn)
    #         except Exception as e:
    #             print(e)
    #             print("THIS FILE DIDN'T WORK")
    #             continue
    # modelling(outfile=outfile2, Name=dyn_tot.name)
    
    # Group 3 #
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
    
    # Removing any eclispe
    start_mjd = dyn_tot.mjd
    tobs = dyn_tot.tobs
    times = dyn_tot.times
    Eclipse_index = SearchEclipse(start_mjd, tobs, times)
    if Eclipse_index is not None and len(Eclipse_index) > 1:
        for i in range(0, len(Eclipse_index)):
            dyn_tot.dyn[:, Eclipse_index[i]-3:Eclipse_index[i]+3] = 0
    elif Eclipse_index is not None:
        dyn_tot.dyn[:, Eclipse_index-3:Eclipse_index+3] = 0
    # Plotting and Saving the Dynamic Spectra for this group
    dyn_tot.refill(linear=True)
    dyn_tot.plot_dyn(filename=str(Spectradir) + str(dyn_tot.name.split('.')[0]) + '_Dynspec.png')
    
    # Processing the spectra
    FreqFloor = 1580
    time_len = int((round(dyn_tot.tobs/60, 0) - time_bin))
    max_freq = int(np.max(dyn_tot.freqs))
    dyn_tot.crop_dyn(fmin=FreqFloor, fmax=np.inf)
    dyn_tot.plot_dyn(filename=str(HighFreqSpectradir) + str(dyn_tot.name.split('.')[0]) + '_CroppedDynspec')
    outfile2 = outdir + dyn_tot.name + '.txt'
    for time_step in range(0, time_len, time_bin):
        for freq_step in range(0, max_freq, freq_bin):
            try:
                dyn = cp(dyn_tot)
                dyn.crop_dyn(fmin=FreqFloor + freq_step, fmax=FreqFloor + freq_bin + freq_step, tmin=0+time_step, tmax=10+time_step)
                if zap:
                    dyn.zap()
                if linear:
                    dyn.refill(linear=True)
                else:
                    dyn.refill(linear=False)
                if dyn.tobs/60 < time_bin - 5:
                    continue
                if np.max(dyn.freqs) - np.min(dyn.freqs) < freq_bin - 5:
                    continue
                dyn.get_acf_tilt(plot=False, display=False)
                dyn.get_scint_params(method='acf2d_approx',
                                     flux_estimate=True, plot=False,
                                     display=False)
                dyn.plot_dyn()
                write_results(outfile, dyn=dyn)
                write_results(outfile2, dyn=dyn)
            except Exception as e:
                print(e)
                print("THIS FILE DIDN'T WORK")
                continue
    modelling(outfile=outfile2, Name=dyn_tot.name)
    
    # Group 4 #
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
    if Eclipse_index is not None and len(Eclipse_index) > 1:
        for i in range(0, len(Eclipse_index)):
            dyn_tot.dyn[:, Eclipse_index[i]-3:Eclipse_index[i]+3] = 0
    elif Eclipse_index is not None:
        dyn_tot.dyn[:, Eclipse_index-3:Eclipse_index+3] = 0
    # Plotting and Saving the Dynamic Spectra for this group
    dyn_tot.refill(linear=True)
    dyn_tot.plot_dyn(filename=str(Spectradir) + str(dyn_tot.name.split('.')[0]) + '_Dynspec.png')
    
    # Processing the spectra
    FreqFloor = 1580
    time_len = int((round(dyn_tot.tobs/60, 0) - time_bin))
    max_freq = int(np.max(dyn_tot.freqs))
    dyn_tot.crop_dyn(fmin=FreqFloor, fmax=np.inf)
    dyn_tot.plot_dyn(filename=str(HighFreqSpectradir) + str(dyn_tot.name.split('.')[0]) + '_CroppedDynspec')
    outfile2 = outdir + dyn_tot.name + '.txt'
    for time_step in range(0, time_len, time_bin):
        for freq_step in range(0, max_freq, freq_bin):
            try:
                dyn = cp(dyn_tot)
                dyn.crop_dyn(fmin=FreqFloor + freq_step, fmax=FreqFloor + freq_bin + freq_step, tmin=0+time_step, tmax=10+time_step)
                if zap:
                    dyn.zap()
                if linear:
                    dyn.refill(linear=True)
                else:
                    dyn.refill(linear=False)
                if dyn.tobs/60 < time_bin - 5:
                    continue
                if np.max(dyn.freqs) - np.min(dyn.freqs) < freq_bin - 5:
                    continue
                dyn.get_acf_tilt(plot=False, display=False)
                dyn.get_scint_params(method='acf2d_approx',
                                     flux_estimate=True, plot=False,
                                     display=False)
                dyn.plot_dyn()
                write_results(outfile, dyn=dyn)
                write_results(outfile2, dyn=dyn)
            except Exception as e:
                print(e)
                print("THIS FILE DIDN'T WORK")
                continue
    modelling(outfile=outfile2, Name=dyn_tot.name)
    
    # Group 5 #
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
    if Eclipse_index is not None and len(Eclipse_index) > 1:
        for i in range(0, len(Eclipse_index)):
            dyn_tot.dyn[:, Eclipse_index[i]-3:Eclipse_index[i]+3] = 0
    elif Eclipse_index is not None:
        dyn_tot.dyn[:, Eclipse_index-3:Eclipse_index+3] = 0
    # Plotting and Saving the Dynamic Spectra for this group
    dyn_tot.refill(linear=True)
    dyn_tot.plot_dyn(filename=str(Spectradir) + str(dyn_tot.name.split('.')[0]) + '_Dynspec.png')
    
    # Processing the spectra
    FreqFloor = 1580
    time_len = int((round(dyn_tot.tobs/60, 0) - time_bin))
    max_freq = int(np.max(dyn_tot.freqs))
    dyn_tot.crop_dyn(fmin=FreqFloor, fmax=np.inf)
    dyn_tot.plot_dyn(filename=str(HighFreqSpectradir) + str(dyn_tot.name.split('.')[0]) + '_CroppedDynspec')
    outfile2 = outdir + dyn_tot.name + '.txt'
    for time_step in range(0, time_len, time_bin):
        for freq_step in range(0, max_freq, freq_bin):
            try:
                dyn = cp(dyn_tot)
                dyn.crop_dyn(fmin=FreqFloor + freq_step, fmax=FreqFloor + freq_bin + freq_step, tmin=0+time_step, tmax=10+time_step)
                if zap:
                    dyn.zap()
                if linear:
                    dyn.refill(linear=True)
                else:
                    dyn.refill(linear=False)
                if dyn.tobs/60 < time_bin - 5:
                    continue
                if np.max(dyn.freqs) - np.min(dyn.freqs) < freq_bin - 5:
                    continue
                dyn.get_acf_tilt(plot=False, display=False)
                dyn.get_scint_params(method='acf2d_approx',
                                     flux_estimate=True, plot=False,
                                     display=False)
                dyn.plot_dyn()
                write_results(outfile, dyn=dyn)
                write_results(outfile2, dyn=dyn)
            except Exception as e:
                print(e)
                print("THIS FILE DIDN'T WORK")
                continue
    modelling(outfile=outfile2, Name=dyn_tot.name)
    
    # Group 6 #
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
    if Eclipse_index is not None and len(Eclipse_index) > 1:
        for i in range(0, len(Eclipse_index)):
            dyn_tot.dyn[:, Eclipse_index[i]-3:Eclipse_index[i]+3] = 0
    elif Eclipse_index is not None:
        dyn_tot.dyn[:, Eclipse_index-3:Eclipse_index+3] = 0
    # Plotting and Saving the Dynamic Spectra for this group
    dyn_tot.refill(linear=True)
    dyn_tot.plot_dyn(filename=str(Spectradir) + str(dyn_tot.name.split('.')[0]) + '_Dynspec.png')
    
    # Processing the spectra
    FreqFloor = 1580
    time_len = int((round(dyn_tot.tobs/60, 0) - time_bin))
    max_freq = int(np.max(dyn_tot.freqs))
    dyn_tot.crop_dyn(fmin=FreqFloor, fmax=np.inf)
    dyn_tot.plot_dyn(filename=str(HighFreqSpectradir) + str(dyn_tot.name.split('.')[0]) + '_CroppedDynspec')
    outfile2 = outdir + dyn_tot.name + '.txt'
    for time_step in range(0, time_len, time_bin):
        for freq_step in range(0, max_freq, freq_bin):
            try:
                dyn = cp(dyn_tot)
                dyn.crop_dyn(fmin=FreqFloor + freq_step, fmax=FreqFloor + freq_bin + freq_step, tmin=0+time_step, tmax=10+time_step)
                if zap:
                    dyn.zap()
                if linear:
                    dyn.refill(linear=True)
                else:
                    dyn.refill(linear=False)
                if dyn.tobs/60 < time_bin - 5:
                    continue
                if np.max(dyn.freqs) - np.min(dyn.freqs) < freq_bin - 5:
                    continue
                dyn.get_acf_tilt(plot=False, display=False)
                dyn.get_scint_params(method='acf2d_approx',
                                     flux_estimate=True, plot=False,
                                     display=False)
                dyn.plot_dyn()
                write_results(outfile, dyn=dyn)
                write_results(outfile2, dyn=dyn)
            except Exception as e:
                print(e)
                print("THIS FILE DIDN'T WORK")
                continue
    modelling(outfile=outfile2, Name=dyn_tot.name)
    
    # Group 7 #
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
    
    # Removing any eclispe
    start_mjd = dyn_tot.mjd
    tobs = dyn_tot.tobs
    times = dyn_tot.times
    Eclipse_index = SearchEclipse(start_mjd, tobs, times)
    if Eclipse_index is not None and len(Eclipse_index) > 1:
        for i in range(0, len(Eclipse_index)):
            dyn_tot.dyn[:, Eclipse_index[i]-3:Eclipse_index[i]+3] = 0
    elif Eclipse_index is not None:
        dyn_tot.dyn[:, Eclipse_index-3:Eclipse_index+3] = 0
    # Plotting and Saving the Dynamic Spectra for this group
    dyn_tot.refill(linear=True)
    dyn_tot.plot_dyn(filename=str(Spectradir) + str(dyn_tot.name.split('.')[0]) + '_Dynspec.png')
    
    # Processing the spectra
    FreqFloor = 1580
    time_len = int((round(dyn_tot.tobs/60, 0) - time_bin))
    max_freq = int(np.max(dyn_tot.freqs))
    dyn_tot.crop_dyn(fmin=FreqFloor, fmax=np.inf)
    dyn_tot.plot_dyn(filename=str(HighFreqSpectradir) + str(dyn_tot.name.split('.')[0]) + '_CroppedDynspec')
    outfile2 = outdir + dyn_tot.name + '.txt'
    for time_step in range(0, time_len, time_bin):
        for freq_step in range(0, max_freq, freq_bin):
            try:
                dyn = cp(dyn_tot)
                dyn.crop_dyn(fmin=FreqFloor + freq_step, fmax=FreqFloor + freq_bin + freq_step, tmin=0+time_step, tmax=10+time_step)
                if zap:
                    dyn.zap()
                if linear:
                    dyn.refill(linear=True)
                else:
                    dyn.refill(linear=False)
                if dyn.tobs/60 < time_bin - 5:
                    continue
                if np.max(dyn.freqs) - np.min(dyn.freqs) < freq_bin - 5:
                    continue
                dyn.get_acf_tilt(plot=False, display=False)
                dyn.get_scint_params(method='acf2d_approx',
                                     flux_estimate=True, plot=False,
                                     display=False)
                dyn.plot_dyn()
                write_results(outfile, dyn=dyn)
                write_results(outfile2, dyn=dyn)
            except Exception as e:
                print(e)
                print("THIS FILE DIDN'T WORK")
                continue
    modelling(outfile=outfile2, Name=dyn_tot.name)
    
    # Group 8 #
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
    if Eclipse_index is not None and len(Eclipse_index) > 1:
        for i in range(0, len(Eclipse_index)):
            dyn_tot.dyn[:, Eclipse_index[i]-3:Eclipse_index[i]+3] = 0
    elif Eclipse_index is not None:
        dyn_tot.dyn[:, Eclipse_index-3:Eclipse_index+3] = 0
    # Plotting and Saving the Dynamic Spectra for this group
    dyn_tot.refill(linear=True)
    dyn_tot.plot_dyn(filename=str(Spectradir) + str(dyn_tot.name.split('.')[0]) + '_Dynspec.png')
    
    # Processing the spectra
    FreqFloor = 1580
    time_len = int((round(dyn_tot.tobs/60, 0) - time_bin))
    max_freq = int(np.max(dyn_tot.freqs))
    dyn_tot.crop_dyn(fmin=FreqFloor, fmax=np.inf)
    dyn_tot.plot_dyn(filename=str(HighFreqSpectradir) + str(dyn_tot.name.split('.')[0]) + '_CroppedDynspec')
    outfile2 = outdir + dyn_tot.name + '.txt'
    for time_step in range(0, time_len, time_bin):
        for freq_step in range(0, max_freq, freq_bin):
            try:
                dyn = cp(dyn_tot)
                dyn.crop_dyn(fmin=FreqFloor + freq_step, fmax=FreqFloor + freq_bin + freq_step, tmin=0+time_step, tmax=10+time_step)
                if zap:
                    dyn.zap()
                if linear:
                    dyn.refill(linear=True)
                else:
                    dyn.refill(linear=False)
                if dyn.tobs/60 < time_bin - 5:
                    continue
                if np.max(dyn.freqs) - np.min(dyn.freqs) < freq_bin - 5:
                    continue
                dyn.get_acf_tilt(plot=False, display=False)
                dyn.get_scint_params(method='acf2d_approx',
                                     flux_estimate=True, plot=False,
                                     display=False)
                dyn.plot_dyn()
                write_results(outfile, dyn=dyn)
                write_results(outfile2, dyn=dyn)
            except Exception as e:
                print(e)
                print("THIS FILE DIDN'T WORK")
                continue
    modelling(outfile=outfile2, Name=dyn_tot.name)
    
    # Group 9 # dyn1 is a loner!
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
    if Eclipse_index is not None and len(Eclipse_index) > 1:
        for i in range(0, len(Eclipse_index)):
            dyn_tot.dyn[:, Eclipse_index[i]-3:Eclipse_index[i]+3] = 0
    elif Eclipse_index is not None:
        dyn_tot.dyn[:, Eclipse_index-3:Eclipse_index+3] = 0
    # Plotting and Saving the Dynamic Spectra for this group
    dyn_tot.refill(linear=True)
    dyn_tot.plot_dyn(filename=str(Spectradir) + str(dyn_tot.name.split('.')[0]) + '_Dynspec.png')
    
    # Processing the spectra
    FreqFloor = 1580
    time_len = int((round(dyn_tot.tobs/60, 0) - time_bin))
    max_freq = int(np.max(dyn_tot.freqs))
    dyn_tot.crop_dyn(fmin=FreqFloor, fmax=np.inf)
    dyn_tot.plot_dyn(filename=str(HighFreqSpectradir) + str(dyn_tot.name.split('.')[0]) + '_CroppedDynspec')
    outfile2 = outdir + dyn_tot.name + '.txt'
    for time_step in range(0, time_len, time_bin):
        for freq_step in range(0, max_freq, freq_bin):
            try:
                dyn = cp(dyn_tot)
                dyn.crop_dyn(fmin=FreqFloor + freq_step, fmax=FreqFloor + freq_bin + freq_step, tmin=0+time_step, tmax=10+time_step)
                if zap:
                    dyn.zap()
                if linear:
                    dyn.refill(linear=True)
                else:
                    dyn.refill(linear=False)
                if dyn.tobs/60 < time_bin - 5:
                    continue
                if np.max(dyn.freqs) - np.min(dyn.freqs) < freq_bin - 5:
                    continue
                dyn.get_acf_tilt(plot=False, display=False)
                dyn.get_scint_params(method='acf2d_approx',
                                     flux_estimate=True, plot=False,
                                     display=False)
                dyn.plot_dyn()
                write_results(outfile, dyn=dyn)
                write_results(outfile2, dyn=dyn)
            except Exception as e:
                print(e)
                print("THIS FILE DIDN'T WORK")
                continue
    modelling(outfile=outfile2, Name=dyn_tot.name)
    
    # Group 10 # dyn3 is a loner!
    dyn1 = Dynspec(filename="/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/RawData/J0737-3039A_2020-02-21-19:42:48_ch5.0_sub5.0.ar.dynspec", process=False)#, lamsteps=True)
    dyn2 = Dynspec(filename="/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/RawData/J0737-3039A_2020-02-21-20:13:12_ch5.0_sub5.0.ar.dynspec", process=False)#, lamsteps=True)
    # dyn3 = Dynspec(filename="/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/RawData/J0737-3039A_2020-03-20-20:18:32_ch5.0_sub5.0.ar.dynspec", process=False)#, lamsteps=True)
    
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
    if Eclipse_index is not None and Eclipse_index.size > 1:
        for i in range(0, len(Eclipse_index)):
            dyn_tot.dyn[:, Eclipse_index[i]-3:Eclipse_index[i]+3] = 0
    elif Eclipse_index is not None:
        dyn_tot.dyn[:, Eclipse_index-3:Eclipse_index+3] = 0
    # Plotting and Saving the Dynamic Spectra for this group
    dyn_tot.refill(linear=True)
    dyn_tot.plot_dyn(filename=str(Spectradir) + str(dyn_tot.name.split('.')[0]) + '_Dynspec.png')
    
    # Processing the spectra
    FreqFloor = 1580
    time_len = int((round(dyn_tot.tobs/60, 0) - time_bin))
    max_freq = int(np.max(dyn_tot.freqs))
    dyn_tot.crop_dyn(fmin=FreqFloor, fmax=np.inf)
    dyn_tot.plot_dyn(filename=str(HighFreqSpectradir) + str(dyn_tot.name.split('.')[0]) + '_CroppedDynspec')
    outfile2 = outdir + dyn_tot.name + '.txt'
    for time_step in range(0, time_len, time_bin):
        for freq_step in range(0, max_freq, freq_bin):
            try:
                dyn = cp(dyn_tot)
                dyn.crop_dyn(fmin=FreqFloor + freq_step, fmax=FreqFloor + freq_bin + freq_step, tmin=0+time_step, tmax=10+time_step)
                if zap:
                    dyn.zap()
                if linear:
                    dyn.refill(linear=True)
                else:
                    dyn.refill(linear=False)
                if dyn.tobs/60 < time_bin - 5:
                    continue
                if np.max(dyn.freqs) - np.min(dyn.freqs) < freq_bin - 5:
                    continue
                dyn.get_acf_tilt(plot=False, display=False)
                dyn.get_scint_params(method='acf2d_approx',
                                     flux_estimate=True, plot=False,
                                     display=False)
                dyn.plot_dyn()
                write_results(outfile, dyn=dyn)
                write_results(outfile2, dyn=dyn)
            except Exception as e:
                print(e)
                print("THIS FILE DIDN'T WORK")
                continue
    modelling(outfile=outfile2, Name=dyn_tot.name)
    
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
    if Eclipse_index is not None and len(Eclipse_index) > 1:
        for i in range(0, len(Eclipse_index)):
            dyn_tot.dyn[:, Eclipse_index[i]-3:Eclipse_index[i]+3] = 0
    elif Eclipse_index is not None:
        dyn_tot.dyn[:, Eclipse_index-3:Eclipse_index+3] = 0
    # Plotting and Saving the Dynamic Spectra for this group
    dyn_tot.refill(linear=True)
    dyn_tot.plot_dyn(filename=str(Spectradir) + str(dyn_tot.name.split('.')[0]) + '_Dynspec.png')
    
    # Processing the spectra
    FreqFloor = 1580
    time_len = int((round(dyn_tot.tobs/60, 0) - time_bin))
    max_freq = int(np.max(dyn_tot.freqs))
    dyn_tot.crop_dyn(fmin=FreqFloor, fmax=np.inf)
    dyn_tot.plot_dyn(filename=str(HighFreqSpectradir) + str(dyn_tot.name.split('.')[0]) + '_CroppedDynspec')
    outfile2 = outdir + dyn_tot.name + '.txt'
    for time_step in range(0, time_len, time_bin):
        for freq_step in range(0, max_freq, freq_bin):
            try:
                dyn = cp(dyn_tot)
                dyn.crop_dyn(fmin=FreqFloor + freq_step, fmax=FreqFloor + freq_bin + freq_step, tmin=0+time_step, tmax=10+time_step)
                if zap:
                    dyn.zap()
                if linear:
                    dyn.refill(linear=True)
                else:
                    dyn.refill(linear=False)
                if dyn.tobs/60 < time_bin - 5:
                    continue
                if np.max(dyn.freqs) - np.min(dyn.freqs) < freq_bin - 5:
                    continue
                dyn.get_acf_tilt(plot=False, display=False)
                dyn.get_scint_params(method='acf2d_approx',
                                     flux_estimate=True, plot=False,
                                     display=False)
                dyn.plot_dyn()
                write_results(outfile, dyn=dyn)
                write_results(outfile2, dyn=dyn)

            except Exception as e:
                print(e)
                print("THIS FILE DIDN'T WORK")
                continue
    modelling(outfile=outfile2, Name=dyn_tot.name)
    
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
    if Eclipse_index is not None and len(Eclipse_index) > 1:
        for i in range(0, len(Eclipse_index)):
            dyn_tot.dyn[:, Eclipse_index[i]-3:Eclipse_index[i]+3] = 0
    elif Eclipse_index is not None:
        dyn_tot.dyn[:, Eclipse_index-3:Eclipse_index+3] = 0
    # Plotting and Saving the Dynamic Spectra for this group
    dyn_tot.refill(linear=True)
    dyn_tot.plot_dyn(filename=str(Spectradir) + str(dyn_tot.name.split('.')[0]) + '_Dynspec.png')
    
    # Processing the spectra
    time_len = int((round(dyn_tot.tobs/60, 0) - time_bin))
    max_freq = int(np.max(dyn_tot.freqs))
    FreqFloor = 1580
    dyn_tot.crop_dyn(fmin=FreqFloor, fmax=np.inf)
    dyn_tot.plot_dyn(filename=str(HighFreqSpectradir) + str(dyn_tot.name.split('.')[0]) + '_CroppedDynspec')
    outfile2 = outdir + dyn_tot.name + '.txt'
    for time_step in range(0, time_len, time_bin):
        for freq_step in range(0, max_freq, freq_bin):
            try:
                dyn = cp(dyn_tot)
                dyn.crop_dyn(fmin=FreqFloor + freq_step, fmax=FreqFloor + freq_bin + freq_step, tmin=0+time_step, tmax=10+time_step)
                if zap:
                    dyn.zap()
                if linear:
                    dyn.refill(linear=True)
                else:
                    dyn.refill(linear=False)
                if dyn.tobs/60 < time_bin - 5:
                    continue
                if np.max(dyn.freqs) - np.min(dyn.freqs) < freq_bin - 5:
                    continue
                dyn.get_acf_tilt(plot=False, display=False)
                dyn.get_scint_params(method='acf2d_approx',
                                     flux_estimate=True, plot=False,
                                     display=False)
                dyn.plot_dyn()
                write_results(outfile, dyn=dyn)
                write_results(outfile2, dyn=dyn)
            except Exception as e:
                print(e)
                print("THIS FILE DIDN'T WORK")
                continue
    modelling(outfile=outfile2, Name=dyn_tot.name)

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


if model:

    results = read_results(outfile)
    results_dir = outdir

    pars = read_par(str(par_dir) + str(psrname) + '.par')
    params = pars_to_params(pars)


    # Read in arrays
    mjd = float_array_from_dict(results, 'mjd')  # MJD for observation start
    df = float_array_from_dict(results, 'df')  # channel bandwidth
    dnu = float_array_from_dict(results, 'dnu')  # scint bandwidth
    dnu_est = float_array_from_dict(results, 'dnu_est')  # estimated bandwidth
    dnuerr = float_array_from_dict(results, 'dnuerr')
    tau = float_array_from_dict(results, 'tau')
    tauerr = float_array_from_dict(results, 'tauerr')
    freq = float_array_from_dict(results, 'freq')
    bw = float_array_from_dict(results, 'bw')
    scintle_num = float_array_from_dict(results, 'scintle_num')
    tobs = float_array_from_dict(results, 'tobs')  # tobs in second
    rcvrs = np.array([rcvr[0] for rcvr in results['name']])
    acf_tilt = float_array_from_dict(results, 'acf_tilt')
    acf_tilt_err = float_array_from_dict(results, 'acf_tilt_err')
    phasegrad = float_array_from_dict(results, 'phasegrad')
    phasegraderr = float_array_from_dict(results, 'phasegraderr')

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
    mjd_min = (mjd*(60*24))
    mjd_min = mjd_min - mjd_min[0]

    # Form Viss from the data
    Aiss = 2.78*10**4  # thin screen, table 2 of Cordes & Rickett (1998)
    # D = 1  # kpc
    viss, visserr = scint_velocity(None, dnu, tau, freq, dnuerr,
                                                  tauerr, a=Aiss)

    mjd_test = []
    for x in range(0, len(mjd)):
        mjd_test.append(round(mjd[x], -1))
    mjd_test = np.unique(mjd_test)

    mjd_annual = mjd % 365.2425
    print('Getting SSB delays')
    ssb_delays = get_ssb_delay(mjd, params['RAJ'].value, params['DECJ'].value)
    mjd += np.divide(ssb_delays, 86400)  # add ssb delay

    """
    Model Viss
    """

    print('Getting Earth velocity')
    vearth_ra, vearth_dec = \
        get_earth_velocity(mjd, params['RAJ'].value, params['DECJ'].value)
    print('Getting true anomaly')
    params['PBDOT'].value *= 10**-12  # add in factor dropped by tempo
    U = get_true_anomaly(mjd, params)
    
    true_anomaly = U.squeeze()
    vearth_ra = vearth_ra.squeeze()
    vearth_dec = vearth_dec.squeeze()
    
    om = params['OM'].value + \
        params['OMDOT'].value*(mjd - params['T0'].value)/365.2425
    # compute orbital phase
    phase = U*180/np.pi + om
    phase[phase>360] = phase[phase>360] - 360

##############################################################################
    # Beggining Fitting #

    """
    Fitting routine
    """

    # A few simple setup steps
    label = '_Scintillation'

    if anisotropy:
        label += '_Anisotropy'
    outdir = plotdir

    bilby.utils.check_directory_exists_and_if_not_mkdir(outdir)
    injection_parameters = None
    if not anisotropy:
        likelihood = bilby.likelihood.GaussianLikelihood(x=mjd,
                                                         y=viss,
                                                         func=veff_thin_screen_bilby,
                                                         sigma=visserr)
    if anisotropy:
        likelihood = bilby.likelihood.GaussianLikelihood(x=mjd,
                                                         y=viss,
                                                         func=veff_thin_screen_bilby_anisotropy,
                                                         sigma=visserr)
    if not anisotropy:
        priors = dict(KIN=bilby.core.prior.Uniform(0, 180, 'KIN',
                                                   boundary='periodic'),
                      d=bilby.core.prior.Uniform(0, 2, 'd'),
                      kappa=bilby.core.prior.Uniform(0, 1, 'kappa'),
                      KOM=bilby.core.prior.Uniform(0, 360, 'KOM',
                                                   boundary='periodic'),
                      s=bilby.core.prior.Uniform(0, 1, 's'),
                      vism_ra=bilby.core.prior.Uniform(-200, 200, 'vism_ra'),
                      vism_dec=bilby.core.prior.Uniform(-200, 200, 'vism_dec'),
                      efac=bilby.core.prior.Uniform(-2, 2, 'efac'),
                      equad=bilby.core.prior.Uniform(-2, 2, 'equad'))
    if anisotropy:
        priors = dict(KIN=bilby.core.prior.Uniform(0, 180, 'KIN',
                                                   boundary='periodic'),
                      d=bilby.core.prior.Uniform(0, 2, 'd'),
                      KOM=bilby.core.prior.Uniform(0, 360, 'KOM',
                                                   boundary='periodic'),
                      s=bilby.core.prior.Uniform(0, 1, 's'),
                      psi=bilby.core.prior.Uniform(0, 180, 'psi',
                                                   boundary='periodic'),
                      R=bilby.core.prior.Uniform(0, 1, 'R'),
                      efac=bilby.core.prior.Uniform(-2, 2, 'efac'),
                      equad=bilby.core.prior.Uniform(-2, 2, 'equad'))

    print("... Running the sampler ...")

    # And run sampler
    result = bilby.core.sampler.run_sampler(
            likelihood, priors=priors, sampler='dynesty', label=label,
            nlive=Nlive, verbose=True, resume=Resume,
            outdir=outdir)

    NUMBER = np.argmax(result.posterior['log_likelihood'].values)
    params.add('KIN', value=result.posterior['KIN'][NUMBER], vary=False)
    params.add('d', value=result.posterior['d'][NUMBER], vary=False)
    params.add('KOM', value=result.posterior['KOM'][NUMBER], vary=False)
    params.add('s', value=result.posterior['s'][NUMBER], vary=False)
    if not anisotropy:
        params.add('vism_ra', value=result.posterior['vism_ra'][NUMBER],
                   vary=False)
        params.add('vism_dec', value=result.posterior['vism_dec'][NUMBER],
                   vary=False)
    if anisotropy:
        params.add('psi', value=result.posterior['psi'][NUMBER], vary=False)
        params.add('R', value=result.posterior['R'][NUMBER], vary=False)
    params.add('efac', value=result.posterior['efac'][NUMBER], vary=False)
    params.add('equad', value=result.posterior['equad'][NUMBER], vary=False)
###############################################################################
    # This is gorgeous DO NOT DELETE
    # res = arc_curvature(params, eta, 1/etaerr, U, vearth_ra, vearth_dec)
    # model = eta - res*etaerr

    # if BinarySystem == 1:
    #     test = cp(params)
    #     test["A1"].value = 0
    #     res_nobinary = arc_curvature(test, eta, 1/etaerr, U, vearth_ra,
    # vearth_dec)
    #     model_nobinary = eta - res_nobinary*etaerr

    #     vbinary = model - model_nobinary
    # else:
    #     pars['A1'] = 0

    # res_nobinary = arc_curvature(params, eta, 1/etaerr, U, vearth_ra,
    # vearth_dec)
    # model_nobinary = eta - res_nobinary*etaerr

    # vbinary = model - model_nobinary

    # res_noearth = arc_curvature(params, eta, 1/etaerr, U,
    #                             np.zeros(np.shape(vearth_ra)),
    #                             np.zeros(np.shape(vearth_dec)))
    # model_noearth = eta - res_noearth*etaerr

    # vearth = model - model_noearth

    # mjd_annual = mjd % 365.2425

    # sort_mjd = np.argsort(mjd)
    # sort_annual = np.argsort(mjd_annual)
    # sort_U = np.argsort(U)
##############################################################################
# Plotting #

if plot:

    font = {'size': 16}
    matplotlib.rc('font', **font)
    result.plot_corner()

    Font = 35
    Size = 80*np.pi #Determines the size of the datapoints used
    font = {'size'   : 32}
    matplotlib.rc('font', **font)

    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(1, 1, 1)
    plt.scatter(phase, viss, c='C0', alpha=0.6, s=Size)
    plt.errorbar(phase, viss, yerr=visserr, fmt=' ', ecolor='C0', alpha=0.4, elinewidth=5)
    plt.xlabel('Orbital Phase (deg)')
    plt.ylabel('Viss (km/s)')
    plt.xlim(0, 360)
    plt.title(psrname + ' Scintillation velocity')
    ax.legend(fontsize='xx-small')
    plt.savefig(plotdir + str(psrname) + "_Viss_OrbitalPhase.png")
    plt.show()
    plt.close()

    # Tau #
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(1, 1, 1)
    plt.scatter(mjd, tau, c='C3', alpha=0.6, s=Size)
    plt.errorbar(mjd, tau, yerr=tauerr, fmt=' ', ecolor='red',
                 alpha=0.4, elinewidth=5)
    plt.xlabel('MJD')
    plt.ylabel('Scintillation Timescale (mins)')
    plt.title(psrname + ' Timescale v Time')
    plt.grid()
    plt.ylim(0, 800)
    plt.savefig(str(plotdir) + "Tau_TimeSeries.png")
    plt.show()
    plt.close()

    # Dnu #
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
    plt.ylim(0, 8)
    plt.title(psrname + ' Scintillation Bandwidth')
    plt.savefig(str(plotdir) + "Dnu_TimeSeries.png")
    plt.show()
    plt.close()

    # ACF Tilt #
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(1, 1, 1)
    plt.scatter(mjd, acf_tilt, c='C4', alpha=0.6, s=Size)
    plt.errorbar(mjd, acf_tilt, yerr=acf_tilt_err, fmt=' ',
                 ecolor='C4', alpha=0.4, elinewidth=5)
    plt.xlabel('MJD')
    plt.ylabel('ACF Tilt (?)')
    plt.ylim(-10, 10)
    plt.grid()
    plt.title(psrname + ' Scintillation Bandwidth')
    plt.savefig(str(plotdir) + "Tilt_Timeseries.png")
    plt.show()
    plt.close()

    # Phase Gradient #
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(1, 1, 1)
    plt.scatter(mjd, phasegrad, c='C7', alpha=0.6, s=Size)
    plt.errorbar(mjd, phasegrad, yerr=phasegraderr, fmt=' ',
                 ecolor='C7', alpha=0.4, elinewidth=5)
    plt.xlabel('MJD')
    plt.ylabel('Phase Gradient (?)')
    plt.ylim(-2, 5)
    plt.title(psrname + ' Scintillation Bandwidth')
    plt.savefig(str(plotdir)  + "Tilt_Timeseries.png")
    plt.show()
    plt.grid()
    plt.close()
    
    vissscatter = np.sqrt(dnu)/(tau)
    vissscattererr = np.sqrt(dnuerr)/(tauerr)
    norm_vissscatter = (vissscatter-np.min(vissscatter))/(np.max(vissscatter) - np.min(vissscatter))
    norm_inversetau = (1/tau-np.min(1/tau))/(np.max(1/tau) - np.min(1/tau))
    norm_viss = (viss-np.min(viss))/(np.max(viss) - np.min(viss))
    
    constant = viss * tau/(np.sqrt(dnu))
    
    
    # Testing the residuals #
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(1, 1, 1)
    plt.scatter(phase, norm_viss - norm_vissscatter, c='C9', alpha=0.6, s=Size, label='sqrt(dnu)/tau')
    plt.scatter(phase, norm_viss - norm_inversetau, c='C3', alpha=0.6, s=Size, label='1/tau')
    # plt.errorbar(mjd, phasegrad, yerr=phasegraderr, fmt=' ',
    #               ecolor='C7', alpha=0.4, elinewidth=5)
    # plt.errorbar(mjd, phasegrad, yerr=phasegraderr, fmt=' ',
    #               ecolor='C7', alpha=0.4, elinewidth=5)
    plt.xlabel('MJD')
    plt.ylabel('Residuals')
    # plt.ylim(-2, 5)
    plt.title(psrname + ' Scintillation Bandwidth')
    ax.legend(fontsize='xx-small')
    plt.savefig(str(plotdir)  + "Tilt_Timeseries.png")
    plt.show()
    plt.grid()

    # Testing the scatter #
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(1, 1, 1)
    plt.scatter(phase, viss, c='C0', alpha=0.6, s=Size, label='viss')
    plt.scatter(phase, constant/tau, c='C3', alpha=0.6, s=Size, label='1/tau')
    plt.errorbar(phase, viss, yerr=visserr, fmt=' ',
                  ecolor='C0', alpha=0.4, elinewidth=5, )
    plt.errorbar(phase, constant/tau, yerr=tau/tauerr, fmt=' ',
                  ecolor='C3', alpha=0.4, elinewidth=5)
    plt.xlabel('Orbital Phase (deg)')
    plt.ylabel('Scintillation Velocity (km/s)')
    # plt.ylim(-2, 5)
    plt.title(psrname + ' Scintillation Velocity')
    ax.legend(fontsize='xx-small')
    plt.savefig(str(plotdir)  + "Tilt_Timeseries.png")
    plt.show()
    plt.grid()
    plt.close()

###############################################################################

# dyn1 = Dynspec(filename="/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/RawData/J0737-3039A_2020-12-22-02:45:28_ch5.0_sub5.0.ar.dynspec", process=False)#, lamsteps=True)
# dyn2 = Dynspec(filename="/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/RawData/J0737-3039A_2020-12-22-03:15:28_ch5.0_sub5.0.ar.dynspec", process=False)#, lamsteps=True)
# dyn3 = Dynspec(filename="/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/RawData/J0737-3039A_2020-12-22-05:15:36_ch5.0_sub5.0.ar.dynspec", process=False)#, lamsteps=True)

# Fmax = 1700
# Fmin = 875
# dyn1.trim_edges()
# dyn1.crop_dyn(fmin=Fmin, fmax=Fmax)
# dyn1.refill(linear=True)
# dyn2.trim_edges()
# dyn2.crop_dyn(fmin=Fmin, fmax=Fmax)
# dyn2.refill(linear=False)
# dyn3.trim_edges()
# dyn3.crop_dyn(fmin=Fmin, fmax=Fmax)
# dyn3.refill(linear=False)

# dyn_tot = dyn1 + dyn2 + dyn3



# # index_step = (dyn_tot.dyn.shape[1])/(dyn_tot.tobs)
# # index1_end = int(dyn1.tobs * (dyn_tot.dyn.shape[1])/(dyn_tot.tobs))
# # index2_start = int((dyn2.mjd - dyn1.mjd)*24*60*60 *
# #                    (dyn_tot.dyn.shape[1])/(dyn_tot.tobs))
# # dyn1_2_gap = int(abs((dyn2.mjd - dyn1.mjd)*24*60*60 - (dyn1.tobs) *
# #                      (dyn_tot.dyn.shape[1])/(dyn_tot.tobs)))
# # index2_end = int((dyn1.tobs + dyn2.tobs + dyn1_2_gap) *
# #                  (dyn_tot.dyn.shape[1])/(dyn_tot.tobs))
# # index3_start = int((dyn3.mjd - dyn1.mjd)*24*60*60 *
# #                    (dyn_tot.dyn.shape[1])/(dyn_tot.tobs))

# # dyn_tot.dyn[:, index1_end:index1_end+1] = 600
# # dyn_tot.dyn[:, index2_end:index3_start] = 600

# # dyn_tot.refill(linear=True)


# # index2 = int(dyn1.tobs * (dyn_tot.dyn.shape[1])/(dyn_tot.tobs))
# # index1 = int(((dyn2.mjd - dyn1.mjd)*24*60*60) * (dyn_tot.dyn.shape[1])/(dyn_tot.tobs))
# # index4 = int(((dyn2.mjd - dyn1.mjd)*24*60*60 + dyn2.tobs) * (dyn_tot.dyn.shape[1])/(dyn_tot.tobs))
# # index3 = int(((dyn3.mjd - dyn1.mjd)*24*60*60) * (dyn_tot.dyn.shape[1])/(dyn_tot.tobs))

# # dyn_tot.dyn[:, index1:index2] = np.nan
# # dyn_tot.dyn[:, index3:index4] = np.nan

# # dyn_tot.refill(linear=False)


# # Removing any eclispe
# # start_mjd = dyn_tot.mjd
# # tobs = dyn_tot.tobs
# # times = dyn_tot.times
# # Eclipse_index = SearchEclipse(start_mjd, tobs, times)
# # print(Eclipse_index)
# # if Eclipse_index is not None and len(Eclipse_index) > 1:
# #     for i in range(0, len(Eclipse_index)):
# #         dyn_tot.dyn[:, Eclipse_index[i]-3:Eclipse_index[i]+3] = np.nan
# # elif Eclipse_index is not None:
# #     dyn_tot.dyn[:, Eclipse_index-3:Eclipse_index+3] = np.nan
    
# # Plotting and Saving the Dynamic Spectra for this group
# # dyn_tot.refill(linear=True)
# dyn_tot.plot_dyn(filename='/Users/jacobaskew/Desktop/Dynspec.png')

# # Cropping the Spectra
# dyn_tot.crop_dyn(fmin=FreqFloor, fmax=np.inf)
# dyn_tot.plot_dyn(filename='/Users/jacobaskew/Desktop/CroppedDynspec')

