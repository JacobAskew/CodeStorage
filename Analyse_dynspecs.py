#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 18:38:56 2021
​
@author: dreardon
"""

from scintools.dynspec import Dynspec
from scintools.scint_utils import write_results, read_results, read_par, \
        float_array_from_dict, get_ssb_delay, get_earth_velocity, \
        get_true_anomaly, scint_velocity, pars_to_params
from scintools.scint_models import veff_thin_screen#, effective_velocity_annual
# import corner
from lmfit import Minimizer#, Parameters
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import glob
from copy import deepcopy as cp
# import bilby
# from bilby.core import result
# import pdb

psrname = '0737-3039A'
datadir = '/Users/jacobaskew/Desktop/DoublePulsar_Scintillation/' + str(psrname) + '/Data/'
plotdir = '/Users/jacobaskew/Desktop/DoublePulsar_Scintillation/' + str(psrname) + '/Plots/'
spectradir = '/Users/jacobaskew/Desktop/DoublePulsar_Scintillation/' + str(psrname) + '/Spectra/'
outdir = '/Users/jacobaskew/Desktop/DoublePulsar_Scintillation/' + str(psrname) + '/Datafiles/'
outfile = str(outdir) + str(psrname) + '_Results.txt'

# dynspecs = sorted(glob.glob(datadir + psrname + '/*2019-12-14-19:44*.dynspec'))
# dynspecs = sorted(glob.glob(datadir + '*.dynspec'))
dynspecs = sorted(glob.glob(str(datadir) + 'J0737-3039A_2019-03-26-19:42:32_ch5.0_sub5.0.ar.dynspec'))

uhf = True
measure = True
model = True

##############################################################################
# for dynspec in dynspecs:
#     print(dynspec)
#     File1=dynspec.split(str(datadir))[1]
#     Filename=str(File1.split('.')[0])
    
#     try:
#         dyn = Dynspec(filename=dynspec, process=False)
#         # dyn.crop_dyn(fmin=1300, tmin=2)
#         dyn.trim_edges()
#         dyn.zap()
#         dyn.refill(linear=False)
#         dyn.calc_acf()
#         # dyn.calc_sspec(window_frac=0.1, prewhite=False, lamsteps=True)
#         dyn.plot_dyn(filename=str(spectradir) + str(Filename) + '_Spectra.png')
#         dyn.plot_acf(filename=str(spectradir) + str(Filename) + '_ACF.png', fit=True, crop='auto')
#         write_results(outfile, dyn=dyn)
#     except Exception as e:
#         print(e)
#         continue

##############################################################################
#Below is Daniels code that I don't understand and is efficient but above is mine
##############################################################################
if measure:
    for dynspec in dynspecs:
        try:
            dyn = Dynspec(filename=dynspec, process=False)
            if not uhf and dyn.freq < 1000:
                continue
            elif uhf and dyn.freq > 1000:
                continue
            dyn.plot_dyn()
            #dyn.refill()
            dyn.plot_dyn(filename=plotdir+dyn.name+'.png',
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
##############################################################################
if model:
    results_dir = outdir
    params = read_results(outfile)
    par_dir = '/Users/jacobaskew/Desktop/Swinburne_Daniel/ParFiles/'
    pars = read_par(str(par_dir) + 'J' + str(psrname) + '.par')

    
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
    tobs = float_array_from_dict(params, 'tobs')  # tobs in second
    rcvrs = np.array([rcvr[0] for rcvr in params['name']])
    
    # Sort by MJD
    sort_ind = np.argsort(mjd)
    
    df = np.array(df[sort_ind]).squeeze()
    dnu = np.array(dnu[sort_ind]).squeeze()
    # dnu_est = np.array(dnu_est[sort_ind]).squeeze()
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
    
    # indicies = np.argwhere((tauerr < 0.2*tau) * (tau < 1200) * (np.sqrt(dnu)/tau < 0.01) * (dnuerr < 0.2*dnu))
    
    # df = df[indicies].squeeze()
    # dnu = dnu[indicies].squeeze()
    # # dnu_est = dnu_est[indicies].squeeze()
    # dnuerr = dnuerr[indicies].squeeze()
    # tau = tau[indicies].squeeze()
    # tauerr = tauerr[indicies].squeeze()
    # mjd = mjd[indicies].squeeze()
    # rcvrs = rcvrs[indicies].squeeze()
    # freq = freq[indicies].squeeze()
    # tobs = tobs[indicies].squeeze()
    # bw = bw[indicies].squeeze()
    
    # Make MJD centre of observation, instead of start
    mjd = mjd + tobs/86400/2
    
    # Form Viss from the data
    Aiss = 2.78*10**4  # thin screen, table 2 of Cordes & Rickett (1998)
    D = 1  # kpc
    # viss = Aiss * np.sqrt(D * dnu ) / (freq/1000 * tau)
    # visserr = 0.2*viss  # temporary
    # viss = 1/tau
    # visserr = 0.2*viss#viss*(1/tauerr**2)
    
    viss, visserr = scint_velocity(None, dnu, tau, freq, dnuerr, 
                                                  tauerr, a=Aiss)
    ###############################################################################
    Weights_Option = 0
    Log_Option = 0
    Anisotropy_Option = 0
    ###############################################################################
    
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
    ##############################################################################
    # Preparing the Modelling #

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
    ###############################################################################
    # Modelling #
    
    """
    Fitting routine
    """
    
    refit = True
    chisqr = np.Inf
    print('Doing fit')
    
    #Method: Classical
    nitr = 50
    if refit:
        posarray = []
        for itr in range(0, nitr):
            ipos=[]
            #Optional Printing of the iteration of the fit
            print(itr)
            
            params = pars_to_params(pars)
            
            # ISM model params #
            params.add('s', value=np.random.uniform(low=0, high=1), vary=True, min=0, max=1.0)
            params.add('d', value=D, vary=False, min=0, max=np.inf)  # psr distance in kpc
            params.add('kappa', value=np.random.normal(loc=1, scale=0.2), vary=True, min=0, max=np.inf)  # scale factor
            params.add('vism_ra', value=np.random.uniform(low=-200, high=200), vary=True, min=-200, max=200)
            params.add('vism_dec', value=np.random.uniform(low=-200, high=200), vary=True, min=-200, max=200)
    
            # Pulsar binary params #
            params.add('sense', value=np.random.uniform(low=0, high=1), vary=True, min=0, max=1)
            params.add('KIN', value=93.54, vary=False, min=0, max=180)
            params.add('KOM', value=np.random.uniform(low=0, high=360), vary=True, min=0, max=360)
            params.add('__lnsigma', value=np.random.uniform(low=-15, high=15), vary=True, min=-15, max=15)
    
            # Log Option #
            if Log_Option == 1:
                params.add('log', value=1, vary=False)
    
            # Anisotropy #
            if Anisotropy_Option == 1:
                params.add('psi', value=np.random.uniform(low=0, high=180), vary=True, min=0, max=180)
                params.add('R', value=np.random.uniform(low=0, high=1), vary=True, min=0, max=1)
            
            if Weights_Option == 0:
                func = Minimizer(veff_thin_screen, params, fcn_args=(viss, 1/visserr, U, vearth_ra, vearth_dec) )
            if Weights_Option == 1:
                func = Minimizer(veff_thin_screen, params, fcn_args=(viss, None, U, vearth_ra, vearth_dec) )
            results = func.minimize()
    
            # posarray.append(ipos)
    
            ipos.append(params['sense'].value)
            ipos.append(params['s'].value)
            ipos.append(params['vism_ra'].value)
            ipos.append(params['vism_dec'].value)
            ipos.append(params['kappa'].value)
            # ipos.append(params['vism_psi'].value)
            ipos.append(params['KOM'].value)
            ipos.append(params['__lnsigma'].value)
            if Anisotropy_Option == 1:
                ipos.append(params['psi'].value)
                ipos.append(params['R'].value)
          
    ###############################################################################
    posarray = np.asarray(posarray)
    if Weights_Option == 0:
        func = Minimizer(veff_thin_screen, params, fcn_args=(viss, 1/visserr, U, vearth_ra, vearth_dec) )
    if Weights_Option == 1:
        func = Minimizer(veff_thin_screen, params, fcn_args=(viss, None, U, vearth_ra, vearth_dec) )
    results_new = func.emcee(steps=10000, burn=2500, nwalkers=nitr, is_weighted=False, progress=True)
    
    results = results_new
    
    # Maximum Likelihood Calculations #
    
    NUMBER = np.argmax(results.lnprob)
    MaximumLikelihood = results.flatchain.to_numpy()[NUMBER,:]
    
    # Parameter List Order WARNING SUBJECT TO CHANGE #
    # s = 0
    results.params['s'].value = MaximumLikelihood[0]
    # kappa = 1
    results.params['kappa'].value = MaximumLikelihood[1]
    # vism_ra = 2
    results.params['vism_ra'].value = MaximumLikelihood[2]
    # vism_dec = 3
    results.params['vism_dec'].value = MaximumLikelihood[3]
    # sense = 6
    results.params['sense'].value = MaximumLikelihood[4]
    # KOM = 7
    results.params['KOM'].value = MaximumLikelihood[5]
    # __lnsigma = 8
    results.params['__lnsigma'].value = MaximumLikelihood[6]
    if Anisotropy_Option == 1:
        # psi = 4
        results.params['psi'].value = MaximumLikelihood[7]
        # R = 5
        results.params['R'].value = MaximumLikelihood[8]
    
    #Optional printing of params
    #print(results.params)
    
    KIN = results.params['KIN'].value
    KINerr = results.params['KIN'].stderr
    KOM = results.params['KOM'].value
    KOMerr = results.params['KOM'].stderr
    s = results.params['s'].value
    serr = results.params['s'].stderr
    vism_ra = results.params['vism_ra'].value
    vism_raerr = results.params['vism_ra'].stderr
    vism_dec = results.params['vism_dec'].value
    vism_decerr = results.params['vism_dec'].stderr
    if Anisotropy_Option == 1:
        psi = results.params['psi'].value
        psierr = results.params['psi'].stderr
        R = results.params['R'].value
        Rerr = results.params['R'].stderr
    
    visserr *= np.e**np.mean(results_new.flatchain['__lnsigma'].to_numpy())
    ###############################################################################
    # Rounding #
    #Here I would like to calculate the errors so I may display them in a publishable format
    
    KOM = float(round(KOM,5))
    KOMerr = float(round(KOMerr, 5))
    vism_decerr = float(round(vism_decerr, 5))
    vism_raerr = float(round(vism_raerr, 5))
    serr = float(round(serr, 5))
    vism_dec = float(round(vism_dec, 5))
    vism_ra = float(round(vism_ra, 5))
    s = float(round(s, 5))
    if Anisotropy_Option == 1:
        psi = float(round(psi, 5))
        psierr = float(round(psierr, 5))
        R = float(round(R, 5))
        Rerr = float(round(Rerr, 5))
    
    # Printing Results #
    
    print('=========================================')
    print('inc', KIN, KINerr)
    print('OM', KOM, KOMerr)
    print('vism (RA)', vism_ra, vism_raerr)
    print('vism (DEC)', vism_dec, vism_decerr)
    if Anisotropy_Option == 1:
        print('psi', psi, psierr)
        print('R', R, Rerr)
    print('s', s, serr)
    print('=========================================')
    

    ###############################################################################
    # Applying Annual and Orbital Model #
    if Weights_Option == 1:
        res = veff_thin_screen(params, viss, None, U, vearth_ra, vearth_dec)
        model = viss - res#*visserr
    if Weights_Option == 0:
        res = veff_thin_screen(params, viss, 1/visserr, U, vearth_ra, vearth_dec)
        model = viss - res*visserr
    
    test = cp(params)
    test["A1"].value = 0
    
    if Weights_Option == 1:
        res_nobinary = veff_thin_screen(test, viss, None, U, vearth_ra, vearth_dec)
        model_nobinary = viss - res_nobinary#*visserr
    if Weights_Option == 0:
        res_nobinary = veff_thin_screen(test, viss, 1/visserr, U, vearth_ra, vearth_dec)
        model_nobinary = viss - res_nobinary*visserr
        
    vbinary = model - model_nobinary
    
    if Weights_Option == 1:
        res_noearth = veff_thin_screen(params, viss, None, U, np.zeros(np.shape(vearth_ra)), np.zeros(np.shape(vearth_dec)))
        model_noearth = viss - res_noearth#*visserr
    if Weights_Option == 0:
        res_noearth = veff_thin_screen(params, viss, 1/visserr, U, np.zeros(np.shape(vearth_ra)), np.zeros(np.shape(vearth_dec)))
        model_noearth = viss - res_noearth*visserr
    
    vearth = model - model_noearth
    
    mjd_annual = mjd % 365.2425
    
    sort_annual = np.argsort(mjd_annual)
    sort_U = np.argsort(U)
    
    ##############################################################################
    # Plots #
    
    # Font = 35
    # Size = 80*np.pi #Determines the size of the datapoints used
    # font = {'size'   : 32}
    # matplotlib.rc('font', **font)
    
    # # __lnsigma #
    # __lnsigmaData = results_new.flatchain['__lnsigma'].to_numpy()
    
    # fig = plt.figure(figsize=(15, 10))
    # ax = fig.add_subplot(1, 1, 1)
    # fig.subplots_adjust(hspace=0.5, wspace=0.5)
    # plt.hist(__lnsigmaData, bins=200)
    # plt.xlabel('__lnsigma', fontsize=Font, ha='center')
    # plt.ylabel('Frequency', fontsize=Font, ha='center')
    # plt.grid(True, which="both", ls="-", color='0.65')
    # plt.savefig(str(plot_dir) + str(psrname) + "_" + str(Backend) + "ErrorDistribution_Scintillation.png")
    # plt.close()
    
    
    # font = {'size'   : 19}
    # matplotlib.rc('font', **font)
    
    # # Corner Plot #
    # truths = []
    # for var in results_new.var_names:
    #     truths.append(results_new.params[var].value)
    # labels = results_new.var_names
    # fig = plt.figure(figsize=(25, 20))
    # fig.subplots_adjust(hspace=0.5, wspace=0.5)
    # corner.corner(results_new.flatchain, labels=labels, truths=truths)
    # ##plt.show()
    # plt.savefig(str(plot_dir) + str(psrname) + "_" + str(Backend) + "CornerPlot_Scintillation.png")
    # plt.close()
    
    Font = 35
    Size = 80*np.pi #Determines the size of the datapoints used
    font = {'size'   : 32}
    matplotlib.rc('font', **font)
    
    # # Omega #
    # OmegaData_Scintillation = results_new.flatchain['KOM'].to_numpy()
    
    # fig = plt.figure(figsize=(15, 10))
    # ax = fig.add_subplot(1, 1, 1)
    # fig.subplots_adjust(hspace=0.5, wspace=0.5)
    # plt.hist(OmegaData_Scintillation, bins=200)
    # plt.xlabel(r'$\Omega$ (degrees)', fontsize=Font, ha='center')
    # plt.ylabel('Frequency', fontsize=Font, ha='center')
    # plt.grid(True, which="both", ls="-", color='0.65')
    # plt.savefig(str(plot_dir) + str(psrname) + "_" + str(Backend) + "OmegaDistribution_Scintillation.png")
    # plt.close()
    
    # # Sense #
    # SenseData_Scintillation = results_new.flatchain['sense'].to_numpy()
    # Ratio = round(len(np.argwhere(SenseData_Scintillation > 0.5))/(len(np.argwhere(SenseData_Scintillation < 0.5))),2)
    # if Ratio < 1 and Ratio != 0:
    #     Ratio = round(1/Ratio,2)
    # elif Ratio == 0:
    #     Ratio = 0
    
    # fig = plt.figure(figsize=(15, 10))
    # ax = fig.add_subplot(1, 1, 1)
    # fig.subplots_adjust(hspace=0.5, wspace=0.5)
    # plt.hist(SenseData_Scintillation, bins=200)
    # plt.xlabel('Sense', fontsize=Font, ha='center')
    # plt.ylabel('Frequency', fontsize=Font, ha='center')
    # plt.grid(True, which="both", ls="-", color='0.65')
    # plt.title('Ratio: ' + str(Ratio))
    # plt.savefig(str(plot_dir) + str(psrname) + "_" + str(Backend) + "SenseDistribution_Scintillation.png")
    # plt.close()
    
    # # s #
    # sData_Scintillation = results_new.flatchain['s'].to_numpy()
    
    # fig = plt.figure(figsize=(15, 10))
    # ax = fig.add_subplot(1, 1, 1)
    # fig.subplots_adjust(hspace=0.5, wspace=0.5)
    # plt.hist(sData_Scintillation, bins=200)
    # plt.xlabel('s', fontsize=Font, ha='center')
    # plt.ylabel('Frequency', fontsize=Font, ha='center')
    # plt.grid(True, which="both", ls="-", color='0.65')
    # plt.savefig(str(plot_dir) + str(psrname) + "_" + str(Backend) + "sDistribution_Scintillation.png")
    # plt.close()
    
    # # vism_RA #
    # vism_RAData_Scintillation = results_new.flatchain['vism_ra'].to_numpy()
    
    # fig = plt.figure(figsize=(15, 10))
    # ax = fig.add_subplot(1, 1, 1)
    # fig.subplots_adjust(hspace=0.5, wspace=0.5)
    # plt.hist(vism_RAData_Scintillation, bins=200)
    # plt.xlabel('vism_RA', fontsize=Font, ha='center')
    # plt.ylabel('Frequency', fontsize=Font, ha='center')
    # plt.grid(True, which="both", ls="-", color='0.65')
    # plt.savefig(str(plot_dir) + str(psrname) + "_" + str(Backend) + "vism_RADistribution_Scintillation.png")
    # plt.close()
    
    # # vism_DEC #
    # vism_DECData_Scintillation = results_new.flatchain['vism_dec'].to_numpy()
    
    # fig = plt.figure(figsize=(15, 10))
    # ax = fig.add_subplot(1, 1, 1)
    # fig.subplots_adjust(hspace=0.5, wspace=0.5)
    # plt.hist(vism_DECData_Scintillation, bins=200)
    # plt.xlabel('vism_DEC', fontsize=Font, ha='center')
    # plt.ylabel('Frequency', fontsize=Font, ha='center')
    # plt.grid(True, which="both", ls="-", color='0.65')
    # plt.savefig(str(plot_dir) + str(psrname) + "_" + str(Backend) + "vism_DECDistribution_Scintillation.png")
    # plt.close()
    
    # dnu #
    fig = plt.figure(figsize=(15, 10))
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    ax = fig.add_subplot(1, 1, 1)
    plt.scatter(mjd, dnu, facecolor='blue', marker='o',  s=Size, alpha=0.85)
    plt.errorbar(mjd, dnu, yerr=dnuerr, fmt=' ', ecolor='dimgrey', elinewidth = 1.75, capsize = 2.75)
    plt.title('Scintillation Bandwidth ' + str(psrname), fontsize=Font, ha='center')
    plt.xlabel('MJD', fontsize=Font, ha='center')
    plt.ylabel(r'$\nu$ (MHz)', fontsize=Font, ha='center')
    ax.legend(fontsize='xx-small')
    plt.grid(True, which="both", ls="-", color='0.65')
    plt.savefig(str(plotdir) + str(psrname) + "_" + "dnu_yrs.png")
    plt.close()
    
    # tau #
    fig = plt.figure(figsize=(15, 10))
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    ax = fig.add_subplot(1, 1, 1)
    plt.scatter(mjd, tau, facecolor='red', marker='o',  s=Size, alpha=0.85)
    plt.errorbar(mjd, tau, yerr=tauerr, fmt=' ', ecolor='dimgrey', elinewidth = 1.75, capsize = 2.75)
    plt.title('Scintillation Timescale ' + str(psrname), fontsize=Font, ha='center')
    plt.xlabel('MJD', fontsize=Font, ha='center')
    plt.ylabel(r'$\tau_{d}$ (s)', fontsize=Font, ha='center')
    ax.legend(fontsize='xx-small')
    plt.grid(True, which="both", ls="-", color='0.65')
    plt.savefig(str(plotdir) + str(psrname) + "_" + "tau_yrs.png")
    plt.close()
    
    # Annual #
    mjd_annual = mjd % 365.2425
    
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(1, 1, 1)
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    plt.scatter(mjd_annual, viss - vbinary, facecolor='black', marker='o',  s=Size, alpha=0.75, label='Scintillation Velocity')
    plt.errorbar(mjd_annual, viss - vbinary, yerr=visserr, fmt=' ', ecolor='black', elinewidth = 3, capsize = 0, alpha=0.4)
    plt.plot(mjd_annual[sort_annual], model_nobinary[sort_annual], 'purple', label='Effective Velocity (Fitted Model)', linewidth = 4, alpha=0.7)
    plt.title('Annual Variations ' + str(psrname), fontsize=Font, ha='center')
    plt.xlabel('Annual Phase (arb)', fontsize=Font, ha='center')
    if Log_Option == 0:
        plt.ylabel(r'$V_{iss}$ ($kms^{-1}$)', fontsize=Font, ha='center')
    if Log_Option == 1:
        plt.ylabel(r'Natural Log $V_{iss}$ ($kms^{-1}$)', fontsize=Font, ha='center')
    ax.legend(fontsize='xx-small')
    plt.grid(True, which="both", ls="-", color='0.65')
    plt.savefig(str(plotdir) + str(psrname) + "_" + "Viss_annual.png")
    plt.close()
    
    
    # Orbital #
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(1, 1, 1)
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    plt.scatter(U, viss - vearth, facecolor='black', marker='o',  s=Size, alpha=0.75, label='Scintillation Velocity')
    plt.errorbar(U, viss - vearth, yerr=visserr, fmt=' ', ecolor='black', elinewidth = 3, capsize = 0, alpha=0.4)
    plt.title('Orbital Variations ' + str(psrname), fontsize=Font, ha='center')
    plt.xlabel('True Anomaly (rad)', fontsize=Font, ha='center')
    if Log_Option == 0:
        plt.ylabel(r'$V_{iss}$ ($kms^{-1}$)', fontsize=Font, ha='center')
    if Log_Option == 1:
        plt.ylabel(r'Natural Log $V_{iss}$ ($kms^{-1}$)', fontsize=Font, ha='center')
    plt.plot(U[sort_U], model_noearth[sort_U], 'purple', label='Effective Velocity (Fitted Model)', linewidth = 4, alpha=0.7)
    ax.legend(fontsize='xx-small')
    plt.grid(True, which="both", ls="-", color='0.65')
    plt.savefig(str(plotdir) + str(psrname) + "_" + "Viss_PB.png")
    plt.close()
    
    # MJD #
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(1, 1, 1)
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    plt.scatter(mjd, viss, facecolor='black', marker='o',  s=Size, alpha=0.75, label='Scintillation Velocity')
    plt.errorbar(mjd, viss, yerr=visserr, fmt=' ', ecolor='black', elinewidth = 3, capsize = 0, alpha=0.4)
    plt.plot(mjd, model, color='purple', label='Effective Velocity (Fitted Model)', linewidth = 4, alpha=0.7)
    plt.title('Effective Velocity ' + str(psrname), fontsize=Font, ha='center')
    plt.xlabel('MJD', fontsize=Font, ha='center')
    if Log_Option == 0:
        plt.ylabel(r'$V_{iss}$ ($kms^{-1}$)', fontsize=Font, ha='center')
    if Log_Option == 1:
        plt.ylabel(r'Natural Log $V_{iss}$ ($kms^{-1}$)', fontsize=Font, ha='center')
    ax.legend(fontsize='xx-small')
    plt.grid(True, which="both", ls="-", color='0.65')
    plt.savefig(str(plotdir) + str(psrname) + "_" + "Viss_yrs.png")
    plt.close()
    
    #visserr *= 15
    
    plt.figure(figsize=(12,6))
    # plt.errorbar(dnu_est, dnu, yerr=dnuerr, fmt='o', alpha=0.8)
    inds = np.argwhere((dnu < df))
    # plt.errorbar(dnu_est[inds], dnu[inds], yerr=dnuerr[inds].squeeze(), fmt='o', alpha=0.8)
    #plt.yscale('log')
    #plt.xscale('log')
    plt.grid()
    plt.xlim([0, 1])
    plt.ylim([0, 1.5])
    plt.plot([0, 1], [0, 1], 'k', zorder=3)
    plt.ylabel('Measured scint bandwidth (MHz)')
    plt.xlabel('Estimated scint bandwidth (MHz)')
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
    
    # MJD #
    
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
    
    # Annual #
    
    mjd_annual = mjd % 365.2425
    plt.errorbar(mjd_annual, viss, yerr=visserr, fmt='o', )
    plt.xlabel('Annual phase (arb)')
    plt.ylabel('Viss')
    # plt.ylim(0, 200)
    plt.title(psrname)
    plt.show()
    
    
    #Orbital Phase
    
    plt.errorbar(phase, viss, yerr=visserr, fmt='o', )
    plt.xlabel('Orbital phase (degrees)')
    plt.ylabel('Viss (km/s)')
    plt.xlim(0, 360)
    plt.title(psrname + ' Scintillation velocity')
    plt.show()
    
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
    #     ...
