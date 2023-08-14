#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 10:41:43 2023

@author: jacobaskew
"""

###############################################################################
# Importing neccessary things #
from scintools.scint_utils import read_par, pars_to_params, get_earth_velocity, \
    get_true_anomaly
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
import bilby
from astropy.time import Time
from VissGaussianLikelihood import VissGaussianLikelihood
import corner
###############################################################################
# Here you change change which parameters or anisotropy you wish to look at
options = {
    "distance": False,  # If False it will choose a delta function as prior
    "s": False,  # If true it will sample uniformly across parameter space
    "A": False,
    "k": False,
    "vism_ra": True,
    "vism_dec": True,
    "KIN": False,
    "KOM": True,
    "OM": True,
    "OMDOT": False,
    "psi": True,
    "R": True,
    "TAUEFAC": False,
    "DNUEFAC": False,
    "TAUESKEW": False,
    "DNUESKEW": False,
    "alpha": False,
    "Anisotropy_Option": True,  # Adds two anisotropic parameters psi and R
    "AltNoise": False,  # White noise terms on viss instead of tau and dnu
    "resume": False,  # Resumes a saved .json file output from bilby
    "sense": True,  # True = Flipped, False = not-flipped
    "windowing_option": False,  
    "mean_option": False,
}
# You can also change the number of nlive points or inclination model
nlive = 250
# input_KIN = 89.35  # Kramer et al. 2021, timing
# input_KIN_err = 0.05  # timing
input_KIN = 89.35  # Kramer et al. 2021, timing
input_KIN_err = 0.05  # timing
# Other values ... earlier timing 88.69 + 0.5 - 0.76 ... eclipse 89.3 +- 0.1
#  earlier scintillation 88.1 +- 0.5
# A list of distances (kpc)
# 0.465  # timing 
# 0.134  # timing err
# 0.77  # VLBI 
# 0.07  # VLBI err
# 0.735  # weighted 
# 0.06  # weighted err
input_D = 0.7  # From our work
input_Derr = 0.1
# The range of Aiss for alpha = 2.9
# input_Aiss1 = 23264.97206
# input_Aiss2 = 24144.33114
# Mid-point
# input_Aiss = 27700.5179676861  # 23704.651599999997
# input_Aiss = 3.347290399e4  # Thin screen Kolmogorov scattering Lambert 1999
input_Aiss = 21324
# input_Aiss = 2.78e4  # Thin screen Kolmogorov scattering Lambert 1999
# input_Aisserr =   # I am not sure yet how to determine this

# Here we import data and define things that shouldn't need to be changed
distance_option = options['distance']
s_option = options['s']
A_option = options['A']
if A_option:
    k_option = False
else:
    k_option = options['k']
vism_ra_option = options['vism_ra']
vism_dec_option = options['vism_dec']
KIN_option = options['KIN']
KOM_option = options['KOM']
OMDOT_option = options['OMDOT']
OM_option = options['OM']
psi_option = options['psi']
R_option = options['R']
AltNoise_option = options['AltNoise']
if AltNoise_option:
    TAUEFAC_option = options['TAUEFAC']
    DNUEFAC_option = False
    TAUESKEW_option = options['TAUESKEW']
    DNUESKEW_option = False
else:
    TAUEFAC_option = options['TAUEFAC']
    DNUEFAC_option = options['DNUEFAC']
    TAUESKEW_option = options['TAUESKEW']
    DNUESKEW_option = options['DNUESKEW']
alpha_option = options['alpha']
Anisotropy_Option = options['Anisotropy_Option']
windowing_option = options['windowing_option']
mean_option = options['mean_option']
resume = options['resume']
sense = options['sense']  # True = Flipped, False = not-flipped
if sense:
    sense_alt = False
else:
    sense_alt = True
if KIN_option:
    sense = False
    sense_alt = False
if sense:
    input_KIN = 180 - input_KIN
if Anisotropy_Option:
    Isotropic = False
else:
    Isotropic = True



desktopdir = '/Users/jacobaskew/Desktop/'
# datadir = desktopdir + 'DoublePulsar_Project/datasets/FullDataStorage/'
datadir = '/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/New/DataFiles/'
par_dir = '/Users/jacobaskew/Desktop/Swinburne_Daniel/ParFiles/'
psrname = 'J0737-3039A'
wd0 = "/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/New/"
outdir0 = wd0 + "Modelling"
pars = read_par(str(par_dir) + str(psrname) + '.par')
params = pars_to_params(pars)
if not distance_option:
    params.add('d', value=input_D, vary=False)
    params.add('derr', value=input_Derr, vary=False)

viss = np.loadtxt(datadir + '_viss.txt', dtype='float')
visserr = np.loadtxt(datadir + '_visserr.txt', dtype='float')
mjd = np.loadtxt(datadir + '_mjd.txt', dtype='float')
freqMHz = np.loadtxt(datadir + '_freqMHz.txt', dtype='float')
freqGHz = freqMHz / 1e3
dnu = np.loadtxt(datadir + '_dnu.txt', dtype='float')
dnuerr = np.loadtxt(datadir + '_dnuerr.txt', dtype='float')
tau = np.loadtxt(datadir + '_tau.txt', dtype='float')
tauerr = np.loadtxt(datadir + '_tauerr.txt', dtype='float')
phase = np.loadtxt(datadir + '_phase.txt', dtype='float')
U = np.loadtxt(datadir + '_U.txt', dtype='float')
ve_ra = np.loadtxt(datadir + '_ve_ra.txt', dtype='float')
ve_dec = np.loadtxt(datadir + '_ve_dec.txt', dtype='float')

# Loading in 11 extra bits around each mjd for the 10min window effect ...
if os.path.exists(datadir + '10minMJD.txt'):
    mjd_range = np.loadtxt(datadir + '10minMJD.txt', dtype='float')
    U_range = np.loadtxt(datadir + '10minU.txt', dtype='float')
    ve_ra_range = np.loadtxt(datadir + '10minVE_RA.txt', dtype='float')
    ve_dec_range = np.loadtxt(datadir + '10minVE_DEC.txt', dtype='float')
else:
    mjd_10min = []
    step = 5 / 1440
    for i in range(0, len(mjd)):
        mjd_10min.append(np.linspace(mjd[i]-step, mjd[i]+step, 11))
    mjd_10min = np.asarray(mjd_10min).flatten()
    np.savetxt(datadir+'10minMJD.txt', mjd_10min, delimiter=',', fmt='%s')
    mjd_range = mjd_10min
    vearth_ra_10min, vearth_dec_10min = get_earth_velocity(mjd_10min, pars['RAJ'], pars['DECJ'])
    U_10min = get_true_anomaly(mjd_10min, pars)
    vearth_ra_10min = vearth_ra_10min.squeeze()
    vearth_dec_10min = vearth_dec_10min.squeeze()
    om_10min = pars['OM'] + pars['OMDOT']*(mjd_10min - pars['T0'])/365.2425
    phase_10min = U_10min*180/np.pi + om_10min
    phase_10min = phase_10min % 360
    U_range = U_10min
    ve_ra_range = vearth_ra_10min
    ve_dec_range = vearth_dec_10min
    phase_range = phase_10min
    np.savetxt(datadir+'10minU.txt', U_range, delimiter=',', fmt='%s')
    np.savetxt(datadir+'10minVE_RA.txt', ve_ra_range, delimiter=',', fmt='%s')
    np.savetxt(datadir+'10minVE_DEC.txt', ve_dec_range, delimiter=',', fmt='%s')
    np.savetxt(datadir+'10minPhase.txt', phase_range, delimiter=',', fmt='%s')

# Loading in model data across min to max mjd, 10000 steps ...
if os.path.exists(datadir + 'Model_mjdData.txt'):
    Model_mjd = np.loadtxt(datadir + 'Model_mjdData.txt', dtype='float')
    Model_phase = np.loadtxt(datadir + 'Model_phaseData.txt', dtype='float')
    Model_U = np.loadtxt(datadir + 'Model_UData.txt', dtype='float')
    Model_vearth_ra = np.loadtxt(datadir + 'Model_vearth_raData.txt', dtype='float')
    Model_vearth_dec = np.loadtxt(datadir + 'Model_vearth_decData.txt', dtype='float')
else:
    # NUM_MJD = round(11*10*1440*(np.max(mjd)-np.min(mjd)), -3)  # Expensive time to calc using this
    NUM_MJD = 8000
    Model_mjd = np.linspace(np.min(mjd), np.max(mjd), NUM_MJD)
    np.savetxt(datadir+'Model_mjdData.txt', Model_mjd, delimiter=',', fmt='%s')
    Model_vearth_ra, Model_vearth_dec = get_earth_velocity(Model_mjd, pars['RAJ'], pars['DECJ'])
    Model_U = get_true_anomaly(Model_mjd, pars)
    Model_vearth_ra = Model_vearth_ra.squeeze()
    Model_vearth_dec = Model_vearth_dec.squeeze()
    om_model = pars['OM'] + pars['OMDOT']*(Model_mjd - pars['T0'])/365.2425
    Model_phase = Model_U*180/np.pi + om_model
    Model_phase = Model_phase % 360
    np.savetxt(datadir+'Model_UData.txt', Model_U, delimiter=',', fmt='%s')
    np.savetxt(datadir+'Model_vearth_raData.txt', Model_vearth_ra, delimiter=',', fmt='%s')
    np.savetxt(datadir+'Model_vearth_decData.txt', Model_vearth_dec, delimiter=',', fmt='%s')
    np.savetxt(datadir+'Model_phaseData.txt', Model_phase, delimiter=',', fmt='%s')

num_obs = 7
# mjd_ciel = [59780, 59800, 59830, 59880, 59910, 59950, 70000]
# mjd_flor = [0,     59780, 59810, 59870, 59900, 59940, 59990]

mjd_ciel = [59780, 59800,     0, 59880, 59910, 59950, 70000]
mjd_flor = [0,     59780, 70000, 59870, 59900, 59940, 59990]


viss_old = viss.copy()
visserr_old = visserr.copy()
mjd_old = mjd.copy()
freqMHz_old = freqMHz.copy()
freqGHz_old = freqMHz_old / 1e3
dnu_old = dnu.copy()
dnuerr_old = dnuerr.copy()
tau_old = tau.copy()
tauerr_old = tauerr.copy()
phase_old = phase.copy()
U_old = U.copy()
ve_ra_old = ve_ra.copy()
ve_dec_old = ve_dec.copy()
mjd_range_old = mjd_range.copy()
U_range_old = U_range.copy()
ve_ra_range_old = ve_ra_range.copy()
ve_dec_range_old = ve_dec_range.copy()

for i in range(0, num_obs):
    if i == 2:
        continue
    
    labellist = {
    "Anisotropic": Anisotropy_Option,
    "Isotropic": Isotropic,
    "Flipped": sense,
    "Not-Flipped": sense_alt,
    "windowing": windowing_option,
    "mean": mean_option,
    "AltNoise": AltNoise_option,
    str(i): True,
    }

    label = "_".join(filter(None, [key if value else None
                               for key, value in labellist.items()]))

    
    one_obs_sort = [(mjd_old < mjd_ciel[i]) * (mjd_old > mjd_flor[i])]
    one_obs_sort2 = [(mjd_range_old < np.max(mjd_old[(mjd_old < mjd_ciel[i]) * (mjd_old > mjd_flor[i])])) *
                      ((mjd_range_old > np.min(mjd_old[(mjd_old < mjd_ciel[i]) * (mjd_old > mjd_flor[i])])))]
    viss = viss_old[one_obs_sort]
    visserr = visserr_old[one_obs_sort]
    mjd = mjd_old[one_obs_sort]
    freqMHz = freqMHz_old[one_obs_sort]
    freqGHz = freqGHz_old[one_obs_sort]
    dnu = dnu_old[one_obs_sort]
    dnuerr = dnuerr_old[one_obs_sort]
    tau = tau_old[one_obs_sort]
    tauerr = tauerr_old[one_obs_sort]
    phase = phase_old[one_obs_sort]
    U = U_old[one_obs_sort]
    ve_ra = ve_ra_old[one_obs_sort]
    ve_dec = ve_dec_old[one_obs_sort]
    mjd_range = mjd_range_old[one_obs_sort2]
    U_range = U_range_old[one_obs_sort2]
    ve_ra_range = ve_ra_range_old[one_obs_sort2]
    ve_dec_range = ve_dec_range_old[one_obs_sort2]
    
    if mean_option:
        dnu_old = dnu.copy()
        # I want to take the average of dnu at each frequency at each observation
        for i in range(0, len(np.unique(freqMHz))):
            for ii in range(0, 7):
                indices = \
                    np.argwhere((np.unique(freqMHz)[i] == freqMHz) *
                                (np.unique(np.round(mjd, -1))[ii] > mjd-20) *
                                (np.unique(np.round(mjd, -1))[ii] < mjd+20))
                # dnu_means.append(np.mean(dnu[indices]) * np.ones(np.shape(dnu[indices])))
                # print(dnu[indices])
                # print(freqMHz[indices])
                # cm = plt.cm.get_cmap('viridis')
                # z = mjd[indices]
                # plt.scatter(freqMHz[indices], dnu[indices], c=z, cmap=cm)
                # plt.xlabel(r"Frequency $\nu$ (MHz)")
                # plt.ylabel(r"$\Delta\nu_d$ (MHz)")
                # plt.show()
                # plt.close()
                # print(type(indices))
                # print(indices.shape)
                # print(indices)
                dnu[indices] = float(np.mean(dnu[indices]))
        # dnu_means = np.concatenate(np.asarray(dnu_means), axis=0).flatten()
        cm = plt.cm.get_cmap('viridis')
        z = freqMHz
        plt.scatter(dnu_old, dnu, c=z, cmap=cm)
        plt.xlabel(r"Original $\Delta\nu_d$ (MHz)")
        plt.ylabel(r"Mean $\Delta\nu_d$ (MHz)")
        plt.show()
        plt.close()
            
    if windowing_option:
        kwargs = {"U": U_range, "ve_ra": ve_ra_range, "ve_dec": ve_dec_range,
                  "params": params}
    else:
        mjd_range = mjd
        kwargs = {"U": U, "ve_ra": ve_ra, "ve_dec": ve_dec, "params": params}
    ###############################################################################
    
    def effective_velocity_annual_bilby(
        xdata, d, s, k, A, vism_ra, vism_dec, KIN, KOM, OM, OMDOT, TAUEFAC,
        DNUEFAC, TAUESKEW, DNUESKEW, alpha, psi, R, **kwargs):
        """
        Effective velocity thin screen model.
        Uses Eq. 4 from Rickett et al. (2014) for anisotropy coefficients.
        
        ydata: arc curvature
        """
        
        # Define the initial parameters
        params_ = dict(kwargs['params'])
        params_['d']        = d                 # (kpc)
        params_['s']        = s
        params_['vism_ra']  = vism_ra           # (km s-1)
        params_['vism_dec'] = vism_dec          # (km s-1)
        if psi != 500:
            params_['psi']  = psi               # (deg)
            params_['R']    = R
        params_['KOM']      = KOM               # (deg)
        params_['KIN']      = KIN               # (deg)
        params_['OMDOT']    = OMDOT             # (deg yr-1)
        params_['OM']       = OM                # longitude of periastron (deg)
        true_anomaly        = kwargs['U']       # (rad)
        vearth_ra           = kwargs['ve_ra']   # (km s-1)
        vearth_dec          = kwargs['ve_dec']  # (km s-1)
        
        
        # Define some constants
        v_c = 299792.458  # km/s
        kmpkpc = 3.085677581e16
        secperyr = 86400*365.2425
        masrad = np.pi/(3600*180*1000)
        
        A1    = params_['A1']                   # projected semi-major axis (light-sec)
        PB    = params_['PB']                   # orbital period (days)
        ECC   = params_['ECC']                  # orbital eccentricity
        OM    = params_['OM']    * np.pi/180    # longitude of periastron (rad)
        OMDOT = params_['OMDOT'] * np.pi/180    # (rad yr-1)
        INC   = params_['KIN']   * np.pi/180    # inclination (rad)
        KOM   = params_['KOM']   * np.pi/180    # longitude ascending node (rad)
        
        PMRA  = params_['PMRA']                 # proper motion in RA
        PMDEC = params_['PMDEC']                # proper motion in DEC
        
        # other parameters in lower-case
        s        = params_['s']                 # fractional screen distance
        d        = params_['d']                 # pulsar distance (kpc)
        d_kmpkpc = d * kmpkpc                   # pulsar distance (km)
        
        
        if psi != 500:
            r = params_['R']  # axial ratio parameter, see Rickett Cordes 1998
            psi = params_['psi'] * np.pi / 180  # anisotropy angle (rad)
            cosa = np.cos(2 * psi)
            sina = np.sin(2 * psi)
            # quadratic coefficients
            a = (1 - r * cosa) / np.sqrt(1 - r**2)
            b = (1 + r * cosa) / np.sqrt(1 - r**2)
            c = -2 * r * sina / np.sqrt(1 - r**2)
        else:
            a = 1
            b = 1
            c = 0
            
            
        mjd = xdata  # (days)
        
        pmra_v = PMRA * masrad * d_kmpkpc / secperyr  # km s-1
        pmdec_v = PMDEC * masrad * d_kmpkpc / secperyr  # km s-1
       
        ## Calculate the first model ##
        # Calculate pulsar velocity aligned with the line of nodes (Vx) and
        #   perpendicular in the plane (Vy)
        omega = OM + OMDOT*(mjd-params_['T0'])/365.2425  # rad
        
        vp_0 = (2 * np.pi * A1 * v_c) / (np.sin(INC) * PB * 86400 *
                                         np.sqrt(1 - ECC**2))  # km s-1
        vp_x = -vp_0 * (ECC * np.sin(omega) + np.sin(true_anomaly + omega))  # km s-1
        vp_y = vp_0 * np.cos(INC) * (ECC * np.cos(omega) + np.cos(true_anomaly
                                                                  + omega))  # km s-1
        
        # Rotate pulsar velocity into RA/DEC
        vp_ra = np.sin(KOM) * vp_x + np.cos(KOM) * vp_y  # km s-1
        vp_dec = np.cos(KOM) * vp_x - np.sin(KOM) * vp_y  # km s-1
        
        # find total effective velocity in RA and DEC
        veff_ra = s * vearth_ra + (1 - s) * (vp_ra + pmra_v) - vism_ra  # km s-1
        veff_dec = s * vearth_dec + (1 - s) * (vp_dec + pmdec_v) - vism_dec  # km s-1
        
        veff = np.sqrt(a * veff_dec**2 + b * veff_ra**2 + c * veff_ra * veff_dec)  # km s-1
        model = veff / s  # km s-1
        model = np.float64(model)  # km s-1
        
        return model  # km s-1
    
    
    ###############################################################################
    # Here you can change the delta functions or the bounds of the priors
    if Anisotropy_Option:
        outdir = outdir0 + '/Anisotropic/' + str(label)
        try:
            os.mkdir(outdir)
        except OSError as error:
            print(error)
    
        if distance_option:
            d = bilby.core.prior.Uniform(0, 2, 'd')
        else:
            d = bilby.core.prior.analytical.DeltaFunction(input_D, 'd')
        if s_option:
            s = bilby.core.prior.Uniform(0.001, 0.999, 's')
        else:
            s = bilby.core.prior.analytical.DeltaFunction(0.69, 's')
        if k_option:
            k = bilby.core.prior.Uniform(0, 10, 'k')
        else:
            k = bilby.core.prior.analytical.DeltaFunction(1, 'k')
        if A_option:
            # A = bilby.core.prior.Uniform(0, 1e6, 'A')
            A = bilby.core.prior.analytical.Normal(23200, 23200*0.2, 'A')
        else:
            A = bilby.core.prior.analytical.DeltaFunction(input_Aiss, 'A')
        if vism_ra_option:
            vism_ra = bilby.core.prior.Uniform(-300, 300, 'vism_ra')
        else:
            vism_ra = bilby.core.prior.analytical.DeltaFunction(-20, 'vism_ra')
        if vism_dec_option:
            vism_dec = bilby.core.prior.Uniform(-300, 300, 'vism_dec')
        else:
            vism_dec = bilby.core.prior.analytical.DeltaFunction(20, 'vism_dec')
        if KIN_option:
            KIN = bilby.core.prior.Uniform(0, 180, 'KIN', boundary='periodic')
        else:
            KIN = bilby.core.prior.analytical.DeltaFunction(input_KIN, 'KIN')
        if KOM_option:
            KOM = bilby.core.prior.Uniform(0, 360, 'KOM', boundary='periodic')
        else:
            KOM = bilby.core.prior.analytical.DeltaFunction(90, name='KOM')
        if OMDOT_option:
            OMDOT = bilby.core.prior.Uniform(10, 25, 'OMDOT')
        else:
            # OMDOT = bilby.core.prior.analytical.DeltaFunction(pars['OMDOT'],
            #                                                   name='OMDOT')
            OMDOT = bilby.core.prior.analytical.DeltaFunction(0, name='OMDOT')
        if OM_option:
            OM = bilby.core.prior.Uniform(0, 360, 'OM') # deg, gets conv into rad in the function
        else:
            OM = bilby.core.prior.analytical.DeltaFunction(pars['OM'], name='OM')
        if TAUEFAC_option:
            TAUEFAC = bilby.core.prior.Uniform(-1, 1, 'TAUEFAC')  # -1, 1
        else:
            TAUEFAC = bilby.core.prior.analytical.DeltaFunction(0, 'TAUEFAC')
        if TAUESKEW_option:
            TAUESKEW = bilby.core.prior.Uniform(-10, 10, 'TAUESKEW')  # -10, 10
        else:
            TAUESKEW = bilby.core.prior.analytical.DeltaFunction(-100, 'TAUESKEW')
        if DNUEFAC_option:
            DNUEFAC = bilby.core.prior.Uniform(-1, 1, 'DNUEFAC')  # -1, 1
        else:
            DNUEFAC = bilby.core.prior.analytical.DeltaFunction(0, 'DNUEFAC')
        if DNUESKEW_option:
            DNUESKEW = bilby.core.prior.Uniform(-10, 10, 'DNUESKEW')  # -10, 10
        else:
            DNUESKEW = bilby.core.prior.analytical.DeltaFunction(-100, 'DNUESKEW')
        if alpha_option:
            alpha = bilby.core.prior.Uniform(0, 6, 'alpha')
        else:
            # alpha = bilby.core.prior.analytical.DeltaFunction(3.5, 'alpha')
            alpha = bilby.core.prior.analytical.DeltaFunction(4.4, 'alpha')
        if psi_option:
            psi = bilby.core.prior.Uniform(0, 180, 'psi')
        else:
            psi = bilby.core.prior.analytical.DeltaFunction(90, 'psi')
        if R_option:
            R = bilby.core.prior.Uniform(0, 0.999, 'R')
        else:
            R = bilby.core.prior.analytical.DeltaFunction(0.5, 'R')
    # xdata, mjd_stop, d, s, s2, k, k2, A, vism_ra, vism_ra2, vism_dec,
    # vism_dec2, KIN, KOM, TAUEFAC, DNUEFAC, TAUESKEW, DNUESKEW, alpha, psi,
    # psi2, R, R2, **kwargs):
    
        priors = dict(d=d, s=s, k=k, A=A, vism_ra=vism_ra, vism_dec=vism_dec,
                      KIN=KIN, KOM=KOM, OM=OM, OMDOT=OMDOT, alpha=alpha, psi=psi, R=R,
                      TAUEFAC=TAUEFAC, DNUEFAC=DNUEFAC, TAUESKEW=TAUESKEW,
                      DNUESKEW=DNUESKEW)
    
        likelihood = \
            VissGaussianLikelihood(
                x=mjd_range, y=viss, func=effective_velocity_annual_bilby,
                freq=freqMHz, tau=tau, dnu=dnu, tauerr=tauerr, dnuerr=dnuerr,
                sigma=None, **kwargs)
    
    else:
        outdir += '/Isotropic/' + str(label)
        try:
            os.mkdir(outdir)
        except OSError as error:
            print(error)
        if distance_option:
            d = bilby.core.prior.Uniform(0, 10, 'd')
        else:
            d = bilby.core.prior.analytical.DeltaFunction(input_D, 'd')
        if s_option:
            s = bilby.core.prior.Uniform(0.001, 0.9, 's')
        else:
            s = bilby.core.prior.analytical.DeltaFunction(0.72, 's')
        if k_option:
            k = bilby.core.prior.Uniform(0, 10, 'k')
        else:
            k = bilby.core.prior.analytical.DeltaFunction(1, 'k')
        if A_option:
            A = bilby.core.prior.Uniform(0, 1e5, 'A')
        else:
            A = bilby.core.prior.analytical.DeltaFunction(input_Aiss, 'A')
        if vism_ra_option:
            vism_ra = bilby.core.prior.Uniform(-300, 300, 'vism_ra')
        else:
            vism_ra = bilby.core.prior.analytical.DeltaFunction(-20, 'vism_ra')
        if vism_dec_option:
            vism_dec = bilby.core.prior.Uniform(-300, 300, 'vism_dec')
        else:
            vism_dec = bilby.core.prior.analytical.DeltaFunction(20, 'vism_dec')
        if KIN_option:
            KIN = bilby.core.prior.Uniform(0, 180, 'KIN', boundary='periodic')
        else:
            KIN = bilby.core.prior.analytical.DeltaFunction(input_KIN, 'KIN')
        if KOM_option:
            KOM = bilby.core.prior.Uniform(0, 360, 'KOM', boundary='periodic')
        else:
            KOM = bilby.core.prior.analytical.DeltaFunction(65, name='KOM')
        if TAUEFAC_option:
            TAUEFAC = bilby.core.prior.Uniform(-2, 1, 'TAUEFAC')
        else:
            TAUEFAC = bilby.core.prior.analytical.DeltaFunction(0, 'TAUEFAC')
        if DNUEFAC_option:
            DNUEFAC = bilby.core.prior.Uniform(-2, 1, 'DNUEFAC')
        else:
            DNUEFAC = bilby.core.prior.analytical.DeltaFunction(0, 'DNUEFAC')
        if TAUESKEW_option:
            TAUESKEW = bilby.core.prior.Uniform(-10, 2.5, 'TAUESKEW')
        else:
            TAUESKEW = bilby.core.prior.analytical.DeltaFunction(-100, 'TAUESKEW')
        if DNUESKEW_option:
            DNUESKEW = bilby.core.prior.Uniform(-10, 1, 'DNUESKEW')
        else:
            DNUESKEW = bilby.core.prior.analytical.DeltaFunction(-100, 'DNUESKEW')
        if alpha_option:
            alpha = bilby.core.prior.Uniform(0, 5, 'alpha')
        else:
            alpha = bilby.core.prior.analytical.DeltaFunction(4, 'alpha')
        psi = bilby.core.prior.analytical.DeltaFunction(500, 'psi')
        R = bilby.core.prior.analytical.DeltaFunction(0.5, 'R')
    
        priors = dict(d=d, s=s, k=k, A=A, vism_ra=vism_ra, vism_dec=vism_dec,
                      KIN=KIN, KOM=KOM, OM=OM, OMDOT=OMDOT, alpha=alpha, psi=psi, R=R,
                      TAUEFAC=TAUEFAC, DNUEFAC=DNUEFAC, TAUESKEW=TAUESKEW,
                      DNUESKEW=DNUESKEW)
    
        likelihood = \
            VissGaussianLikelihood(
                x=mjd_range, y=viss, func=effective_velocity_annual_bilby, freq=freqMHz,
                tau=tau, dnu=dnu, tauerr=tauerr, dnuerr=dnuerr, sigma=None,
                **kwargs)
    
    # And run sampler
    result = bilby.core.sampler.run_sampler(
            likelihood, priors=priors, sampler='dynesty', label=label,
            nlive=nlive, verbose=True, resume=resume, plot=True,
            outdir=outdir)
    
    font = {'size': 16}
    matplotlib.rc('font', **font)
    result.plot_corner()
    plt.show()
    ###############################################################################
    # Here we determine our results and create new models across time
    NUM = np.argmax(result.posterior['log_likelihood'])
    d_result = result.posterior['d'][NUM]
    derr_result = np.std(result.posterior['d'].values)
    params.add('d', value=d_result, vary=False)
    params.add('derr', value=derr_result, vary=False)
    s_result = result.posterior['s'][NUM]
    serr_result = np.std(result.posterior['s'].values)
    params.add('s', value=s_result, vary=False)
    params.add('serr', value=serr_result, vary=False)
    k_result = result.posterior['k'][NUM]
    kerr_result = np.std(result.posterior['k'].values)
    params.add('k', value=k_result, vary=False)
    params.add('kerr', value=kerr_result, vary=False)
    A_result = result.posterior['A'][NUM]
    Aerr_result = np.std(result.posterior['A'].values)
    params.add('A', value=A_result, vary=False)
    params.add('Aerr', value=Aerr_result, vary=False)
    vism_ra_result = result.posterior['vism_ra'][NUM]
    vism_raerr_result = np.std(result.posterior['vism_ra'].values)
    params.add('vism_ra', value=vism_ra_result, vary=False)
    params.add('vism_raerr', value=vism_raerr_result, vary=False)
    vism_dec_result = result.posterior['vism_dec'][NUM]
    vism_decerr_result = np.std(result.posterior['vism_dec'].values)
    params.add('vism_dec', value=vism_dec_result, vary=False)
    params.add('vism_decerr', value=vism_decerr_result, vary=False)
    KIN_result = result.posterior['KIN'][NUM]
    KINerr_result = np.std(result.posterior['KIN'].values)
    params.add('KIN', value=KIN_result, vary=False)
    params.add('KINrr', value=KINerr_result, vary=False)
    KOM_result = result.posterior['KOM'][NUM]
    KOMerr_result = np.std(result.posterior['KOM'].values)
    params.add('KOM', value=KOM_result, vary=False)
    params.add('KOMerr', value=KOMerr_result, vary=False)
    OM_result = result.posterior['OM'][NUM]
    OMerr_result = np.std(result.posterior['OM'].values)
    params.add('OM', value=OM_result, vary=False)
    params.add('OMerr', value=OMerr_result, vary=False)
    OMDOT_result = result.posterior['OMDOT'][NUM]
    OMDOTerr_result = np.std(result.posterior['OMDOT'].values)
    params.add('OMDOT', value=OMDOT_result, vary=False)
    params.add('OMDOTerr', value=OMDOTerr_result, vary=False)
    TAUESKEW_result = float(result.posterior['TAUESKEW'][NUM])
    TAUESKEWerr_result = np.std(result.posterior['TAUESKEW'].values)
    DNUESKEW_result = float(result.posterior['DNUESKEW'][NUM])
    DNUESKEWerr_result = np.std(result.posterior['DNUESKEW'].values)
    TAUEFAC_result = float(result.posterior['TAUEFAC'][NUM])
    TAUEFACerr_result = np.std(result.posterior['TAUEFAC'].values)
    DNUEFAC_result = float(result.posterior['DNUEFAC'][NUM])
    DNUEFACerr_result = np.std(result.posterior['DNUEFAC'].values)
    alpha_result = result.posterior['alpha'][NUM]
    alphaerr_result = np.std(result.posterior['alpha'].values)
    if Anisotropy_Option:
        psi_result = result.posterior['psi'][NUM]
        psierr_result = np.std(result.posterior['psi'].values)
        params.add('psi', value=psi_result, vary=False)
        params.add('psierr', value=psierr_result, vary=False)
        R_result = result.posterior['R'][NUM]
        Rerr_result = np.std(result.posterior['R'].values)
        params.add('R', value=R_result, vary=False)
        params.add('Rerr', value=Rerr_result, vary=False)
    else:
        psi_result = 500
        psierr_result = 0
        R_result = 0.5
        Rerr_result = 0
    # Aiss will change depending on s and k which change in our model
    
    Aiss = k_result * A_result * np.sqrt((2*(1-s_result))/(s_result))
    New_viss = Aiss * (np.sqrt(d_result*dnu))/(freqGHz*tau)
    
    if AltNoise_option:
        New_visserr = New_viss * np.sqrt((derr_result/(2*d_result))**2 +
                                           (dnuerr/(2*dnu))**2 + (-tauerr/tau)**2)
        New_visserr = np.sqrt((New_visserr * 10**TAUEFAC_result)**2 +
                              ((10**TAUESKEW_result)*(freqGHz/1)**(alpha_result)
                                )**2)
    else:
        # Re-calculate the errors on dnu, tau, viss as well as the magnitude of viss
        New_dnuerr = np.sqrt((dnuerr * 10**DNUEFAC_result)**2 +
                             ((10**DNUESKEW_result) * (freqGHz/1)**alpha_result)**2)
        New_tauerr = np.sqrt((tauerr * 10**TAUEFAC_result)**2 +
                             ((10**TAUESKEW_result) * (freqGHz/1)**(alpha_result/2))**2)
        New_visserr = New_viss * np.sqrt((derr_result/(2*d_result))**2 +
                                     (New_dnuerr/(2*dnu))**2 + (-New_tauerr/tau)**2)
    
    # # Determine mjd_year and mjd_annual and mjd_doy
    # mjd_year = Time(mjd, format='mjd').byear
    # mjd_annual = mjd % 365.2425
    # # Finding the true 'day of year' jan 1st = 1
    # rawtime = Time(mjd, format='mjd').yday
    # total_day = []
    # for i in range(0, len(rawtime)):
    #     days = float(rawtime[i][5:8])
    #     hours = float(rawtime[i][9:11])
    #     minuets = float(rawtime[i][12:14])
    #     seconds = float(rawtime[i][15:21])
    #     total_day.append(days + (hours/24) + (minuets/1440) + (seconds/86400))
    # mjd_doy = np.asarray(total_day)
    # # Determine mjd_year and mjd_annual and mjd_doy
    # Model_mjd_year = Time(Model_mjd, format='mjd').byear
    # Model_mjd_annual = Model_mjd % 365.2425
    # # Finding the true 'day of year' jan 1st = 1
    # Model_rawtime = Time(Model_mjd, format='mjd').yday
    # Model_total_day = []
    # for i in range(0, len(rawtime)):
    #     days = float(Model_rawtime[i][5:8])
    #     hours = float(Model_rawtime[i][9:11])
    #     minuets = float(Model_rawtime[i][12:14])
    #     seconds = float(Model_rawtime[i][15:21])
    #     Model_total_day.append(days + (hours/24) + (minuets/1440) +
    #                            (seconds/86400))
    # Model_mjd_doy = np.asarray(Model_total_day)
    ###############################################################################
    # Calculating new models to the plot across with our data
    A1_temp = params['A1'].value
    zeros = np.zeros(np.shape(mjd_range))
    if windowing_option:
        kwargs = {"U": U_range, "ve_ra": ve_ra_range, "ve_dec": ve_dec_range,
              "params": params}
    else:
        kwargs = {"U": U, "ve_ra": ve_ra, "ve_dec": ve_dec, "params": params}
    
    orig_veff = effective_velocity_annual_bilby(
        mjd_range, d_result, s_result, k_result, A_result, vism_ra_result,
        vism_dec_result, KIN_result, KOM_result, OM_result, OMDOT_result, TAUEFAC_result, DNUEFAC_result,
        TAUESKEW_result, DNUESKEW_result, alpha_result, psi_result,
        R_result, **kwargs)
    
    params.add('A1', value=0, vary=False)
    
    orig_veff_VE = effective_velocity_annual_bilby(
        mjd_range, d_result, s_result, k_result,
        A_result, 0, 0, KIN_result, KOM_result, OM_result, OMDOT_result, TAUEFAC_result,
        DNUEFAC_result, TAUESKEW_result, DNUESKEW_result, alpha_result, psi_result,
        R_result, **kwargs)
    
    params.add('A1', value=A1_temp, vary=False)
    if windowing_option:
        kwargs = {"U": U_range, "ve_ra": zeros, "ve_dec": zeros,
              "params": params}
    else:
        kwargs = {"U": U, "ve_ra": zeros, "ve_dec": zeros, "params": params}
    
    orig_veff_VP = effective_velocity_annual_bilby(
        mjd_range, d_result, s_result, k_result,
        A_result, 0, 0, KIN_result, KOM_result, OM_result, OMDOT_result, TAUEFAC_result,
        DNUEFAC_result, TAUESKEW_result, DNUESKEW_result, alpha_result, psi_result,
        R_result, **kwargs)
    
    params.add('A1', value=0, vary=False)
    
    orig_veff_IISM = effective_velocity_annual_bilby(
        mjd_range, d_result, s_result, k_result,
        A_result, vism_ra_result, vism_dec_result,
        KIN_result, KOM_result, OM_result, OMDOT_result, TAUEFAC_result, DNUEFAC_result,
        TAUESKEW_result, DNUESKEW_result, alpha_result, psi_result,
        R_result, **kwargs)
    
    params.add('A1', value=A1_temp, vary=False)
    
    if windowing_option:
        veff_new = []
        veff_VE_new = []
        veff_VP_new = []
        veff_IISM_new = []
        step = 0
        for i in range(0, len(mjd)):
            veff_new.append(np.mean(orig_veff[step:11+step]))
            veff_VE_new.append(np.mean(orig_veff_VE[step:11+step]))
            veff_VP_new.append(np.mean(orig_veff_VP[step:11+step]))
            veff_IISM_new.append(np.mean(orig_veff_IISM[step:11+step]))
            step += 11
        veff = np.asarray(veff_new)
        veff_VE = np.asarray(veff_VE_new)
        veff_VP = np.asarray(veff_VP_new)
        veff_IISM = np.asarray(veff_IISM_new)
    else:
        veff = orig_veff
        veff_VE = orig_veff_VE
        veff_VP = orig_veff_VP
        veff_IISM = orig_veff_IISM
    ############ Now we do the same thing across the 10000 model space ############
    kwargs = {"U": Model_U, "ve_ra": Model_vearth_ra, "ve_dec": Model_vearth_dec,
              "params": params}
    Model_zeros = np.zeros(np.shape(Model_mjd))
    
    Model_veff = effective_velocity_annual_bilby(
        Model_mjd, d_result, s_result, k_result,
        A_result, vism_ra_result, vism_dec_result,
        KIN_result, KOM_result, OM_result, OMDOT_result, TAUEFAC_result, DNUEFAC_result,
        TAUESKEW_result, DNUESKEW_result, alpha_result, psi_result,
        R_result, **kwargs)
    
    params.add('A1', value=0, vary=False)
    
    Model_veff_VE = effective_velocity_annual_bilby(
        Model_mjd, d_result, s_result, k_result,
        A_result, 0, 0, KIN_result, KOM_result, OM_result, OMDOT_result, TAUEFAC_result,
        DNUEFAC_result, TAUESKEW_result, DNUESKEW_result, alpha_result, psi_result,
        R_result, **kwargs)
    
    params.add('A1', value=A1_temp, vary=False)
    kwargs = {"U": Model_U, "ve_ra": Model_zeros, "ve_dec": Model_zeros,
              "params": params}
    
    Model_veff_VP = effective_velocity_annual_bilby(
        Model_mjd, d_result, s_result, k_result,
        A_result, 0, 0, KIN_result, KOM_result, OM_result, OMDOT_result, TAUEFAC_result,
        DNUEFAC_result, TAUESKEW_result, DNUESKEW_result, alpha_result, psi_result,
        R_result, **kwargs)
    
    params.add('A1', value=0, vary=False)
    
    Model_veff_IISM = effective_velocity_annual_bilby(
        Model_mjd, d_result, s_result, k_result,
        A_result, vism_ra_result, vism_dec_result,
        KIN_result, KOM_result, OM_result, OMDOT_result, TAUEFAC_result, DNUEFAC_result,
        TAUESKEW_result, DNUESKEW_result, alpha_result, psi_result,
        R_result, **kwargs)
    
    params.add('A1', value=A1_temp, vary=False)
    
    # Determining the values of our measurements only effected by orb/annual phase
    viss_VP_only = New_viss - veff_VE  # - veff_IISM
    viss_VE_only = New_viss - veff_VP  # - veff_IISM
    ###############################################################################
    # Here we begin creating the necessary diagnostic plots and save results
    
    
    # Defining the argsort for our datasets
    mjd_sort = np.argsort(mjd)
    phase_sort = np.argsort(phase)
    # mjd_annual_sort = np.argsort(mjd_annual)
    freq_sort = np.argsort(freqMHz)
    Model_phase_sort = np.argsort(Model_phase)
    Model_mjd_sort = np.argsort(Model_mjd)
    # Model_mjd_annual_sort = np.argsort(Model_mjd_annual)
    ###############################################################################
    # If we are modelling with OM and OMDOT then we need to recalculate the phase here
    if pars['OM'] != OM_result or pars['OMDOT'] != OMDOT_result:
        if windowing_option:
            old_phase_range = phase_range.copy()
            om_10min = OM_result + OMDOT_result*(mjd_range - pars['T0'])/365.2425
            phase_range = U_range*180/np.pi + om_10min
            phase_range = phase_range % 360
        else:
            old_phase = phase.copy()
            om = OM_result + OMDOT_result*(mjd - pars['T0'])/365.2425
            phase = U*180/np.pi + om
            phase = phase % 360
    
        old_Model_phase = Model_phase.copy()
        om_model = OM_result + OMDOT_result*(Model_mjd - pars['T0'])/365.2425
        Model_phase = Model_U*180/np.pi + om_model
        Model_phase = Model_phase % 360
    ###############################################################################

    Size = 80*np.pi  # Determines the size of the datapoints used
    font = {'size': 28}
    matplotlib.rc('font', **font)
    
    # # FIGURE 1: Year against corrected data with model
    # fig = plt.figure(figsize=(20, 10))
    # fig.subplots_adjust(hspace=0.5, wspace=0.5)
    # ax = fig.add_subplot(1, 1, 1)
    # # ax.errorbar(mjd_year[mjd_sort], New_viss[mjd_sort],
    # #             yerr=New_visserr[mjd_sort],
    # #             fmt='o', ecolor='k', elinewidth=2, capsize=3, alpha=0.55)
    # ax.plot(Model_mjd_year[Model_mjd_sort], Model_veff[Model_mjd_sort], c='k',
    #         alpha=0.2)
    # xl = plt.xlim()
    # ax.set_xlabel('Year')
    # ax.set_ylabel(r'Scintillation Velocity (km$\,$s$^{-1}$)')
    # ax.set_xlim(xl)
    # ax.set_xlim(np.min(Model_mjd_year), np.max(Model_mjd_year))
    # plt.tight_layout()
    # plt.savefig(str(outdir) + "/NewViss+Model_year.png", dpi=400)
    # plt.savefig(str(outdir) + "/NewViss+Model_year.pdf", dpi=400)
    # plt.show()
    # plt.close()
    
    # # # Phase
    # # phase_sort = np.argsort(phase)
    # fig = plt.figure(figsize=(20, 10))
    # fig.subplots_adjust(hspace=0.5, wspace=0.5)
    # #
    # ax = fig.add_subplot(1, 1, 1)
    # # ax.errorbar(phase[phase_sort],
    # #             viss_VP_only[phase_sort],
    # #             yerr=New_visserr[phase_sort],
    # #             fmt='o', ecolor='k', elinewidth=2, capsize=3, alpha=0.55)
    # ax.plot(Model_phase[Model_phase_sort],
    #         Model_veff_VP[Model_phase_sort], c='k', alpha=0.9)
    # # xl = plt.xlim()
    # ax.set_xlabel('Orbital Phase (degrees)')
    # ax.set_ylabel(r'Binary Scintillation Velocity (km$\,$s$^{-1}$)')
    # # ax.set_xlim(xl)
    # ax.set_xlim(np.min(Model_phase), np.max(Model_phase))
    # plt.tight_layout()
    # plt.savefig(str(outdir) + "/NewViss+Model_phase.png", dpi=400)
    # plt.savefig(str(outdir) + "/NewViss+Model_phase.pdf", dpi=400)
    # plt.show()
    # plt.close()
    
    # # # Annual
    
    # fig = plt.figure(figsize=(20, 10))
    # fig.subplots_adjust(hspace=0.5, wspace=0.5)
    # #
    # ax = fig.add_subplot(1, 1, 1)
    # # ax.errorbar(mjd_annual[mjd_annual_sort],
    # #             viss_VE_only[mjd_annual_sort],
    # #             yerr=New_visserr[mjd_annual_sort], fmt='o', ecolor='k',
    # #             elinewidth=2, capsize=3, alpha=0.55)
    # ax.plot(Model_mjd_annual[Model_mjd_annual_sort],
    #         Model_veff_VE[Model_mjd_annual_sort], c='k', alpha=0.9)
    # xl = plt.xlim()
    # ax.set_xlabel('Annual Phase (days)')
    # ax.set_ylabel(r'Annual Scintillation Velocity (km$\,$s$^{-1}$)')
    # ax.set_xlim(xl)
    # ax.set_xlim(np.min(Model_mjd_annual), np.max(Model_mjd_annual))
    # plt.tight_layout()
    # plt.savefig(str(outdir) + "/NewViss+Model_annual.png", dpi=400)
    # plt.savefig(str(outdir) + "/NewViss+Model_annual.pdf", dpi=400)
    # plt.show()
    # plt.close()
    
    # # RESIDUALS
    # New_viss, New_visserr
    # Phase
    phase_sort = np.argsort(phase)
    fig = plt.figure(figsize=(20, 10))
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    ax = fig.add_subplot(1, 1, 1)
    cm = plt.cm.get_cmap('viridis')
    z = mjd[phase_sort]
    plt.scatter(phase[phase_sort], (New_viss[phase_sort] - veff[phase_sort]),
                alpha=0.7, c=z, cmap=cm,
                label=r'$\sigma_V$ = ' + str(round(np.std((New_viss[phase_sort] -
                                                            veff[phase_sort])), 3)))
    ax.errorbar(phase[phase_sort],
                (New_viss[phase_sort] - veff[phase_sort]),
                yerr=New_visserr[phase_sort],
                fmt=' ', ecolor='k', elinewidth=2, capsize=3, alpha=0.2)
    ax.plot([np.min(phase), np.max(phase)], [0, 0], c='C3')
    xl = plt.xlim()
    ax.set_xlabel('Orbital Phase (degrees)')
    ax.set_ylabel(r'Scintillation Velocity (km$\,$s$^{-1}$)')
    ax.set_xlim(xl)
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(outdir) + "/NewViss+Residuals_phase.png", dpi=400)
    plt.savefig(str(outdir) + "/NewViss+Residuals_phase.pdf", dpi=400)
    plt.show()
    plt.close()
    
    # # Normalised residuals attempt
    residuals = New_viss[phase_sort] - veff[phase_sort]
    Y_i2 = residuals / (New_visserr[phase_sort])
    
    
    # fig = plt.figure(figsize=(20, 10))
    # fig.subplots_adjust(hspace=0.5, wspace=0.5)
    # ax = fig.add_subplot(1, 1, 1)
    # cm = plt.cm.get_cmap('viridis')
    # z = mjd[phase_sort]
    # plt.scatter(phase[phase_sort], Y_i2, alpha=0.7, c=z, cmap=cm,
    #             label=r'$\sigma_V$ = ' + str(round(np.std(Y_i2), 3)))
    # ax.plot([np.min(phase), np.max(phase)], [0, 0], c='C3')
    # xl = ax.get_xlim()
    # ax.set_xlabel('Orbital Phase (degrees)')
    # ax.set_ylabel(r'Normalised Residuals, $V_{ISS}$ (km$\,$s$^{-1}$)')
    # ax.set_xlim(xl)
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(str(outdir) + "/NewViss+Residuals_phase.png", dpi=400)
    # plt.savefig(str(outdir) + "/NewViss+Residuals_phase.pdf", dpi=400)
    # plt.show()
    # plt.close()
    
    # THREE PLOTS IN ONE, PHASE; Data V Model, Residual Data, Normalised Residual Data
    
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True, figsize=(20, 25))
    cm = plt.cm.get_cmap('viridis')
    z = mjd[phase_sort]
    # model_offset = (( params['PB'].value * 24 * 60 ) / 360) * 5
    #
    ax1.scatter(phase[phase_sort], New_viss[phase_sort],
                alpha=0.7, c=z, cmap=cm, label='data')
    ax1.errorbar(phase[phase_sort],
                New_viss[phase_sort],
                yerr=New_visserr[phase_sort],
                fmt=' ', ecolor='k', elinewidth=2, capsize=2, alpha=0.2)
    xl = (np.min(Model_phase), np.max(Model_phase))
    # ax1.plot(Model_phase[Model_phase_sort],
    #         Model_veff[Model_phase_sort], c='C3', alpha=0.1)
    # ax1.plot(test_phase[test_sort],
    #         test_veff[test_sort], c='C2', alpha=0.1)
    ax1.plot(phase[phase_sort], veff[phase_sort], c='C3', alpha=0.4,
             label='model')
    ax1.set_ylabel(r'Binary $V_{ISS}$ (km$\,$s$^{-1}$)')
    ax1.set_xlim(xl)
    ax1.legend(fontsize='small')
    #
    ax2.scatter(phase[phase_sort], (New_viss[phase_sort] - veff[phase_sort]),
                alpha=0.7, c=z, cmap=cm,
                label=r'$\sigma_V$ = ' + str(round(np.std((New_viss[phase_sort] -
                                                           veff[phase_sort])), 3)) + r' (km$\,$s$^{-1}$)')
    ax2.errorbar(phase[phase_sort],
                (New_viss[phase_sort] - veff[phase_sort]),
                yerr=New_visserr[phase_sort],
                fmt=' ', ecolor='k', elinewidth=2, capsize=2, alpha=0.2)
    xl = ax2.get_xlim()
    ax2.plot([xl[0], xl[1]], [0, 0], c='C3')
    ax2.set_ylabel(r'Residual $V_{ISS}$ (km$\,$s$^{-1}$)')
    ax2.set_xlim(xl)
    ax2.legend(fontsize='small')
    #
    ax3.scatter(phase[phase_sort], Y_i2, alpha=0.7, c=z, cmap=cm,
                label=r'$\sigma_V$ = ' + str(round(np.std(Y_i2), 3)))
    xl = ax3.get_xlim()
    ax3.plot([xl[0], xl[1]], [0, 0], c='C3')
    ax3.set_xlabel('Orbital Phase (degrees)')
    ax3.set_ylabel(r'Normalised Residuals, $V_{ISS}$')
    ax3.set_xlim(xl)
    ax3.legend(fontsize='small')
    #
    plt.tight_layout()
    plt.savefig(str(outdir) + "/NewViss+ALL_Residuals_phase.pdf", dpi=400)
    plt.savefig(str(outdir) + "/NewViss+ALL_Residuals_phase.png", dpi=400)
    plt.show()
    plt.close()
    
    # # Annual
    # fig = plt.figure(figsize=(20, 10))
    # fig.subplots_adjust(hspace=0.5, wspace=0.5)
    # ax = fig.add_subplot(1, 1, 1)
    # ax.errorbar(mjd_annual[mjd_annual_sort],
    #             (New_viss - veff)[mjd_annual_sort],
    #             yerr=New_visserr[mjd_annual_sort],
    #             fmt='o', ecolor='k', elinewidth=2, capsize=3, alpha=0.55)
    # ax.plot([np.min(mjd_annual), np.max(mjd_annual)], [0, 0], c='C3')
    # xl = plt.xlim()
    # ax.set_xlabel('Annual Phase (days)')
    # ax.set_ylabel(r'Scintillation Velocity (km$\,$s$^{-1}$)')
    # ax.set_xlim(xl)
    # plt.tight_layout()
    # plt.savefig(str(outdir) + "/NewViss+Residuals_annual.png", dpi=400)
    # plt.savefig(str(outdir) + "/NewViss+Residuals_annual.pdf", dpi=400)
    # plt.show()
    # plt.close()
    
    # # # Year
    # fig = plt.figure(figsize=(20, 10))
    # fig.subplots_adjust(hspace=0.5, wspace=0.5)
    # ax = fig.add_subplot(1, 1, 1)
    # ax.errorbar(mjd_year[mjd_sort], (New_viss - veff)[mjd_sort],
    #             yerr=New_visserr[mjd_sort],
    #             fmt='o', ecolor='k', elinewidth=2, capsize=3, alpha=0.55)
    # ax.plot([np.min(mjd_year), np.max(mjd_year)], [0, 0], c='C3')
    # xl = plt.xlim()
    # ax.set_xlabel('Year')
    # ax.set_ylabel(r'Scintillation Velocity (km$\,$s$^{-1}$)')
    # ax.set_xlim(xl)
    # plt.tight_layout()
    # plt.savefig(str(outdir) + "/NewViss+Residuals_year.png", dpi=400)
    # plt.savefig(str(outdir) + "/NewViss+Residuals_year.pdf", dpi=400)
    # plt.show()
    # plt.close()
    
    # Frequency variations
    
    # fig = plt.figure(figsize=(20, 10))
    # fig.subplots_adjust(hspace=0.5, wspace=0.5)
    # ax = fig.add_subplot(1, 1, 1)
    # ax.errorbar(freqMHz[freq_sort], dnu[freq_sort],
    #             yerr=New_dnuerr[freq_sort],
    #             fmt='o', ecolor='k', elinewidth=2, capsize=3, alpha=0.55)
    # xl = plt.xlim()
    # ax.set_xlabel('Frequency (MHz)')
    # ax.set_ylabel(r'Scintillation Bandwidth (MHz)')
    # ax.set_xlim(xl)
    # plt.tight_layout()
    # plt.savefig(str(outdir) + "/NewDnuFreq.png", dpi=400)
    # plt.savefig(str(outdir) + "/NewDnuFreq.pdf", dpi=400)
    # plt.show()
    # plt.close()
    
    # fig = plt.figure(figsize=(20, 10))
    # fig.subplots_adjust(hspace=0.5, wspace=0.5)
    # ax = fig.add_subplot(1, 1, 1)
    # ax.errorbar(freqMHz[freq_sort], tau[freq_sort],
    #             yerr=New_tauerr[freq_sort],
    #             fmt='o', ecolor='k', elinewidth=2, capsize=3, alpha=0.55)
    # xl = plt.xlim()
    # ax.set_xlabel('Frequency (MHz)')
    # ax.set_ylabel(r'Scintillation Timescale (s)')
    # ax.set_xlim(xl)
    # plt.tight_layout()
    # plt.savefig(str(outdir) + "/NewTauFreq.png", dpi=400)
    # plt.savefig(str(outdir) + "/NewTauFreq.pdf", dpi=400)
    # plt.show()
    # plt.close()
    
    ###############################################################################
    # Collecting a result.csv file
    ResultsFileLoc = outdir + "/" + str(label) + "_Results.csv"
    ResultsFileLoc2 = wd0 + "/Modelling/" + str(label) + "_Results.csv"
    
    sigma_viss = np.std(New_viss-veff)
    
    ln_evidence = result.log_evidence
    max_likelihood = np.max(result.posterior['log_likelihood'])
    
    # Determining the reduced chisqr
    Nparam = np.shape(result.nested_samples)[1] - 2
    Ndata = len(mjd)
    Nfree = Ndata - Nparam
    
    chisqr = np.sum((New_viss-veff)**2/(veff))
    red_chisqr = chisqr / Nfree
    
    print("The chisqr result is", round(chisqr, 3))
    print("The red_chisqr result is", round(red_chisqr, 3))
    
    ResultsFile = np.array(["ln_evidence", ln_evidence, "max_likelihood",
                            max_likelihood, "chisqr", chisqr, "red_chisqr",
                            red_chisqr, "sigma_viss", sigma_viss, "d", d_result,
                            "derr", derr_result, "s", s_result, "serr",
                            serr_result, "k", k_result, "kerr", kerr_result,
                            "vism_ra", vism_ra_result, "vism_dec",
                            vism_dec_result, "vism_raerr", vism_raerr_result,
                            "vism_decerr", vism_decerr_result, "psi", psi_result,
                            "psierr", psierr_result, "R", R_result, "Rerr",
                            Rerr_result, "KIN", KIN_result, "KINerr",
                            KINerr_result, "KOM", KOM_result, "KOMerr",
                            KOMerr_result, "TAUEFAC", TAUEFAC_result, "TAUEFACerr",
                            TAUEFACerr_result, "DNUEFAC", DNUEFAC_result,
                            "DNUEFACerr", DNUEFACerr_result, "TAUESKEW",
                            TAUESKEW_result, "TAUESKEWerr", TAUESKEWerr_result,
                            "DNUESKEW", DNUESKEW_result, "DNUESKEWerr",
                            DNUESKEWerr_result])
    
    np.savetxt(ResultsFileLoc, ResultsFile, delimiter=',', fmt='%s')
    np.savetxt(ResultsFileLoc2, ResultsFile, delimiter=',', fmt='%s')
    ###############################################################################
    
    # if distance_option or A_option:
    #     d_scint_data = np.random.normal(loc=d_result*1e3, scale=derr_result*1e3,
    #                                 size=10000)
    d_timing = 465
    d_timing_err = 134
    
    d_VLBI = 770
    d_VLBI_err = 70
    
    d_weighted = 735
    d_weighted_err = 60
    
    # NEW METHOD
    kolmogorov_AISS = 3.35*10**4
    AISS_new = 27700.5179676861
    A_kol = np.random.normal(loc=kolmogorov_AISS, scale=0, size = 100000)
    A_new = np.random.normal(loc=AISS_new, scale=2000, size = 100000)
    if distance_option:
        D_dist = np.random.normal(loc=d_result*1e3, scale=derr_result*1e3, size = 100000)
        d_scint1_data = D_dist * ((A_kol)/(A_new))**2
        scintdist1 = np.median(d_scint1_data)
        scintdisterr1 = np.std(d_scint1_data)
        
    else:
        D_dist = np.random.normal(loc=1000, scale=0, size = 100000)
        kappa_dist = np.random.normal(loc=k_result, scale=kerr_result, size = 100000)
        
        d_scint1_data = kappa_dist**2 * D_dist * ((A_kol)/(A_new))**2
        scintdist1 = np.median(d_scint1_data)
        scintdisterr1 = np.std(d_scint1_data)
    
    d_scint2_data = result.posterior['d'] * 1e3
    scintdist2 = np.median(d_scint2_data)
    scintdisterr2 = np.std(d_scint2_data)
    # OLD METHOD
    # A_issOG1 = 33472.90
    # A_issOG1err = A_issOG1 * 0.2
    # A_issOG2 = 27691.41036
    # A_issOG2err = A_issOG2 * 0.2
    # model_k = k_result  # 0.5892796948500499
    # model_kerr = kerr_result  # 0.005910350788901568	
    # model_Aiss1 = model_k * A_issOG1
    # model_Aiss1err = np.sqrt((model_k * A_issOG1err)**2 + (A_issOG1 * model_kerr)**2)
    # model_Aiss2 = model_k * (A_issOG1/A_issOG2) * A_issOG1
    # model_Aiss2err = np.sqrt((model_k * A_issOG2err)**2 + (A_issOG2 * model_kerr)**2)
    # A_iss_result = 27700.51797   # The middle value for a range of Aiss values 24584.01068
    # A_iss_resulterr = A_iss_result*0.2  # The range of possible values for Aiss(err) 879.3590799999984
    # model1_k2 = model_Aiss1 / A_iss_result
    # model2_k2 = model_Aiss2 / A_iss_result
    # scintdist1 = model1_k2**2 * d_weighted
    # scintdist2 = model2_k2**2 * d_weighted
    
    # dAres = -2*d_weighted * ((model_k * model_Aiss1)/(A_iss_result))**2 * (1/A_iss_result)
    # dD = ((model_k * model_Aiss1)/(A_iss_result))**2
    # dAiss = 2*d_weighted * model_Aiss1 * ((model_k)/(A_iss_result))**2
    # dkm = 2*d_weighted * model_k * ((model_Aiss1)/(A_iss_result))**2
    # scintdisterr1 = np.sqrt(((dAres)*(A_iss_resulterr))**2 + ((dD)*(d_weighted_err))**2 + ((dAiss)*(model_Aiss1err))**2 + ((dkm)*(model_kerr))**2)
    # d_scint1_data = np.random.normal(loc=scintdist1, scale=scintdisterr1, size=100000)
    
    # dAres2 = -2*d_weighted * ((model_k * model_Aiss2)/(A_iss_result))**2 * (1/A_iss_result)
    # dD2 = ((model_k * model_Aiss2)/(A_iss_result))**2
    # dAiss2 = 2*d_weighted * model_Aiss2 * ((model_k)/(A_iss_result))**2
    # dkm2 = 2*d_weighted * model_k * ((model_Aiss2)/(A_iss_result))**2
    # scintdisterr2 = np.sqrt(((dAres2)*(A_iss_resulterr))**2 + ((dD2)*(d_weighted_err))**2 + ((dAiss2)*(model_Aiss2err))**2 + ((dkm2)*(model_kerr))**2)
    # d_scint2_data = np.random.normal(loc=scintdist2, scale=scintdisterr2, size=100000)
    
    # ((A_issOG1)/(A_issOG2) * A_issOG1 * model_k)**2 * ((d_weighted)/(A_issOG2))
    d_timing_data = np.random.normal(loc=d_timing, scale=d_timing_err,
                                     size=100000)
    d_VLBI_data = np.random.normal(loc=d_VLBI, scale=d_VLBI_err, size=100000)
    d_weighted_data = np.random.normal(loc=d_weighted, scale=d_weighted_err,
                                       size=100000)
    
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(1, 1, 1)
    plt.hist(d_timing_data, bins=100, alpha=0.6, density=True, label='Timing        ' + str(d_timing) + ' +/- ' + str(d_timing_err) +  ' pc')
    plt.hist(d_VLBI_data, bins=100, alpha=0.6, density=True, label='VLBI            ' + str(d_VLBI) + ' +/- ' + str(d_VLBI_err) +  '  pc')
    # plt.hist(d_weighted_data, bins=100, alpha=0.6, density=True,
    #          label='Weighted ' + str(d_weighted) + ' +/- ' + str(d_weighted_err) +  ' pc')
    # plt.hist(d_scint1_data, bins=100, color='C3', alpha=0.6, density=True, label='Scintillation ' + str(int(scintdist1)) + ' +/- ' + str(int(round(scintdisterr1, -1))) + '  pc')
    plt.hist(d_scint2_data, bins=100, color='C3', alpha=0.6, density=True, label='Scintillation ' + str(int(scintdist2)) + ' +/- ' + str(int(round(scintdisterr2, -1))) + '  pc')
    plt.xlabel("Distance to Double Pulsar (pc)")
    plt.ylabel("Density")
    ax.legend(fontsize='small', loc='upper right')
    xl = plt.xlim()
    plt.xlim(0, 1100)
    plt.savefig(str(outdir) + "/DistanceHistogram.png", dpi=400)
    plt.savefig(str(outdir) + "/DistanceHistogram.pdf", dpi=400)
    plt.show()
    plt.close()
    
    ###############################################################################
    # Determining Ar from R
    
    Ar = np.sqrt(abs(-R_result - 1)) / np.sqrt(abs(R_result - 1))
    Ar = np.sqrt(abs(-0.58 - 1)) / np.sqrt(abs(0.58 - 1))
    # Ar_dist = np.random.normal(loc=Ar, scale=Rerr_result, size = 100000)
    Ar_dist = np.random.normal(loc=Ar, scale=0.08, size = 100000)
    Ar_new = np.median(Ar_dist)
    Ar_err = np.std(Ar_dist)
    
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(1, 1, 1)
    plt.hist(Ar_dist, bins=100, alpha=0.6, density=True, label='A_R ' + str(round(Ar_new, 2)) + ' +/- ' + str(round(Ar_err, 2)))
    plt.xlabel("Axial Ratio of Anisotropy (unitless)")
    plt.ylabel("Density")
    ax.legend()
    yl = plt.ylim()
    plt.vlines(Ar_new, yl[0], yl[1], 'C3')
    plt.vlines(Ar_new+Ar_err, yl[0], yl[1], 'C3', linestyles='dashed')
    plt.vlines(Ar_new-Ar_err, yl[0], yl[1], 'C3', linestyles='dashed')
    # xl = plt.xlim()
    # plt.xlim(0, 1250)
    plt.ylim(yl[0], yl[1])
    plt.savefig(str(outdir) + "/ARHistogram.png", dpi=400)
    plt.savefig(str(outdir) + "/ARHistogram.pdf", dpi=400)
    plt.show()
    plt.close()
    
    ###############################################################################
    
    
    def max_likelihood_2d(param_1data, param_2data, paramlabel1="Parameter1",
                          paramlabel2="Parameter2", plot=False):
        
        paramlabel1 = str(paramlabel1)
        paramlabel2 = str(paramlabel2)
        
        H, xedges, yedges = np.histogram2d(param_1data, param_2data, bins=100)
        
        param1 = xedges[np.argwhere(H == H.max())[0][0]]
        param2 = yedges[np.argwhere(H == H.max())[0][0]]
    
        paramvaluestr1 = paramlabel1 + ": " + str(round(param1, 3))
        paramvaluestr2 = paramlabel2 + ": " + str(round(param2, 3))
    
        if plot:
            fig = plt.figure(figsize=(20, 10))
            ax = fig.add_subplot(1, 1, 1)
            corner.hist2d(param_1data, param_2data, fill_contours=True, smooth=True)
            xl = plt.xlim()
            yl = plt.ylim()
            plt.scatter(param1, param2, s=150, c='C0', marker='o',
                        label=paramvaluestr1)
            plt.scatter(param1, param2, s=150, c='C0', marker='o',
                        label=paramvaluestr2)
            plt.hlines(param2, xl[0], xl[1], colors='C0')
            plt.vlines(param1, yl[0], yl[1], colors='C0')
            plt.xlabel(paramlabel1)
            plt.ylabel(paramlabel2)
            ax.legend()
            plt.show()
            plt.close()
    
        return param1, param2
    
    
    ###############################################################################
    max_likelihood_2d(result.posterior['d'].values, result.posterior['A'].values,
                      paramlabel1="Distance (kpc)",
                      paramlabel2=r"$A_{iss}$ ($km\,s^{-1}$)", plot=True)
    

mjds = np.unique(np.round(mjd, -1))
OMs    = np.asarray([40.77, 42.06, 35.98, 45.35, 47.34, 49.25, 51.99])
OMerrs = np.asarray([1, 1, 1, 1, 1, 1, 1])
fit = np.polyfit(mjds, OMs, 1)
fitFN = np.poly1d(fit)
fitdata = fitFN(mjds)
y = params['OM'].value + params['OMDOT'].value * ((mjds-params['T0'].value)/365.25)
y = y % 360
fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(1, 1, 1)
plt.errorbar(mjds, OMs, yerr=OMerrs, fmt='o')
plt.xlabel("MJD (days)")
plt.ylabel(r"$\omega$ (deg)")
plt.plot(mjds, fitdata, label=r'Fit: $\dot\omega$= ' + str(round(fitFN[1]*365.25, 3)))
plt.plot(mjds, y, label=r'Kramer: $\dot\omega$=16.899')
ax.legend()
plt.show()
plt.close()
###############################################################################
# Checking out epochs

num_obs = 7
mjd_ciel = [59780, 59800, 59830, 59880, 59910, 59950, 70000]
mjd_flor = [0,     59780, 59810, 59870, 59900, 59940, 59990]

for i in range(0, num_obs):
    epochsort = np.argwhere((mjd_flor[i] < mjd) * (mjd_ciel[i] > mjd))

    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(1, 1, 1)
    plt.errorbar(phase, tau, yerr=tauerr, fmt='o')
    plt.errorbar(phase[epochsort], tau[epochsort], yerr=tauerr[epochsort].flatten(), ecolor='C3', fmt='o')
    plt.xlabel("Orbital Phase (deg)")
    plt.ylabel(r"$\tau_d$ (s)")
    plt.show()
    plt.close()

