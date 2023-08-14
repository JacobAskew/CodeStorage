#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 10:41:43 2023

@author: jacobaskew
"""

###############################################################################
# Importing neccessary things #
from scintools.scint_utils import read_par, pars_to_params
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
import bilby
from astropy.time import Time
from VissGaussianLikelihood import VissGaussianLikelihood
###############################################################################
# Here you change change which parameters or anisotropy you wish to look at
options = {
    "distance": True,  # If False it will choose a delta function as prior
    "s": True,  # If true it will sample uniformly across parameter space
    "A": False,
    "k": False,
    "vism_ra": True,
    "vism_dec": True,
    "KIN": True,
    "KOM": True,
    "psi": True,
    "R": True,
    "TAUEFAC": True,
    "DNUEFAC": True,
    "TAUESKEW": True,
    "DNUESKEW": True,
    "alpha": False,
    "Anisotropy_Option": True,  # Adds two anisotropic parameters psi and R
    "AltNoise": False,  # White noise terms on viss instead of tau and dnu
    "resume": False,  # Resumes a saved .json file output from bilby
    "sense": False,  # True = Flipped, False = not-flipped
    "windowing_option": True,  
    "multiple_s": False,  
}
# You can also change the number of nlive points or inclination model
nlive = 250
# input_KIN = 89.35  # Kramer et al. 2021, timing
# input_KIN_err = 0.05  # timing
input_KIN = 89.2  # Kramer et al. 2021, timing
input_KIN_err = 0.2  # timing
# Other values ... earlier timing 88.69 + 0.5 - 0.76 ... eclipse 89.3 +- 0.1
#  earlier scintillation 88.1 +- 0.5
# A list of distances (kpc)
# 0.465  # timing 
# 0.134  # timing err
# 0.77  # VLBI 
# 0.07  # VLBI err
# 0.735  # weighted 
# 0.06  # weighted err
input_D = 544
input_Derr = 80
# The range of Aiss for alpha = 2.9
# input_Aiss1 = 23264.97206
# input_Aiss2 = 24144.33114
# Mid-point
# input_Aiss = 27700.5179676861  # 23704.651599999997
input_Aiss = 3.347290399e4  # Thin screen Kolmogorov scattering Lambert 1999
# input_Aiss = 2.78e4  # Thin screen Kolmogorov scattering Lambert 1999
# input_Aisserr =   # I am not sure yet how to determine this

# Here we import data and define things that shouldn't need to be changed
distance_option = options['distance']
s_option = options['s']
multiple_s_option = options['multiple_s']
A_option = options['A']
if A_option:
    k_option = False
else:
    k_option = options['k']
vism_ra_option = options['vism_ra']
vism_dec_option = options['vism_dec']
KIN_option = options['KIN']
KOM_option = options['KOM']
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

labellist = {
    "Anisotropic": Anisotropy_Option,
    "Isotropic": Isotropic,
    "Flipped": sense,
    "Not-Flipped": sense_alt,
    "windowing": windowing_option,
    "AltNoise": AltNoise_option,
    "MultipleScreens": multiple_s_option,
}

desktopdir = '/Users/jacobaskew/Desktop/'
datadir = desktopdir + 'DoublePulsar_Project/datasets/FullDataStorage/'
par_dir = '/Users/jacobaskew/Desktop/Swinburne_Daniel/ParFiles/'
psrname = 'J0737-3039A'
label = "_".join(filter(None, [key if value else None
                               for key, value in labellist.items()]))
wd0 = "/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/New/"
outdir = wd0 + "Modelling"
pars = read_par(str(par_dir) + str(psrname) + '.par')
params = pars_to_params(pars)
if not distance_option:
    params.add('d', value=input_D, vary=False)
    params.add('derr', value=input_Derr, vary=False)

viss = np.loadtxt(datadir + 'Full_VissData.txt', dtype='float')
visserr = np.loadtxt(datadir + 'Full_VisserrData.txt', dtype='float')
mjd = np.loadtxt(datadir + 'Full_MJDData.txt', dtype='float')
freqMHz = np.loadtxt(datadir + 'Full_FreqData.txt', dtype='float')
freqGHz = freqMHz / 1e3
dnu = np.loadtxt(datadir + 'Full_DnuData.txt', dtype='float')
dnuerr = np.loadtxt(datadir + 'Full_DnuerrData.txt', dtype='float')
tau = np.loadtxt(datadir + 'Full_TauData.txt', dtype='float')
tauerr = np.loadtxt(datadir + 'Full_TauerrData.txt', dtype='float')
phase = np.loadtxt(datadir + 'Full_PhaseData.txt', dtype='float')
mjd_range = np.loadtxt(datadir + '10minMJD.txt', dtype='float')
U_range = np.loadtxt(datadir + '10minU.txt', dtype='float')
ve_ra_range = np.loadtxt(datadir + '10minVE_RA.txt', dtype='float')
ve_dec_range = np.loadtxt(datadir + '10minVE_DEC.txt', dtype='float')
U = np.loadtxt(datadir + 'Full_UData.txt', dtype='float')
ve_ra = np.loadtxt(datadir + 'Full_ve_raData.txt', dtype='float')
ve_dec = np.loadtxt(datadir + 'Full_ve_decData.txt', dtype='float')
if not multiple_s_option:
    mjd0 = np.max(mjd) + 10

# Ryan S. Can you fit just one orbit and see what happens ...
one_obs_sort = [mjd < 59780]
one_obs_sort2 = [mjd_range < np.max(mjd[mjd < 59780])]
viss = viss[one_obs_sort]
visserr = visserr[one_obs_sort]
mjd = mjd[one_obs_sort]
freqMHz = freqMHz[one_obs_sort]
freqGHz = freqGHz[one_obs_sort]
dnu = dnu[one_obs_sort]
dnuerr = dnuerr[one_obs_sort]
tau = tau[one_obs_sort]
tauerr = tauerr[one_obs_sort]
phase = phase[one_obs_sort]
U = U[one_obs_sort]
ve_ra = ve_ra[one_obs_sort]
ve_dec = ve_dec[one_obs_sort]
mjd_range = mjd_range[one_obs_sort2]
U_range = U_range[one_obs_sort2]
ve_ra_range = ve_ra_range[one_obs_sort2]
ve_dec_range = ve_dec_range[one_obs_sort2]

# testdir = "/Users/jacobaskew/Desktop/Data4Ryan/"

# np.savetxt(testdir+"viss.txt", viss, delimiter=',', fmt='%s')
# np.savetxt(testdir+"visserr.txt", visserr, delimiter=',', fmt='%s')
# np.savetxt(testdir+"mjd.txt", mjd, delimiter=',', fmt='%s')
# np.savetxt(testdir+"freqMHz.txt", freqMHz, delimiter=',', fmt='%s')
# np.savetxt(testdir+"dnu.txt", dnu, delimiter=',', fmt='%s')
# np.savetxt(testdir+"dnuerr.txt", dnuerr, delimiter=',', fmt='%s')
# np.savetxt(testdir+"tau.txt", tau, delimiter=',', fmt='%s')
# np.savetxt(testdir+"tauerr.txt", tauerr, delimiter=',', fmt='%s')
# np.savetxt(testdir+"phase.txt", phase, delimiter=',', fmt='%s')
# np.savetxt(testdir+"U.txt", U, delimiter=',', fmt='%s')
# np.savetxt(testdir+"ve_ra.txt", ve_ra, delimiter=',', fmt='%s')
# np.savetxt(testdir+"ve_dec.txt", ve_dec, delimiter=',', fmt='%s')

if windowing_option:
    kwargs = {"U": U_range, "ve_ra": ve_ra_range, "ve_dec": ve_dec_range,
              "params": params}
else:
    mjd_range = mjd
    kwargs = {"U": U, "ve_ra": ve_ra, "ve_dec": ve_dec, "params": params}
###############################################################################

def effective_velocity_annual_bilby(
        xdata, mjd_stop, d, s, s2, k, k2, A, vism_ra, vism_ra2, vism_dec,
        vism_dec2, KIN, KOM, TAUEFAC, DNUEFAC, TAUESKEW, DNUESKEW, alpha, psi,
        psi2, R, R2, **kwargs):
    """
    Effective velocity thin screen model.
    Uses Eq. 4 from Rickett et al. (2014) for anisotropy coefficients.

        ydata: arc curvature
    """
    # Define the initial parameters
    params_ = dict(kwargs['params'])
    params_['d'] = d
    params_['s'] = s
    params_['vism_ra'] = vism_ra
    params_['vism_dec'] = vism_dec
    if psi != 500:
        params_['psi'] = psi
        params_['R'] = R
    params_['KOM'] = KOM
    params_['KIN'] = KIN
    if psi2 != 500:
        params_['psi2'] = psi2
        params_['R2'] = R2
    params_['s2'] = s2
    params_['vism_ra2'] = vism_ra2
    params_['vism_dec2'] = vism_dec2
    true_anomaly = kwargs['U']
    vearth_ra = kwargs['ve_ra']
    vearth_dec = kwargs['ve_dec']

    mjd_stop = int(mjd_stop)
    mjd = xdata
    argsort1 = mjd < mjd_stop
    argsort2 = mjd > mjd_stop
    mjd1 = mjd[argsort1]
    mjd2 = mjd[argsort2]
    true_anomaly1 = true_anomaly[argsort1]
    true_anomaly2 = true_anomaly[argsort2]
    vearth_ra1 = vearth_ra[argsort1]
    vearth_ra2 = vearth_ra[argsort2]
    vearth_dec1 = vearth_dec[argsort1]
    vearth_dec2 = vearth_dec[argsort2]

    # Define some constants
    v_c = 299792.458  # km/s
    kmpkpc = 3.085677581e16
    secperyr = 86400*365.2425
    masrad = np.pi/(3600*180*1000)

    A1 = params_['A1']  # projected semi-major axis in lt-s
    PB = params_['PB']  # orbital period in days
    ECC = params_['ECC']  # orbital eccentricity
    OM = params_['OM'] * np.pi/180  # longitude of periastron rad
    INC = params_['KIN'] * np.pi/180  # inclination
    KOM = params_['KOM'] * np.pi/180  # longitude ascending node

    PMRA = params_['PMRA']  # proper motion in RA
    PMDEC = params_['PMDEC']  # proper motion in DEC

    # other parameters in lower-case
    s = params_['s']  # fractional screen distance
    d = params_['d']  # pulsar distance in kpc
    d_kmpkpc = d * kmpkpc  # distance in km

    pmra_v = PMRA * masrad * d_kmpkpc / secperyr
    pmdec_v = PMDEC * masrad * d_kmpkpc / secperyr

    if psi != 500:
        r = params_['R']  # axial ratio parameter, see Rickett Cordes 1998
        psi = params_['psi'] * np.pi / 180  # anisotropy angle
        cosa = np.cos(2 * psi)
        sina = np.sin(2 * psi)
        # quadratic coefficients
        a1 = (1 - r * cosa) / np.sqrt(1 - r**2)
        b1 = (1 + r * cosa) / np.sqrt(1 - r**2)
        c1 = -2 * r * sina / np.sqrt(1 - r**2)
    else:
        a1 = 1
        b1 = 1
        c1 = 0
    if psi2 != 500:
        r2 = params_['R2']  # axial ratio parameter, see Rickett Cordes 1998
        psi2 = params_['psi2'] * np.pi / 180  # anisotropy angle
        cosa2 = np.cos(2 * psi2)
        sina2 = np.sin(2 * psi2)
        # quadratic coefficients
        a2 = (1 - r2 * cosa2) / np.sqrt(1 - r2**2)
        b2 = (1 + r2 * cosa2) / np.sqrt(1 - r2**2)
        c2 = -2 * r2 * sina2 / np.sqrt(1 - r2**2)
    else:
        a2 = 1
        b2 = 1
        c2 = 0

    ## Calculate the first model ##
    # Calculate pulsar velocity aligned with the line of nodes (Vx) and
    #   perpendicular in the plane (Vy)
    omega1 = OM + params_['OMDOT']*np.pi/180*(mjd1-params_['T0'])/365.2425

    vp_01 = (2 * np.pi * A1 * v_c) / (np.sin(INC) * PB * 86400 *
                                     np.sqrt(1 - ECC**2))
    vp_x1 = -vp_01 * (ECC * np.sin(omega1) + np.sin(true_anomaly1 + omega1))
    vp_y1 = vp_01 * np.cos(INC) * (ECC * np.cos(omega1) + np.cos(true_anomaly1
                                                              + omega1))

    # Rotate pulsar velocity into RA/DEC
    vp_ra1 = np.sin(KOM) * vp_x1 + np.cos(KOM) * vp_y1
    vp_dec1 = np.cos(KOM) * vp_x1 - np.sin(KOM) * vp_y1

    # find total effective velocity in RA and DEC
    veff_ra1 = s * vearth_ra1 + (1 - s) * (vp_ra1 + pmra_v) - vism_ra
    veff_dec1 = s * vearth_dec1 + (1 - s) * (vp_dec1 + pmdec_v) - vism_dec

    veff1 = np.sqrt(a1 * veff_dec1**2 + b1 * veff_ra1**2 + c1 * veff_ra1 *
                   veff_dec1)
    model1 = veff1 / s
    model1 = np.float64(model1)

    ## Calculate the second model ##
    omega2 = OM + params_['OMDOT']*np.pi/180*(mjd2-params_['T0'])/365.2425

    vp_02 = (2 * np.pi * A1 * v_c) / (np.sin(INC) * PB * 86400 *
                                     np.sqrt(1 - ECC**2))
    vp_x2 = -vp_02 * (ECC * np.sin(omega2) + np.sin(true_anomaly2 + omega2))
    vp_y2 = vp_02 * np.cos(INC) * (ECC * np.cos(omega2) + np.cos(true_anomaly2
                                                              + omega2))

    # Rotate pulsar velocity into RA/DEC
    vp_ra2 = np.sin(KOM) * vp_x2 + np.cos(KOM) * vp_y2
    vp_dec2 = np.cos(KOM) * vp_x2 - np.sin(KOM) * vp_y2

    # find total effective velocity in RA and DEC
    veff_ra2 = s2 * vearth_ra2 + (1 - s2) * (vp_ra2 + pmra_v) - vism_ra2
    veff_dec2 = s2 * vearth_dec2 + (1 - s2) * (vp_dec2 + pmdec_v) - vism_dec2

    veff2 = np.sqrt(a2 * veff_dec2**2 + b2 * veff_ra2**2 + c2 * veff_ra2 *
                   veff_dec2)

    model2 = veff2 / s2
    model2 = np.float64(model2)

    model = np.concatenate((model1, model2))
    return model


###############################################################################
# Here you can change the delta functions or the bounds of the priors
if Anisotropy_Option:
    outdir += '/Anisotropic/' + str(label)
    try:
        os.mkdir(outdir)
    except OSError as error:
        print(error)
    if multiple_s_option:
        mjd_stop = bilby.core.prior.Uniform(min(mjd)-10, max(mjd)+10,
                                            'mjd_stop')
        # mjd_stop = bilby.core.prior.analytical.DeltaFunction(np.median(mjd),
        #                                                      'mjd_stop')
        s2 = bilby.core.prior.Uniform(0.001, 1, 's2')
        k2 = bilby.core.prior.Uniform(0, 10, 'k2')
        vism_ra2 = bilby.core.prior.Uniform(-300, 300, 'vism_ra2')
        vism_dec2 = bilby.core.prior.Uniform(-300, 300, 'vism_dec2')
        psi2 = bilby.core.prior.Uniform(0, 180, 'psi2')
        R2 = bilby.core.prior.Uniform(0, 1, 'R2')
    else:
        mjd_stop = bilby.core.prior.analytical.DeltaFunction(mjd0, 'mjd_stop')
        s2 = bilby.core.prior.analytical.DeltaFunction(0.5, 's2')
        k2 = bilby.core.prior.analytical.DeltaFunction(1, 'k2')
        vism_ra2 = bilby.core.prior.analytical.DeltaFunction(0, 'vism_ra2')
        vism_dec2 = bilby.core.prior.analytical.DeltaFunction(0, 'vism_dec2')
        psi2 = bilby.core.prior.analytical.DeltaFunction(500, 'psi2')
        R2 = bilby.core.prior.analytical.DeltaFunction(0.5, 'R2')

    if distance_option:
        d = bilby.core.prior.Uniform(0, 10, 'd')
    else:
        d = bilby.core.prior.analytical.DeltaFunction(input_D, 'd')
    if s_option:
        s = bilby.core.prior.Uniform(0.001, 0.95, 's')
    else:
        s = bilby.core.prior.analytical.DeltaFunction(0.711, 's')
    if k_option:
        k = bilby.core.prior.Uniform(0, 10, 'k')
    else:
        k = bilby.core.prior.analytical.DeltaFunction(1, 'k')
    if A_option:
        A = bilby.core.prior.Uniform(0, 1e5, 'A')
    else:
        # A = bilby.core.prior.Uniform(input_Aiss1, input_Aiss2, 'A')
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
        R = bilby.core.prior.Uniform(0, 1, 'R')
    else:
        R = bilby.core.prior.analytical.DeltaFunction(0.5, 'R')
# xdata, mjd_stop, d, s, s2, k, k2, A, vism_ra, vism_ra2, vism_dec,
# vism_dec2, KIN, KOM, TAUEFAC, DNUEFAC, TAUESKEW, DNUESKEW, alpha, psi,
# psi2, R, R2, **kwargs):

    priors = dict(mjd_stop=mjd_stop, d=d, s=s, s2=s2, k=k, k2=k2, A=A,
                  vism_ra=vism_ra, vism_ra2=vism_ra2, vism_dec=vism_dec,
                  vism_dec2=vism_dec2, KIN=KIN, KOM=KOM, alpha=alpha, psi=psi,
                  psi2=psi2, R=R, R2=R2, TAUEFAC=TAUEFAC, DNUEFAC=DNUEFAC,
                  TAUESKEW=TAUESKEW, DNUESKEW=DNUESKEW)

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
    if multiple_s_option:
        mjd_stop = bilby.core.prior.Uniform(min(mjd), max(mjd), 'mjd_stop')
        s2 = bilby.core.prior.Uniform(0.001, 0.9, 's2')
        k2 = bilby.core.prior.Uniform(0, 10, 'k2')
        vism_ra2 = bilby.core.prior.Uniform(-300, 300, 'vism_ra2')
        vism_dec2 = bilby.core.prior.Uniform(-300, 300, 'vism_dec2')
        psi2 = bilby.core.prior.Uniform(0, 180, 'psi2')
        R2 = bilby.core.prior.Uniform(0, 1, 'R2')
    else:
        mjd_stop = bilby.core.prior.analytical.DeltaFunction(mjd0, 'mjd_stop')
        s2 = bilby.core.prior.analytical.DeltaFunction(0.5, 's2')
        k2 = bilby.core.prior.analytical.DeltaFunction(1, 'k2')
        vism_ra2 = bilby.core.prior.analytical.DeltaFunction(0, 'vism_ra2')
        vism_dec2 = bilby.core.prior.analytical.DeltaFunction(0, 'vism_dec2')
        psi2 = bilby.core.prior.analytical.DeltaFunction(500, 'psi2')
        R2 = bilby.core.prior.analytical.DeltaFunction(0.5, 'R2')
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

    priors = dict(mjd_stop=mjd_stop, d=d, s=s, s2=s2, k=k, k2=k2, A=A,
                  vism_ra=vism_ra, vism_ra2=vism_ra2, vism_dec=vism_dec,
                  vism_dec2=vism_dec2, KIN=KIN, KOM=KOM, alpha=alpha, psi=psi,
                  psi2=psi2, R=R, R2=R2, TAUEFAC=TAUEFAC, DNUEFAC=DNUEFAC,
                  TAUESKEW=TAUESKEW, DNUESKEW=DNUESKEW)

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
mjd_stop_result = result.posterior['mjd_stop'][NUM]
mjd_stoperr_result = np.std(result.posterior['mjd_stop'].values)
params.add('mjd_stop', value=mjd_stop_result, vary=False)
params.add('mjd_stoperr', value=mjd_stoperr_result, vary=False)
d_result = result.posterior['d'][NUM]
derr_result = np.std(result.posterior['d'].values)
params.add('d', value=d_result, vary=False)
params.add('derr', value=derr_result, vary=False)
s_result = result.posterior['s'][NUM]
serr_result = np.std(result.posterior['s'].values)
params.add('s', value=s_result, vary=False)
params.add('serr', value=serr_result, vary=False)
s2_result = result.posterior['s2'][NUM]
s2err_result = np.std(result.posterior['s2'].values)
params.add('s2', value=s2_result, vary=False)
params.add('s2err', value=s2err_result, vary=False)
k_result = result.posterior['k'][NUM]
kerr_result = np.std(result.posterior['k'].values)
params.add('k', value=k_result, vary=False)
params.add('kerr', value=kerr_result, vary=False)
k2_result = result.posterior['k2'][NUM]
k2err_result = np.std(result.posterior['k2'].values)
params.add('k2', value=k2_result, vary=False)
params.add('k2err', value=k2err_result, vary=False)
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
vism_ra2_result = result.posterior['vism_ra2'][NUM]
vism_ra2err_result = np.std(result.posterior['vism_ra2'].values)
params.add('vism_ra2', value=vism_ra2_result, vary=False)
params.add('vism_raerr2', value=vism_ra2err_result, vary=False)
vism_dec2_result = result.posterior['vism_dec2'][NUM]
vism_dec2err_result = np.std(result.posterior['vism_dec2'].values)
params.add('vism_dec2', value=vism_dec2_result, vary=False)
params.add('vism_decerr2', value=vism_dec2err_result, vary=False)
KIN_result = result.posterior['KIN'][NUM]
KINerr_result = np.std(result.posterior['KIN'].values)
params.add('KIN', value=KIN_result, vary=False)
params.add('KINrr', value=KINerr_result, vary=False)
KOM_result = result.posterior['KOM'][NUM]
KOMerr_result = np.std(result.posterior['KOM'].values)
params.add('KOM', value=KOM_result, vary=False)
params.add('KOMerr', value=KOMerr_result, vary=False)
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
    psi2_result = result.posterior['psi2'][NUM]
    psi2err_result = np.std(result.posterior['psi2'].values)
    params.add('psi2', value=psi2_result, vary=False)
    params.add('psi2err', value=psi2err_result, vary=False)
    R2_result = result.posterior['R2'][NUM]
    R2err_result = np.std(result.posterior['R2'].values)
    params.add('R2', value=R2_result, vary=False)
    params.add('R2err', value=R2err_result, vary=False)
else:
    psi_result = 500
    psierr_result = 0
    R_result = 0.5
    Rerr_result = 0
    psi2_result = 500
    psi2err_result = 0
    R2_result = 0.5
    R2err_result = 0
# Aiss will change depending on s and k which change in our model
mjd_stop = int(mjd_stop_result)
argsort1 = mjd < mjd_stop
argsort2 = mjd > mjd_stop
mjd1 = mjd[argsort1]
mjd2 = mjd[argsort2]
freqGHz1 = freqGHz[argsort1]
freqGHz2 = freqGHz[argsort2]
dnu1 = dnu[argsort1]
dnu2 = dnu[argsort2]
tau1 = tau[argsort1]
tau2 = tau[argsort2]
dnuerr1 = dnuerr[argsort1]
dnuerr2 = dnuerr[argsort2]
tauerr1 = tauerr[argsort1]
tauerr2 = tauerr[argsort2]

if not A_option:
    Aiss1 = k_result * input_Aiss * np.sqrt((2*(1-s_result))/(s_result))
    Aiss2 = k2_result * input_Aiss * np.sqrt((2*(1-s2_result))/(s2_result))
else:
    Aiss1 = A_result * np.sqrt((2*(1-s_result))/(s_result))
    Aiss2 = A_result * np.sqrt((2*(1-s2_result))/(s2_result))
New_viss1 = Aiss1 * (np.sqrt(d_result*dnu1))/(freqGHz1*tau1)
New_viss2 = Aiss2 * (np.sqrt(d_result*dnu2))/(freqGHz2*tau2)

if AltNoise_option:
    New_visserr1 = New_viss1 * np.sqrt((derr_result/(2*d_result))**2 +
                                       (dnuerr1/(2*dnu1))**2 + (-tauerr1/tau1
                                                                  )**2)
    New_visserr2 = New_viss2 * np.sqrt((derr_result/(2*d_result))**2 +
                                       (dnuerr2/(2*dnu2))**2 + (-tauerr2/tau2
                                                                  )**2)
    New_visserr1 = np.sqrt((New_visserr1 * 10**TAUEFAC_result)**2 +
                          ((10**TAUESKEW_result)*(freqGHz1/1)**(alpha_result)
                            )**2)
    New_visserr2 = np.sqrt((New_visserr2 * 10**TAUEFAC_result)**2 +
                          ((10**TAUESKEW_result)*(freqGHz2/1)**(alpha_result)
                            )**2)
else:
    # Re-calculate the errors on dnu, tau, viss as well as the magnitude of viss
    New_dnuerr1 = np.sqrt((dnuerr1 * 10**DNUEFAC_result)**2 +
                         ((10**DNUESKEW_result) * (freqGHz1/1)**alpha_result)**2)
    New_dnuerr2 = np.sqrt((dnuerr2 * 10**DNUEFAC_result)**2 +
                         ((10**DNUESKEW_result) * (freqGHz2/1)**alpha_result)**2)
    New_tauerr1 = np.sqrt((tauerr1 * 10**TAUEFAC_result)**2 +
                         ((10**TAUESKEW_result) * (freqGHz1/1)**(alpha_result/2))**2)
    New_tauerr2 = np.sqrt((tauerr2 * 10**TAUEFAC_result)**2 +
                         ((10**TAUESKEW_result) * (freqGHz2/1)**(alpha_result/2))**2)
    New_visserr1 = New_viss1 * np.sqrt((derr_result/(2*d_result))**2 +
                                 (New_dnuerr1/(2*dnu1))**2 + (-New_tauerr1/tau1)**2)
    New_visserr2 = New_viss2 * np.sqrt((derr_result/(2*d_result))**2 +
                                 (New_dnuerr2/(2*dnu2))**2 + (-New_tauerr2/tau2)**2)

New_viss = np.concatenate((New_viss1, New_viss2))
New_visserr = np.concatenate((New_visserr1, New_visserr2))

# Determine mjd_year and mjd_annual and mjd_doy
mjd_year = Time(mjd, format='mjd').byear
mjd_annual = mjd % 365.2425
# Finding the true 'day of year' jan 1st = 1
rawtime = Time(mjd, format='mjd').yday
total_day = []
for i in range(0, len(rawtime)):
    days = float(rawtime[i][5:8])
    hours = float(rawtime[i][9:11])
    minuets = float(rawtime[i][12:14])
    seconds = float(rawtime[i][15:21])
    total_day.append(days + (hours/24) + (minuets/1440) + (seconds/86400))
mjd_doy = np.asarray(total_day)
# Loading in model data across min to max mjd, 10000 steps ...
Model_mjd = np.loadtxt(datadir + 'Model_mjdData.txt', dtype='float')
Model_phase = np.loadtxt(datadir + 'Model_phaseData.txt', dtype='float')
Model_U = np.loadtxt(datadir + 'Model_UData.txt', dtype='float')
Model_vearth_ra = np.loadtxt(datadir + 'Model_vearth_raData.txt',
                              dtype='float')
Model_vearth_dec = np.loadtxt(datadir + 'Model_vearth_decData.txt',
                              dtype='float')
# Determine mjd_year and mjd_annual and mjd_doy
Model_mjd_year = Time(Model_mjd, format='mjd').byear
Model_mjd_annual = Model_mjd % 365.2425
# Finding the true 'day of year' jan 1st = 1
Model_rawtime = Time(Model_mjd, format='mjd').yday
Model_total_day = []
for i in range(0, len(rawtime)):
    days = float(Model_rawtime[i][5:8])
    hours = float(Model_rawtime[i][9:11])
    minuets = float(Model_rawtime[i][12:14])
    seconds = float(Model_rawtime[i][15:21])
    Model_total_day.append(days + (hours/24) + (minuets/1440) +
                           (seconds/86400))
Model_mjd_doy = np.asarray(Model_total_day)
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
    mjd_range, mjd_stop_result, d_result, s_result, s2_result, k_result,
    k2_result, A_result, vism_ra_result, vism_ra2_result, vism_dec_result,
    vism_dec2_result, KIN_result, KOM_result, TAUEFAC_result, DNUEFAC_result,
    TAUESKEW_result, DNUESKEW_result, alpha_result, psi_result, psi2_result,
    R_result, R2_result, **kwargs)

params.add('A1', value=0, vary=False)

orig_veff_VE = effective_velocity_annual_bilby(
    mjd_range, mjd_stop_result, d_result, s_result, s2_result, k_result,
    k2_result, A_result, 0, 0, 0, 0, KIN_result, KOM_result, TAUEFAC_result,
    DNUEFAC_result, TAUESKEW_result, DNUESKEW_result, alpha_result, psi_result,
    psi2_result, R_result, R2_result, **kwargs)

params.add('A1', value=A1_temp, vary=False)
if windowing_option:
    kwargs = {"U": U_range, "ve_ra": zeros, "ve_dec": zeros,
          "params": params}
else:
    kwargs = {"U": U, "ve_ra": zeros, "ve_dec": zeros, "params": params}

orig_veff_VP = effective_velocity_annual_bilby(
    mjd_range, mjd_stop_result, d_result, s_result, s2_result, k_result,
    k2_result, A_result, 0, 0, 0, 0, KIN_result, KOM_result, TAUEFAC_result,
    DNUEFAC_result, TAUESKEW_result, DNUESKEW_result, alpha_result, psi_result,
    psi2_result, R_result, R2_result, **kwargs)

params.add('A1', value=0, vary=False)

orig_veff_IISM = effective_velocity_annual_bilby(
    mjd_range, mjd_stop_result, d_result, s_result, s2_result, k_result,
    k2_result, A_result, vism_ra_result, vism_ra2_result, vism_dec_result,
    vism_dec2_result, KIN_result, KOM_result, TAUEFAC_result, DNUEFAC_result,
    TAUESKEW_result, DNUESKEW_result, alpha_result, psi_result, psi2_result,
    R_result, R2_result, **kwargs)

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
    Model_mjd, mjd_stop_result, d_result, s_result, s2_result, k_result,
    k2_result, A_result, vism_ra_result, vism_ra2_result, vism_dec_result,
    vism_dec2_result, KIN_result, KOM_result, TAUEFAC_result, DNUEFAC_result,
    TAUESKEW_result, DNUESKEW_result, alpha_result, psi_result, psi2_result,
    R_result, R2_result, **kwargs)

params.add('A1', value=0, vary=False)

Model_veff_VE = effective_velocity_annual_bilby(
    Model_mjd, mjd_stop_result, d_result, s_result, s2_result, k_result,
    k2_result, A_result, 0, 0, 0, 0, KIN_result, KOM_result, TAUEFAC_result,
    DNUEFAC_result, TAUESKEW_result, DNUESKEW_result, alpha_result, psi_result,
    psi2_result, R_result, R2_result, **kwargs)

params.add('A1', value=A1_temp, vary=False)
kwargs = {"U": Model_U, "ve_ra": Model_zeros, "ve_dec": Model_zeros,
          "params": params}

Model_veff_VP = effective_velocity_annual_bilby(
    Model_mjd, mjd_stop_result, d_result, s_result, s2_result, k_result,
    k2_result, A_result, 0, 0, 0, 0, KIN_result, KOM_result, TAUEFAC_result,
    DNUEFAC_result, TAUESKEW_result, DNUESKEW_result, alpha_result, psi_result,
    psi2_result, R_result, R2_result, **kwargs)

params.add('A1', value=0, vary=False)

Model_veff_IISM = effective_velocity_annual_bilby(
    Model_mjd, mjd_stop_result, d_result, s_result, s2_result, k_result,
    k2_result, A_result, vism_ra_result, vism_ra2_result, vism_dec_result,
    vism_dec2_result, KIN_result, KOM_result, TAUEFAC_result, DNUEFAC_result,
    TAUESKEW_result, DNUESKEW_result, alpha_result, psi_result, psi2_result,
    R_result, R2_result, **kwargs)

params.add('A1', value=A1_temp, vary=False)

# Determining the values of our measurements only effected by orb/annual phase
viss_VP_only = New_viss - veff_VE  # - veff_IISM
viss_VE_only = New_viss - veff_VP  # - veff_IISM
###############################################################################
# Here we begin creating the necessary diagnostic plots and save results


# Defining the argsort for our datasets
mjd_sort = np.argsort(mjd)
phase_sort = np.argsort(phase)
mjd_annual_sort = np.argsort(mjd_annual)
freq_sort = np.argsort(freqMHz)
Model_phase_sort = np.argsort(Model_phase)
Model_mjd_sort = np.argsort(Model_mjd)
Model_mjd_annual_sort = np.argsort(Model_mjd_annual)

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
ax1.plot(Model_phase[Model_phase_sort],
        Model_veff[Model_phase_sort], c='C3', alpha=0.1)
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
                                                       veff[phase_sort])), 3)))
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
plt.hist(d_scint1_data, bins=100, color='C3', alpha=0.6, density=True, label='Scintillation ' + str(int(scintdist1)) + ' +/- ' + str(int(round(scintdisterr1, -1))) + '  pc')
plt.xlabel("Distance to Double Pulsar (pc)")
plt.ylabel("Density")
ax.legend(fontsize='small', loc='upper right')
xl = plt.xlim()
plt.xlim(0, 1250)
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
