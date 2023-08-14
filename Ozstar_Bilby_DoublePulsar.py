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
from scintvelocityfunction import scint_velocity_alternate
###############################################################################


def effective_velocity_annual_bilby(
        xdata, d, s, k, vism_ra, vism_dec, KIN, KOM, TAUEFAC,
        DNUEFAC, TAUESKEW, DNUESKEW, alpha, psi, R, **kwargs):
    """
    Effective velocity thin screen model.
    Uses Eq. 4 from Rickett et al. (2014) for anisotropy coefficients.

        ydata: arc curvature
    """
    # Define the initial parameters
    params_ = dict(kwargs['params'])
    params_['d'] = d
    params_['s'] = s
    params_['kappa'] = k  # This does not effect our model, but our data
    params_['vism_ra'] = vism_ra
    params_['vism_dec'] = vism_dec
    if psi != 500:
        params_['psi'] = psi
        params_['R'] = R
    params_['KOM'] = KOM
    params_['KIN'] = KIN

    mjd = xdata
    true_anomaly = kwargs['U']
    vearth_ra = kwargs['ve_ra']
    vearth_dec = kwargs['ve_dec']
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

    omega = OM + params_['OMDOT']*np.pi/180*(mjd-params_['T0'])/365.2425

    PMRA = params_['PMRA']  # proper motion in RA
    PMDEC = params_['PMDEC']  # proper motion in DEC

    # other parameters in lower-case
    s = params_['s']  # fractional screen distance
    d = params_['d']  # pulsar distance in kpc
    d_kmpkpc = d * kmpkpc  # distance in km
    if psi != 500:
        r = params_['R']  # axial ratio parameter, see Rickett Cordes 1998
        psi = params_['psi'] * np.pi / 180  # anisotropy angle

    # Calculate pulsar velocity aligned with the line of nodes (Vx) and
    #   perpendicular in the plane (Vy)
    vp_0 = (2 * np.pi * A1 * v_c) / (np.sin(INC) * PB * 86400 *
                                     np.sqrt(1 - ECC**2))
    vp_x = -vp_0 * (ECC * np.sin(omega) + np.sin(true_anomaly + omega))
    vp_y = vp_0 * np.cos(INC) * (ECC * np.cos(omega) + np.cos(true_anomaly
                                                              + omega))

    pmra_v = PMRA * masrad * d_kmpkpc / secperyr
    pmdec_v = PMDEC * masrad * d_kmpkpc / secperyr

    # Rotate pulsar velocity into RA/DEC
    vp_ra = np.sin(KOM) * vp_x + np.cos(KOM) * vp_y
    vp_dec = np.cos(KOM) * vp_x - np.sin(KOM) * vp_y

    # find total effective velocity in RA and DEC
    veff_ra = s * vearth_ra + (1 - s) * (vp_ra + pmra_v) - vism_ra
    veff_dec = s * vearth_dec + (1 - s) * (vp_dec + pmdec_v) - vism_dec

    if psi != 500:
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

    veff = np.sqrt(a * veff_dec**2 + b * veff_ra**2 + c * veff_ra * veff_dec)
    model = veff / s
    model = np.float64(model)

    return model


###############################################################################
# Here you change change which parameters or anisotropy you wish to look at
options = {
    "distance": False,  # If False it will choose a delta function as prior
    "s": True,  # If true it will sample uniformly across parameter space
    "k": True,
    "vism_ra": True,
    "vism_dec": True,
    "KIN": False,
    "KOM": True,
    "psi": True,
    "R": True,
    "TAUEFAC": True,
    "DNUEFAC": True,
    "TAUESKEW": True,
    "DNUESKEW": True,
    "alpha": False,
    "Anisotropy_Option": True,
    "resume": False,  # Resumes a saved .json file output from bilby
    "sense": False,  # True = Flipped, False = not-flipped
}
# You can also change the number of nlive points or inclination model
nlive = 2000
input_KIN = 89.35  # Kramer et al. 2021, timing
input_KIN_err = 0.05  # timing
# Other values ... earlier timing 88.69 + 0.5 - 0.76 ... eclipse 89.3 +- 0.1
#  earlier scintillation 88.1 +- 0.5
###############################################################################
# Here we import data and define things that shouldn't need to be changed
distance_option = options['distance']
s_option = options['s']
k_option = options['k']
vism_ra_option = options['vism_ra']
vism_dec_option = options['vism_dec']
KIN_option = options['KIN']
KOM_option = options['KOM']
psi_option = options['psi']
R_option = options['R']
TAUEFAC_option = options['TAUEFAC']
DNUEFAC_option = options['DNUEFAC']
TAUESKEW_option = options['TAUESKEW']
DNUESKEW_option = options['DNUESKEW']
alpha_option = options['alpha']
Anisotropy_Option = options['Anisotropy_Option']
resume = options['resume']
sense = options['sense']  # True = Flipped, False = not-flipped
if sense:
    input_KIN = 180 - input_KIN
if Anisotropy_Option:
    Isotropic = False
else:
    Isotropic = True
if sense:
    sense_alt = False
else:
    sense_alt = True

labellist = {
    "Anisotropic": Anisotropy_Option,
    "Isotropic": Isotropic,
    "Flipped": sense,
    "Not-Flipped": sense_alt,
}

desktopdir = 'fred/oz002/jaskew/'
datadir = desktopdir + '0737_Project/Saved_Measurements/'
par_dir = desktopdir + 'ParFiles/'
psrname = 'J0737-3039A'
label = "_".join(filter(None, [key if value else None
                               for key, value in labellist.items()]))
wd0 = desktopdir + '0737_Project/'
outdir = wd0 + 'Modelling/'
pars = read_par(str(par_dir) + str(psrname) + '.par')
params = pars_to_params(pars)
viss = np.loadtxt(datadir + 'Full_VissData.txt', dtype='float')
visserr = np.loadtxt(datadir + 'Full_VisserrData.txt', dtype='float')
mjd = np.loadtxt(datadir + 'Full_MJDData.txt', dtype='float')
freq = np.loadtxt(datadir + 'Full_FreqData.txt', dtype='float')
dnu = np.loadtxt(datadir + 'Full_DnuData.txt', dtype='float')
dnuerr = np.loadtxt(datadir + 'Full_DnuerrData.txt', dtype='float')
tau = np.loadtxt(datadir + 'Full_TauData.txt', dtype='float')
tauerr = np.loadtxt(datadir + 'Full_TauerrData.txt', dtype='float')
phase = np.loadtxt(datadir + 'Full_PhaseData.txt', dtype='float')
U = np.loadtxt(datadir + 'Full_UData.txt', dtype='float')
ve_ra = np.loadtxt(datadir + 'Full_ve_raData.txt', dtype='float')
ve_dec = np.loadtxt(datadir + 'Full_ve_decData.txt', dtype='float')

kwargs = {"U": U, "ve_ra": ve_ra, "ve_dec": ve_dec, "params": params}
###############################################################################
# Here you can change the delta functions or the bounds of the priors
if Anisotropy_Option:
    outdir += '/Anisotropic/' + str(label)
    try:
        os.mkdir(outdir)
    except OSError as error:
        print(error)
    if distance_option:
        d = bilby.core.prior.Uniform(0, 10, 'd')
    else:
        d = bilby.core.prior.analytical.DeltaFunction(0.735, 'd')
    if s_option:
        s = bilby.core.prior.Uniform(0.001, 1, 's')
    else:
        s = bilby.core.prior.analytical.DeltaFunction(0.72, 's')
    if k_option:
        k = bilby.core.prior.Uniform(0, 10, 'k')
    else:
        k = bilby.core.prior.analytical.DeltaFunction(1, 'k')
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
    if psi_option:
        psi = bilby.core.prior.Uniform(0, 180, 'psi', boundary='periodic')
    else:
        psi = bilby.core.prior.analytical.DeltaFunction(90, 'psi')
    if R_option:
        R = bilby.core.prior.Uniform(0, 1, 'R')
    else:
        R = bilby.core.prior.analytical.DeltaFunction(0.5, 'R')

    priors = dict(d=d, s=s, k=k, vism_ra=vism_ra, vism_dec=vism_dec, psi=psi,
                  R=R, KIN=KIN, KOM=KOM, TAUEFAC=TAUEFAC, DNUEFAC=DNUEFAC,
                  TAUESKEW=TAUESKEW, DNUESKEW=DNUESKEW, alpha=alpha)

    likelihood = \
        VissGaussianLikelihood(
            x=mjd, y=viss, func=effective_velocity_annual_bilby,
            freq=freq, tau=tau, dnu=dnu, tauerr=tauerr, dnuerr=dnuerr,
            sigma=visserr, **kwargs)

else:
    outdir += '/Isotropic/' + str(label)
    try:
        os.mkdir(outdir)
    except OSError as error:
        print(error)
    if distance_option:
        d = bilby.core.prior.Uniform(0, 10, 'd')
    else:
        d = bilby.core.prior.analytical.DeltaFunction(0.735, 'd')
    if s_option:
        s = bilby.core.prior.Uniform(0.001, 1, 's')
    else:
        s = bilby.core.prior.analytical.DeltaFunction(0.72, 's')
    if k_option:
        k = bilby.core.prior.Uniform(0, 10, 'k')
    else:
        k = bilby.core.prior.analytical.DeltaFunction(1, 'k')
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

    priors = dict(d=d, s=s, k=k, vism_ra=vism_ra, vism_dec=vism_dec, psi=psi,
                  R=R, KIN=KIN, KOM=KOM, TAUEFAC=TAUEFAC, DNUEFAC=DNUEFAC,
                  TAUESKEW=TAUESKEW, DNUESKEW=DNUESKEW, alpha=alpha)

    likelihood = \
        VissGaussianLikelihood(
            x=mjd, y=viss, func=effective_velocity_annual_bilby, freq=freq,
            tau=tau, dnu=dnu, tauerr=tauerr, dnuerr=dnuerr, sigma=visserr,
            **kwargs)

# And run sampler
result = bilby.core.sampler.run_sampler(
        likelihood, priors=priors, sampler='dynesty', label=label,
        nlive=nlive, verbose=True, resume=resume,
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
TAUESKEW_result = result.posterior['TAUESKEW'][NUM]
TAUESKEWerr_result = np.std(result.posterior['TAUESKEW'].values)
DNUESKEW_result = result.posterior['DNUESKEW'][NUM]
DNUESKEWerr_result = np.std(result.posterior['DNUESKEW'].values)
TAUEFAC_result = result.posterior['TAUEFAC'][NUM]
TAUEFACerr_result = np.std(result.posterior['TAUEFAC'].values)
DNUEFAC_result = result.posterior['DNUEFAC'][NUM]
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

# Re-calculate the errors on dnu, tau, viss as well as the magnitude of viss
new_freq = freq / 1e3
New_dnuerr = np.sqrt((dnuerr * 10**DNUEFAC_result)**2 +
                     ((10**(float(DNUESKEW_result))) *
                      (new_freq/1)**alpha_result)**2)
New_tauerr = np.sqrt((tauerr * 10**TAUEFAC_result)**2 +
                     ((10**float(TAUESKEW_result)) *
                      (new_freq/1)**(alpha_result/2))**2)
New_viss, New_visserr = scint_velocity_alternate(params, dnu, tau, freq,
                                                 New_dnuerr, New_tauerr)

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
# Calculating new models to the plot across with our data
A1_temp = params['A1'].value
zeros = np.zeros(np.shape(mjd))
kwargs = {"U": U, "ve_ra": ve_ra, "ve_dec": ve_dec, "params": params}

veff = effective_velocity_annual_bilby(
    mjd, d_result, s_result, k_result, vism_ra_result, vism_dec_result,
    KIN_result, KOM_result, TAUEFAC_result, DNUEFAC_result, TAUESKEW_result,
    DNUESKEW_result, alpha_result, psi_result, R_result, **kwargs)

params.add('A1', value=0, vary=False)

veff_VE = effective_velocity_annual_bilby(
    mjd, d_result, s_result, k_result, 0, 0, KIN_result, KOM_result,
    TAUEFAC_result, DNUEFAC_result, TAUESKEW_result, DNUESKEW_result,
    alpha_result, psi_result, R_result, **kwargs)

params.add('A1', value=A1_temp, vary=False)
kwargs = {"U": U, "ve_ra": zeros, "ve_dec": zeros, "params": params}

veff_VP = effective_velocity_annual_bilby(
    mjd, d_result, s_result, k_result, 0, 0, KIN_result, KOM_result,
    TAUEFAC_result, DNUEFAC_result, TAUESKEW_result, DNUESKEW_result,
    alpha_result, psi_result, R_result, **kwargs)

params.add('A1', value=0, vary=False)

veff_IISM = effective_velocity_annual_bilby(
    mjd, d_result, s_result, k_result, vism_ra_result, vism_dec_result,
    KIN_result, KOM_result, TAUEFAC_result, DNUEFAC_result, TAUESKEW_result,
    DNUESKEW_result, alpha_result, psi_result, R_result, **kwargs)

params.add('A1', value=A1_temp, vary=False)

# Now we do the same thing across the 10000 model space
kwargs = {"U": Model_U, "ve_ra": Model_vearth_ra, "ve_dec": Model_vearth_dec,
          "params": params}
Model_zeros = np.zeros(np.shape(Model_mjd))

Model_veff = effective_velocity_annual_bilby(
    Model_mjd, d_result, s_result, k_result, vism_ra_result, vism_dec_result,
    KIN_result, KOM_result, TAUEFAC_result, DNUEFAC_result, TAUESKEW_result,
    DNUESKEW_result, alpha_result, psi_result, R_result, **kwargs)

params.add('A1', value=0, vary=False)

Model_veff_VE = effective_velocity_annual_bilby(
    Model_mjd, d_result, s_result, k_result, 0, 0, KIN_result, KOM_result,
    TAUEFAC_result, DNUEFAC_result, TAUESKEW_result, DNUESKEW_result,
    alpha_result, psi_result, R_result, **kwargs)

params.add('A1', value=A1_temp, vary=False)
kwargs = {"U": Model_U, "ve_ra": Model_zeros, "ve_dec": Model_zeros,
          "params": params}

Model_veff_VP = effective_velocity_annual_bilby(
    Model_mjd, d_result, s_result, k_result, 0, 0, KIN_result, KOM_result,
    TAUEFAC_result, DNUEFAC_result, TAUESKEW_result, DNUESKEW_result,
    alpha_result, psi_result, R_result, **kwargs)

params.add('A1', value=0, vary=False)

Model_veff_IISM = effective_velocity_annual_bilby(
    Model_mjd, d_result, s_result, k_result, vism_ra_result, vism_dec_result,
    KIN_result, KOM_result, TAUEFAC_result, DNUEFAC_result, TAUESKEW_result,
    DNUESKEW_result, alpha_result, psi_result, R_result, **kwargs)

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
freq_sort = np.argsort(freq)
Model_phase_sort = np.argsort(Model_phase)
Model_mjd_sort = np.argsort(Model_mjd)
Model_mjd_annual_sort = np.argsort(Model_mjd_annual)

Size = 80*np.pi  # Determines the size of the datapoints used
font = {'size': 28}
matplotlib.rc('font', **font)

# FIGURE 1: Year against corrected data with model
fig = plt.figure(figsize=(20, 10))
fig.subplots_adjust(hspace=0.5, wspace=0.5)
ax = fig.add_subplot(1, 1, 1)
ax.errorbar(mjd_year[mjd_sort], New_viss[mjd_sort],
            yerr=New_visserr[mjd_sort],
            fmt='o', ecolor='k', elinewidth=2, capsize=3, alpha=0.55)
ax.plot(Model_mjd_year[Model_mjd_sort], Model_veff[Model_mjd_sort], c='k',
        alpha=0.2)
xl = plt.xlim()
ax.set_xlabel('Year')
ax.set_ylabel(r'Scintillation Velocity (km$\,$s$^{-1}$)')
ax.set_xlim(xl)
plt.tight_layout()
plt.savefig(str(outdir) + "/NewViss+Model_year.png", dpi=400)
plt.savefig(str(outdir) + "/NewViss+Model_year.pdf", dpi=400)
plt.show()
plt.close()

# # Phase
# phase_sort = np.argsort(phase)
fig = plt.figure(figsize=(20, 10))
fig.subplots_adjust(hspace=0.5, wspace=0.5)
#
ax = fig.add_subplot(1, 1, 1)
ax.errorbar(phase[phase_sort],
            viss_VP_only[phase_sort],
            yerr=New_visserr[phase_sort],
            fmt='o', ecolor='k', elinewidth=2, capsize=3, alpha=0.55)
ax.plot(Model_phase[Model_phase_sort],
        Model_veff_VP[Model_phase_sort], c='k', alpha=0.2)
xl = plt.xlim()
ax.set_xlabel('Orbital Phase (degrees)')
ax.set_ylabel(r'Binary Scintillation Velocity (km$\,$s$^{-1}$)')
ax.set_xlim(xl)
plt.tight_layout()
plt.savefig(str(outdir) + "/NewViss+Model_phase.png", dpi=400)
plt.savefig(str(outdir) + "/NewViss+Model_phase.pdf", dpi=400)
plt.show()
plt.close()

# # Annual

fig = plt.figure(figsize=(20, 10))
fig.subplots_adjust(hspace=0.5, wspace=0.5)
#
ax = fig.add_subplot(1, 1, 1)
ax.errorbar(mjd_annual[mjd_annual_sort],
            viss_VE_only[mjd_annual_sort],
            yerr=New_visserr[mjd_annual_sort], fmt='o', ecolor='k',
            elinewidth=2, capsize=3, alpha=0.55)
ax.plot(Model_mjd_annual[Model_mjd_annual_sort],
        Model_veff_VE[Model_mjd_annual_sort], c='k', alpha=0.2)
xl = plt.xlim()
ax.set_xlabel('Annual Phase (days)')
ax.set_ylabel(r'Annual Scintillation Velocity (km$\,$s$^{-1}$)')
ax.set_xlim(xl)
plt.tight_layout()
plt.savefig(str(outdir) + "/NewViss+Model_annual.png", dpi=400)
plt.savefig(str(outdir) + "/NewViss+Model_annual.pdf", dpi=400)
plt.show()
plt.close()

# RESIDUALS
# New_viss, New_visserr
# Phase
phase_sort = np.argsort(phase)
fig = plt.figure(figsize=(20, 10))
fig.subplots_adjust(hspace=0.5, wspace=0.5)
ax = fig.add_subplot(1, 1, 1)
ax.errorbar(phase[phase_sort],
            (New_viss[phase_sort] - veff[phase_sort]),
            yerr=New_visserr[phase_sort],
            fmt='o', ecolor='k', elinewidth=2, capsize=3, alpha=0.55)
ax.plot([np.min(phase), np.max(phase)], [0, 0], c='C3')
xl = plt.xlim()
ax.set_xlabel('Orbital Phase (degrees)')
ax.set_ylabel(r'Scintillation Velocity (km$\,$s$^{-1}$)')
ax.set_xlim(xl)
plt.tight_layout()
plt.savefig(str(outdir) + "/NewViss+Residuals_phase.png", dpi=400)
plt.savefig(str(outdir) + "/NewViss+Residuals_phase.pdf", dpi=400)
plt.show()
plt.close()

# # Annual
fig = plt.figure(figsize=(20, 10))
fig.subplots_adjust(hspace=0.5, wspace=0.5)
ax = fig.add_subplot(1, 1, 1)
ax.errorbar(mjd_annual[mjd_annual_sort],
            (New_viss - veff)[mjd_annual_sort],
            yerr=New_visserr[mjd_annual_sort],
            fmt='o', ecolor='k', elinewidth=2, capsize=3, alpha=0.55)
ax.plot([np.min(mjd_annual), np.max(mjd_annual)], [0, 0], c='C3')
xl = plt.xlim()
ax.set_xlabel('Annual Phase (days)')
ax.set_ylabel(r'Scintillation Velocity (km$\,$s$^{-1}$)')
ax.set_xlim(xl)
plt.tight_layout()
plt.savefig(str(outdir) + "/NewViss+Residuals_annual.png", dpi=400)
plt.savefig(str(outdir) + "/NewViss+Residuals_annual.pdf", dpi=400)
plt.show()
plt.close()

# # Year
fig = plt.figure(figsize=(20, 10))
fig.subplots_adjust(hspace=0.5, wspace=0.5)
ax = fig.add_subplot(1, 1, 1)
ax.errorbar(mjd_year[mjd_sort], (New_viss - veff)[mjd_sort],
            yerr=New_visserr[mjd_sort],
            fmt='o', ecolor='k', elinewidth=2, capsize=3, alpha=0.55)
ax.plot([np.min(mjd_year), np.max(mjd_year)], [0, 0], c='C3')
xl = plt.xlim()
ax.set_xlabel('Year')
ax.set_ylabel(r'Scintillation Velocity (km$\,$s$^{-1}$)')
ax.set_xlim(xl)
plt.tight_layout()
plt.savefig(str(outdir) + "/NewViss+Residuals_year.png", dpi=400)
plt.savefig(str(outdir) + "/NewViss+Residuals_year.pdf", dpi=400)
plt.show()
plt.close()

# Frequency variations

fig = plt.figure(figsize=(20, 10))
fig.subplots_adjust(hspace=0.5, wspace=0.5)
ax = fig.add_subplot(1, 1, 1)
ax.errorbar(freq[freq_sort], dnu[freq_sort],
            yerr=New_dnuerr[freq_sort],
            fmt='o', ecolor='k', elinewidth=2, capsize=3, alpha=0.55)
xl = plt.xlim()
ax.set_xlabel('Frequency (MHz)')
ax.set_ylabel(r'Scintillation Bandwidth (MHz)')
ax.set_xlim(xl)
plt.tight_layout()
plt.savefig(str(outdir) + "/NewDnuFreq.png", dpi=400)
plt.savefig(str(outdir) + "/NewDnuFreq.pdf", dpi=400)
plt.show()
plt.close()

fig = plt.figure(figsize=(20, 10))
fig.subplots_adjust(hspace=0.5, wspace=0.5)
ax = fig.add_subplot(1, 1, 1)
ax.errorbar(freq[freq_sort], tau[freq_sort],
            yerr=New_tauerr[freq_sort],
            fmt='o', ecolor='k', elinewidth=2, capsize=3, alpha=0.55)
xl = plt.xlim()
ax.set_xlabel('Frequency (MHz)')
ax.set_ylabel(r'Scintillation Timescale (s)')
ax.set_xlim(xl)
plt.tight_layout()
plt.savefig(str(outdir) + "/NewTauFreq.png", dpi=400)
plt.savefig(str(outdir) + "/NewTauFreq.pdf", dpi=400)
plt.show()
plt.close()

###############################################################################
# Collecting a result.csv file
ResultsFileLoc = outdir + "/" + str(label) + "_Results.csv"
ResultsFileLoc2 = wd0 + "Modelling/" + str(label) + "_Results.csv"

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

if distance_option:

    d_timing = 465
    d_timing_err = 134

    d_VLBI = 770
    d_VLBI_err = 70

    d_weighted = 735
    d_weighted_err = 60

    d_timing_data = np.random.normal(loc=d_timing, scale=d_timing_err,
                                     size=100000)
    d_VLBI_data = np.random.normal(loc=d_VLBI, scale=d_VLBI_err, size=100000)
    d_weighted_data = np.random.normal(loc=d_weighted, scale=d_weighted_err,
                                       size=100000)
    d_scint_data = np.random.normal(loc=d_result*1e3, scale=derr_result*1e3,
                                    size=100000)

    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(1, 1, 1)
    plt.hist(d_timing_data, bins=100, alpha=0.6, density=True, label='Timing')
    plt.hist(d_VLBI_data, bins=100, alpha=0.6, density=True, label='VLBI')
    plt.hist(d_weighted_data, bins=100, alpha=0.6, density=True,
             label='Weighted')
    plt.hist(d_scint_data, bins=100, alpha=0.6, density=True,
             label='Scintillation')
    plt.xlabel("Distance to Double Pulsar (pc)")
    plt.ylabel("Density")
    ax.legend()
    plt.savefig(str(outdir) + "/DistanceHistogram.png", dpi=400)
    plt.savefig(str(outdir) + "/DistanceHistogram.pdf", dpi=400)
    plt.show()
    plt.close()

###############################################################################
