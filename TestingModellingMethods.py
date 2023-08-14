#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 11:51:28 2023

@author: jacobaskew
"""
###############################################################################
import numpy as np
import matplotlib.pyplot as plt
from random import randrange
from scintools.scint_utils import write_results, read_results, read_par, \
        float_array_from_dict, get_ssb_delay, get_earth_velocity, \
        get_true_anomaly, scint_velocity, pars_to_params
from scintools.scint_models import effective_velocity_annual
import matplotlib
from VissGaussianLikelihood import VissGaussianLikelihood
###############################################################################


def effective_velocity_annual_bilby(xdata, D, s, vism_ra, vism_dec, k, KIN,
                                    KOM, TAUEFAC, DNUEFAC, TAUESKEW, DNUESKEW,
                                    alpha, **kwargs):
    """
    Effective velocity with annual and pulsar terms
        Note: Does NOT include IISM velocity, but returns veff in IISM frame
    """
    # Define the initial parameters
    params_ = dict(params)
    params_['d'] = D
    params_['s'] = s
    params_['kappa'] = k
    params_['KOM'] = KOM
    params_['KIN'] = KIN
    params_['vism_ra'] = vism_ra
    params_['vism_dec'] = vism_dec
    mjd = xdata
    mjd_sort = np.argsort(mjd)
    true_anomaly = U[mjd_sort]
    vearth_ra = ve_ra[mjd_sort]
    vearth_dec = ve_dec[mjd_sort]
    #
    # true_anomaly = U
    # vearth_ra, vearth_dec = ve_ra, ve_dec
    #
    # true_anomaly = get_true_anomaly(mjd, pars)
    # vearth_ra, vearth_dec = get_earth_velocity(mjd, pars['RAJ'],
    #                                             pars['DECJ'])
    # Define some constants
    v_c = 299792.458  # km/s
    kmpkpc = 3.085677581e16
    secperyr = 86400*365.2425
    masrad = np.pi/(3600*180*1000)

    # tempo2 parameters from par file in capitals
    if 'PB' in params_.keys():
        A1 = params_['A1']  # projected semi-major axis in lt-s
        PB = params_['PB']  # orbital period in days
        ECC = params_['ECC']  # orbital eccentricity
        OM = params_['OM'] * np.pi/180  # longitude of periastron rad
        if 'OMDOT' in params_.keys():
            if mjd is None:
                print('Warning, OMDOT present but no mjd for calculation')
                omega = OM
            else:
                omega = OM + \
                    params_['OMDOT']*np.pi/180*(mjd-params_['T0'])/365.2425
        else:
            omega = OM
        # Note: fifth Keplerian param T0 used in true anomaly calculation
        if 'KIN' in params_.keys():
            INC = params_['KIN']*np.pi/180  # inclination
        elif 'COSI' in params_.keys():
            INC = np.arccos(params_['COSI'])
        elif 'SINI' in params_.keys():
            INC = np.arcsin(params_['SINI'])
        else:
            print('Warning: inclination parameter (KIN, COSI, or SINI) ' +
                  'not found')

        if 'sense' in params_.keys():
            sense = params_['sense']
            if sense < 0.5:  # KIN < 90
                if INC > np.pi/2:
                    INC = np.pi - INC
            if sense >= 0.5:  # KIN > 90
                if INC < np.pi/2:
                    INC = np.pi - INC

        KOM = params_['KOM']*np.pi/180  # longitude ascending node

        # Calculate pulsar velocity aligned with the line of nodes (Vx) and
        #   perpendicular in the plane (Vy)
        vp_0 = (2 * np.pi * A1 * v_c) / (np.sin(INC) * PB * 86400 *
                                         np.sqrt(1 - ECC**2))
        vp_x = -vp_0 * (ECC * np.sin(omega) + np.sin(true_anomaly + omega))
        vp_y = vp_0 * np.cos(INC) * (ECC * np.cos(omega) + np.cos(true_anomaly
                                                                  + omega))
    else:
        vp_x = 0
        vp_y = 0

    if 'PMRA' in params_.keys():
        PMRA = params_['PMRA']  # proper motion in RA
        PMDEC = params_['PMDEC']  # proper motion in DEC
    else:
        PMRA = 0
        PMDEC = 0

    # other parameters in lower-case
    s = params_['s']  # fractional screen distance
    d = params_['d']  # pulsar distance in kpc
    kappa = params_['kappa']
    d_kmpkpc = d * kmpkpc  # distance in km

    pmra_v = PMRA * masrad * d_kmpkpc / secperyr
    pmdec_v = PMDEC * masrad * d_kmpkpc / secperyr

    # Rotate pulsar velocity into RA/DEC
    vp_ra = np.sin(KOM) * vp_x + np.cos(KOM) * vp_y
    vp_dec = np.cos(KOM) * vp_x - np.sin(KOM) * vp_y

    # find total effective velocity in RA and DEC
    veff_ra = s * vearth_ra + (1 - s) * (vp_ra + pmra_v) - vism_ra
    veff_dec = s * vearth_dec + (1 - s) * (vp_dec + pmdec_v) - vism_dec

    # coefficient to match model with data
    coeff = 1 / np.sqrt(2 * d * (1 - s) / s)

    veff = kappa * (np.sqrt(veff_dec**2 + veff_ra**2))
    model = coeff * veff / s
    model = np.float64(model)

    return model


###############################################################################


def effective_velocity_annual_anisotropy_bilby(xdata, D, s, k, vism_ra,
                                               vism_dec, psi, R, KIN, KOM,
                                               TAUEFAC, DNUEFAC, TAUESKEW,
                                               DNUESKEW, alpha, **kwargs):
    """
    Effective velocity thin screen model.
    Uses Eq. 4 from Rickett et al. (2014) for anisotropy coefficients.

        ydata: arc curvature
    """

    # Define the initial parameters
    params_ = dict(params)
    params_['d'] = D
    params_['s'] = s
    params_['kappa'] = k
    params_['vism_ra'] = vism_ra
    params_['vism_dec'] = vism_dec
    params_['psi'] = psi
    params_['R'] = R
    params_['KOM'] = KOM
    params_['KIN'] = KIN

    mjd = xdata
    mjd_sort = np.argsort(mjd)
    mjd = mjd[mjd_sort]
    true_anomaly = U[mjd_sort]
    vearth_ra = ve_ra[mjd_sort]
    vearth_dec = ve_dec[mjd_sort]

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

    r = params_['R']  # axial ratio parameter, some complicated relationship
    psi = params_['psi'] * np.pi / 180  # anisotropy angle
    kappa = params_['kappa']

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

    cosa = np.cos(2 * psi)
    sina = np.sin(2 * psi)

    # quadratic coefficients
    a = (1 - r * cosa) / np.sqrt(1 - r**2)
    b = (1 + r * cosa) / np.sqrt(1 - r**2)
    c = -2 * r * sina / np.sqrt(1 - r**2)

    # coefficient to match model with data
    coeff = 1 / np.sqrt(2 * d * (1 - s) / s)

    veff = kappa * (np.sqrt(a*veff_dec**2 + b*veff_ra**2 +
                            c*veff_ra*veff_dec))
    model = coeff * veff / s
    model = np.float64(model)

    return model


###############################################################################

viss = np.loadtxt('/Users/jacobaskew/Desktop/Full_VissData.txt', dtype='float')
visserr = np.loadtxt('/Users/jacobaskew/Desktop/Full_VisserrData.txt',
                     dtype='float')
mjd = np.loadtxt('/Users/jacobaskew/Desktop/Full_MJDData.txt', dtype='float')
freq = np.loadtxt('/Users/jacobaskew/Desktop/Full_FreqData.txt', dtype='float')
dnu = np.loadtxt('/Users/jacobaskew/Desktop/Full_DnuData.txt', dtype='float')
dnuerr = np.loadtxt('/Users/jacobaskew/Desktop/Full_DnuerrData.txt',
                    dtype='float')
tau = np.loadtxt('/Users/jacobaskew/Desktop/Full_TauData.txt', dtype='float')
tauerr = np.loadtxt('/Users/jacobaskew/Desktop/Full_TauerrData.txt',
                    dtype='float')
phase = np.loadtxt('/Users/jacobaskew/Desktop/Full_PhaseData.txt',
                   dtype='float')

par_dir = '/Users/jacobaskew/Desktop/Swinburne_Daniel/ParFiles/'
psrname = 'J0737-3039A'
pars = read_par(str(par_dir) + str(psrname) + '.par')
params = pars_to_params(pars)
# U = get_true_anomaly(mjd, pars)
# ve_ra, ve_dec = get_earth_velocity(mjd, pars['RAJ'], pars['DECJ'])

phase_sort = np.argsort(phase)
mjd_model = np.linspace(np.min(mjd), np.max(mjd), 2000)
ssb_delays = get_ssb_delay(mjd_model, pars['RAJ'], pars['DECJ'])
mjd_model += np.divide(ssb_delays, 86400)  # add ssb delay
"""
Model Viss
"""
print('Getting Earth velocity')
ve_ra, ve_dec = get_earth_velocity(mjd_model, pars['RAJ'],
                                                       pars['DECJ'])
print('Getting true anomaly')
U = get_true_anomaly(mjd_model, pars)

ve_ra = ve_ra.squeeze()
ve_dec = ve_dec.squeeze()

om = pars['OM'] + pars['OMDOT']*(mjd_model - pars['T0'])/365.2425
# compute orbital phase
phase_model = U*180/np.pi + om
phase_model = phase_model % 360
phase_model_sort = np.argsort(phase_model)

veff = effective_velocity_annual_anisotropy_bilby(mjd_model, D=0.735, s=0.72,
                                                  k=1, vism_ra=-20,
                                                  vism_dec=20,
                                                  psi=60, R=0.5, KIN=89.35,
                                                  KOM=65, TAUEFAC=0, DNUEFAC=0,
                                                  TAUESKEW=-100, DNUESKEW=-100,
                                                  alpha=4,
                                                  kwargs=(U, ve_ra, ve_dec))
isotropic_veff = effective_velocity_annual_bilby(mjd_model, D=0.735, s=0.72,
                                                 vism_ra=-20, vism_dec=20, k=1,
                                                 KIN=89.35, KOM=65, TAUEFAC=0,
                                                 DNUEFAC=0, TAUESKEW=-100,
                                                 DNUESKEW=-100, alpha=4,
                                                 kwargs=(U, ve_ra, ve_dec))
###############################################################################
Size = 80*np.pi  # Determines the size of the datapoints used
font = {'size': 14}
matplotlib.rc('font', **font)

plt.plot(phase_model[phase_model_sort], veff[phase_model_sort], alpha=0.6)
plt.plot(phase_model[phase_model_sort], isotropic_veff[phase_model_sort],
         alpha=0.6)
plt.errorbar(phase[phase_sort], viss[phase_sort], yerr=visserr[phase_sort],
             fmt='o', alpha=1)
plt.ylabel(r"Scintillation Velocity ($km\,s^{-1}$)")
plt.xlabel("Orbital Phase (degrees)")
plt.show()
plt.close()

# phase_sort = np.argsort(phase)
# freq_sort = np.argsort(freq)
# plt.errorbar(phase[phase_sort], viss[phase_sort], yerr=visserr[phase_sort],
#              fmt='o', c='C0', ecolor='k', alpha=0.4, capsize=1, elinewidth=1)
# plt.ylabel(r"Scintillation Velocity ($km\,s^{-1}$)")
# plt.xlabel("Orbital Phase (degrees)")
# plt.title("Original Data")
# plt.show()
# plt.close()

# DNUESKEW_range = np.linspace(-2, 2, 100)
# TAUESKEW_range = np.linspace(-2, 2, 100)
# DNUEFAC_range = np.linspace(-2, 2, 100)
# TAUEFAC_range = np.linspace(-2, 2, 100)

# for i in range(0, 100):

#     # DNUESKEW = -100
#     TAUESKEW = -100
#     DNUEFAC = 0
#     TAUEFAC = 0
#     # DNUESKEW = randrange(-2, 2) * (np.random.rand(1) - 0.5) * 4
#     # TAUESKEW = randrange(-2, 2) * (np.random.rand(1) - 0.5) * 4
#     # DNUEFAC = randrange(-2, 2) * (np.random.rand(1) - 0.5) * 4
#     # TAUEFAC = randrange(-2, 2) * (np.random.rand(1) - 0.5) * 4
#     DNUESKEW = DNUESKEW_range[i]
#     # TAUESKEW = TAUESKEW_range[i]
#     # DNUEFAC = DNUEFAC_range[i]
#     # TAUEFAC = TAUEFAC_range[i]
#     alpha = 4
#     new_freq = freq / 1e3
#     new_dnuerr = np.sqrt((dnuerr * 10**DNUEFAC)**2 +
#                          ((10**DNUESKEW)*(new_freq/1)**alpha)**2)
#     new_tauerr = np.sqrt((tauerr * 10**TAUEFAC)**2 +
#                          ((10**TAUESKEW)*(new_freq/1)**(alpha/2))**2)
#     a = 2.78*10**4
#     d = 0.735
#     d_err = 0
#     s = 0.72
#     s_err = 0
#     coeff = a * np.sqrt(2 * d * (1 - s) / s)  # thin screen coefficient
#     coeff_err = (dnu / s) * ((1 - s) * d_err**2 / (2 * d) +
#                              (d * s_err**2 / (2 * s**2 * (1 - s))))
#     new_viss = coeff * np.sqrt(dnu) / (new_freq * tau)
#     new_viss_err = (1 / (new_freq * tau)) * \
#         np.sqrt(coeff**2 * ((new_dnuerr**2 / (4 * dnu)) +
#                             (dnu * new_tauerr**2 / tau**2)) + coeff_err)

#     plt.errorbar(phase[phase_sort], new_viss[phase_sort], yerr=new_viss_err[phase_sort],
#                  fmt='o', c='C0', ecolor='k', alpha=0.2, capsize=1, elinewidth=1)
#     plt.title("DNUESKEW " + str(round(DNUESKEW, 2)) + " TAUESKEW " + str(round(TAUESKEW, 2)) +
#               " DNUEFAC " + str(round(DNUEFAC, 2)) + " TAUEFAC " + str(round(TAUEFAC, 2)))
#     plt.ylabel(r"Scintillation Velocity ($km\,s^{-1}$)")
#     plt.xlabel("Orbital Phase (degrees)")
#     plt.show()
#     plt.close()
#     #
#     # plt.errorbar(freq[freq_sort], dnu[freq_sort], yerr=new_dnuerr[freq_sort],
#     #              fmt='o', c='C0', ecolor='k', alpha=0.2, capsize=1, elinewidth=1)
#     # plt.title("DNUESKEW " + str(round(DNUESKEW, 2)) + " TAUESKEW " + str(round(TAUESKEW, 2)) +
#     #           " DNUEFAC " + str(round(DNUEFAC, 2)) + " TAUEFAC " + str(round(TAUEFAC, 2)))
#     # plt.ylabel("Scintillation bandwdith (MHz)")
#     # plt.xlabel("Frequency (MHz)")
#     # plt.show()
#     # plt.close()
#     #
#     # plt.errorbar(freq[freq_sort], tau[freq_sort], yerr=new_tauerr[freq_sort],
#     #              fmt='o', c='C0', ecolor='k', alpha=0.2, capsize=1, elinewidth=1)
#     # plt.title("DNUESKEW " + str(round(DNUESKEW, 2)) + " TAUESKEW " + str(round(TAUESKEW, 2)) +
#     #           " DNUEFAC " + str(round(DNUEFAC, 2)) + " TAUEFAC " + str(round(TAUEFAC, 2)))
#     # plt.ylabel("Scintillation bandwdith (MHz)")
#     # plt.xlabel("Frequency (MHz)")
#     # plt.show()
#     # plt.close()
