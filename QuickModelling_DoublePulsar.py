#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 14:23:33 2023

@author: jacobaskew
"""

###############################################################################
# Importing neccessary things
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
import shutil
from scipy.optimize import curve_fit
from scintools.scint_sim import Simulation
from astropy.table import Table, vstack
import bilby
import pickle
from random import randrange
from astropy.time import Time
from VissGaussianLikelihood import VissGaussianLikelihood
###############################################################################


# def scint_velocity_alternate(params, dnu, tau, freq, dnuerr, tauerr):
#     """
#     Calculate scintillation velocity from ACF frequency and time scales
#     """

#     freq = freq / 1e3  # convert to GHz
#     s = params['s']
#     d = params['d']
#     derr = params['Derr']
#     d_normal = np.random.normal(loc=float(d), scale=float(derr), size=1000)
#     viss = []
#     viss_err = []
#     for i in range(0, len(dnu)):
#         dnu_normal = np.random.normal(loc=dnu[i], scale=dnuerr[i], size=1000)
#         tau_normal = np.random.normal(loc=tau[i], scale=tauerr[i], size=1000)
#         coeff = 2.78e4 * np.sqrt((2*(1-s))/(s))
#         viss_normal = \
#             coeff * (np.sqrt(d_normal*dnu_normal))/(freq[i]*tau_normal)
#         viss.append(np.median(viss_normal))
#         viss_err.append(np.std(viss_normal))
#     return viss, viss_err


def scint_velocity_alternate(params, dnu, tau, freq, dnuerr, tauerr):
    """
    Calculate scintillation velocity from ACF frequency and time scales
    """
    d = params['d']
    s = params['s']
    d_err = params['derr']

    freq = freq / 1e3  # convert to GHz
    coeff = 2.78e4 * np.sqrt((2*(1-s))/(s))
    viss = coeff * (np.sqrt(d*dnu))/(freq*tau)
    viss_err = viss * np.sqrt((d_err/(2*d))**2+(dnuerr/(2*dnu))**2 +
                              (-tauerr/tau)**2)
    return viss, viss_err


###############################################################################


# def effective_velocity_annual_bilby(xdata, d, s, k, vism_ra, vism_dec, KIN,
#                                     KOM, TAUEFAC, DNUEFAC, TAUESKEW, DNUESKEW,
#                                     alpha, **kwargs):
#     """
#     Effective velocity with annual and pulsar terms
#         Note: Does NOT include IISM velocity, but returns veff in IISM frame
#     """
#     # Define the initial parameters
#     params_ = dict(params)
#     params_['d'] = d
#     params_['s'] = s
#     params_['kappa'] = k
#     params_['KOM'] = KOM
#     params_['KIN'] = KIN
#     params_['vism_ra'] = vism_ra
#     params_['vism_dec'] = vism_dec
#     mjd = xdata
#     mjd_sort = np.argsort(mjd)
#     true_anomaly = U[mjd_sort]
#     vearth_ra = ve_ra[mjd_sort]
#     vearth_dec = ve_dec[mjd_sort]
#     #
#     # true_anomaly = U
#     # vearth_ra, vearth_dec = ve_ra, ve_dec
#     #
#     # true_anomaly = get_true_anomaly(mjd, pars)
#     # vearth_ra, vearth_dec = get_earth_velocity(mjd, pars['RAJ'],
#     #                                             pars['DECJ'])
#     # Define some constants
#     v_c = 299792.458  # km/s
#     kmpkpc = 3.085677581e16
#     secperyr = 86400*365.2425
#     masrad = np.pi/(3600*180*1000)

#     # tempo2 parameters from par file in capitals
#     if 'PB' in params_.keys():
#         A1 = params_['A1']  # projected semi-major axis in lt-s
#         PB = params_['PB']  # orbital period in days
#         ECC = params_['ECC']  # orbital eccentricity
#         OM = params_['OM'] * np.pi/180  # longitude of periastron rad
#         if 'OMDOT' in params_.keys():
#             if mjd is None:
#                 print('Warning, OMDOT present but no mjd for calculation')
#                 omega = OM
#             else:
#                 omega = OM + \
#                     params_['OMDOT']*np.pi/180*(mjd-params_['T0'])/365.2425
#         else:
#             omega = OM
#         # Note: fifth Keplerian param T0 used in true anomaly calculation
#         if 'KIN' in params_.keys():
#             INC = params_['KIN']*np.pi/180  # inclination
#         elif 'COSI' in params_.keys():
#             INC = np.arccos(params_['COSI'])
#         elif 'SINI' in params_.keys():
#             INC = np.arcsin(params_['SINI'])
#         else:
#             print('Warning: inclination parameter (KIN, COSI, or SINI) ' +
#                   'not found')

#         if 'sense' in params_.keys():
#             sense = params_['sense']
#             if sense < 0.5:  # KIN < 90
#                 if INC > np.pi/2:
#                     INC = np.pi - INC
#             if sense >= 0.5:  # KIN > 90
#                 if INC < np.pi/2:
#                     INC = np.pi - INC

#         KOM = params_['KOM']*np.pi/180  # longitude ascending node

#     if 'PMRA' in params_.keys():
#         PMRA = params_['PMRA']  # proper motion in RA
#         PMDEC = params_['PMDEC']  # proper motion in DEC
#     else:
#         PMRA = 0
#         PMDEC = 0

#     # other parameters in lower-case
#     s = params_['s']  # fractional screen distance
#     d = params_['d']  # pulsar distance in kpc
#     kappa = params_['kappa']
#     d_kmpkpc = d * kmpkpc  # distance in km

#     pmra_v = PMRA * masrad * d_kmpkpc / secperyr
#     pmdec_v = PMDEC * masrad * d_kmpkpc / secperyr

#     # Calculate pulsar velocity aligned with the line of nodes (Vx) and
#     #   perpendicular in the plane (Vy)
#     vp_0 = (2 * np.pi * A1 * v_c) / (np.sin(INC) * PB * 86400 *
#                                      np.sqrt(1 - ECC**2))
#     vp_x = -vp_0 * (ECC * np.sin(omega) + np.sin(true_anomaly + omega))
#     vp_y = vp_0 * np.cos(INC) * (ECC * np.cos(omega) + np.cos(true_anomaly
#                                                               + omega))

#     # Rotate pulsar velocity into RA/DEC
#     vp_ra = np.sin(KOM) * vp_x + np.cos(KOM) * vp_y
#     vp_dec = np.cos(KOM) * vp_x - np.sin(KOM) * vp_y

#     # find total effective velocity in RA and DEC
#     veff_ra = s * vearth_ra + (1 - s) * (vp_ra + pmra_v) - vism_ra
#     veff_dec = s * vearth_dec + (1 - s) * (vp_dec + pmdec_v) - vism_dec

#     # coefficient to match model with data
#     coeff = 1 / np.sqrt((2 * (1 - s)) / s)

#     veff = kappa * (np.sqrt(veff_dec**2 + veff_ra**2))
#     model = coeff * veff / s
#     model = np.float64(model)

#     return model


def effective_velocity_annual_bilby(
        xdata, d, s, k, vism_ra, vism_dec, KIN, KOM, TAUEFAC,
        DNUEFAC, TAUESKEW, DNUESKEW, alpha, psi, R, **kwargs):
    """
    Effective velocity thin screen model.
    Uses Eq. 4 from Rickett et al. (2014) for anisotropy coefficients.

        ydata: arc curvature
    """
    # Define the initial parameters
    params_ = params
    params_['d'] = d
    params_['s'] = s
    params_['kappa'] = k
    params_['vism_ra'] = vism_ra
    params_['vism_dec'] = vism_dec
    if Anisotropy_Option:
        params_['psi'] = psi
        params_['R'] = R
    params_['KOM'] = KOM
    params_['KIN'] = KIN

    mjd = xdata
    true_anomaly = U
    vearth_ra = ve_ra
    vearth_dec = ve_dec
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
    if Anisotropy_Option:
        r = params_['R']  # axial ratio parameter, see Rickett Cordes 1998
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

    if Anisotropy_Option:
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

    # coefficient to match model with data
    coeff = 1 / np.sqrt((2 * (1 - s)) / s)

    veff = kappa * (np.sqrt(a*veff_dec**2 + b*veff_ra**2 +
                            c*veff_ra*veff_dec))
    model = coeff * veff / s
    model = np.float64(model)

    return model


###############################################################################


# def effective_velocity_annual_anisotropy_bilby(xdata, d, s, k, vism_ra,
#                                                vism_dec, psi, R, KIN, KOM,
#                                                TAUEFAC, DNUEFAC, TAUESKEW,
#                                                DNUESKEW, alpha, **kwargs):
#     """
#     Effective velocity thin screen model.
#     Uses Eq. 4 from Rickett et al. (2014) for anisotropy coefficients.

#         ydata: arc curvature
#     """

#     # Define the initial parameters
#     params_ = dict(params)
#     params_['d'] = d
#     params_['s'] = s
#     params_['kappa'] = k
#     params_['vism_ra'] = vism_ra
#     params_['vism_dec'] = vism_dec
#     params_['psi'] = psi
#     params_['R'] = R
#     params_['KOM'] = KOM
#     params_['KIN'] = KIN

#     mjd = xdata
#     mjd_sort = np.argsort(mjd)
#     mjd = mjd[mjd_sort]
#     true_anomaly = U[mjd_sort]
#     vearth_ra = ve_ra[mjd_sort]
#     vearth_dec = ve_dec[mjd_sort]

#     # Define some constants
#     v_c = 299792.458  # km/s
#     kmpkpc = 3.085677581e16
#     secperyr = 86400*365.2425
#     masrad = np.pi/(3600*180*1000)

#     A1 = params_['A1']  # projected semi-major axis in lt-s
#     PB = params_['PB']  # orbital period in days
#     ECC = params_['ECC']  # orbital eccentricity
#     OM = params_['OM'] * np.pi/180  # longitude of periastron rad
#     INC = params_['KIN'] * np.pi/180  # inclination
#     KOM = params_['KOM'] * np.pi/180  # longitude ascending node

#     omega = OM + params_['OMDOT']*np.pi/180*(mjd-params_['T0'])/365.2425

#     PMRA = params_['PMRA']  # proper motion in RA
#     PMDEC = params_['PMDEC']  # proper motion in DEC

#     # other parameters in lower-case
#     s = params_['s']  # fractional screen distance
#     d = params_['d']  # pulsar distance in kpc
#     d_kmpkpc = d * kmpkpc  # distance in km

#     r = params_['R']  # axial ratio parameter, some complicated relationship
#     psi = params_['psi'] * np.pi / 180  # anisotropy angle
#     kappa = params_['kappa']

#     # Calculate pulsar velocity aligned with the line of nodes (Vx) and
#     #   perpendicular in the plane (Vy)
#     vp_0 = (2 * np.pi * A1 * v_c) / (np.sin(INC) * PB * 86400 *
#                                      np.sqrt(1 - ECC**2))
#     vp_x = -vp_0 * (ECC * np.sin(omega) + np.sin(true_anomaly + omega))
#     vp_y = vp_0 * np.cos(INC) * (ECC * np.cos(omega) + np.cos(true_anomaly
#                                                               + omega))

#     pmra_v = PMRA * masrad * d_kmpkpc / secperyr
#     pmdec_v = PMDEC * masrad * d_kmpkpc / secperyr

#     # Rotate pulsar velocity into RA/DEC
#     vp_ra = np.sin(KOM) * vp_x + np.cos(KOM) * vp_y
#     vp_dec = np.cos(KOM) * vp_x - np.sin(KOM) * vp_y

#     # find total effective velocity in RA and DEC
#     veff_ra = s * vearth_ra + (1 - s) * (vp_ra + pmra_v) - vism_ra
#     veff_dec = s * vearth_dec + (1 - s) * (vp_dec + pmdec_v) - vism_dec

#     cosa = np.cos(2 * psi)
#     sina = np.sin(2 * psi)

#     # quadratic coefficients
#     a = (1 - r * cosa) / np.sqrt(1 - r**2)
#     b = (1 + r * cosa) / np.sqrt(1 - r**2)
#     c = -2 * r * sina / np.sqrt(1 - r**2)

#     # coefficient to match model with data
#     coeff = 1 / np.sqrt((2 * (1 - s)) / s)

#     veff = kappa * (np.sqrt(a*veff_dec**2 + b*veff_ra**2 +
#                             c*veff_ra*veff_dec))
#     model = coeff * veff / s
#     model = np.float64(model)

#     return model


###############################################################################
options = {
    "weights_2dacf": False,
    "compare": False,
    "Modelling": True,
    "distance": False,
    "FullData": True,
    "Anisotropy_Option": True,
    "GetPhaseNviss": False,
    "resume": False,
    "sense": False,
}

weights_2dacf = options['weights_2dacf']
distance = options['distance']
compare = options['compare']
Modelling = options['Modelling']
FullData = options['FullData']
Anisotropy_Option = options['Anisotropy_Option']
if Anisotropy_Option:
    Isotropic = False
else:
    Isotropic = True
GetPhaseNviss = options['GetPhaseNviss']
resume = options['resume']
sense = options['sense']  # True = Flipped, False = not-flipped
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

# weights_2dacf = None
# compare = False
# Modelling = True
# FullData = True
# Anisotropy_Option = False
# GetPhaseNviss = False
# resume = False
# sense = False  # True = Flipped, False = not-flipped
nlive = 200
input_KIN = 89.35  # Kramer et al. 2021
input_KIN_err = 0.05
# Other values ... earlier timing 88.69 + 0.5 - 0.76 ... eclipse 89.3 +- 0.1
#  earlier scintillation 88.1 +- 0.5
if sense:
    input_KIN = 180 - input_KIN
if compare:
    Method_X = 'dnuscale_dynamic'
    Method_Y = '1/sqrt(N)+Cropping+curated'
    compare_file = ''

Size = 80*np.pi  # Determines the size of the datapoints used
font = {'size': 28}
matplotlib.rc('font', **font)

desktopdir = '/Users/jacobaskew/Desktop/'
datadir = desktopdir + 'DoublePulsar_Project/datasets/FullDataStorage/'
par_dir = '/Users/jacobaskew/Desktop/Swinburne_Daniel/ParFiles/'
psrname = 'J0737-3039A'
pars = read_par(str(par_dir) + str(psrname) + '.par')
datafile = '/Users/jacobaskew/Desktop/1:NWeights_NoCropping_NoPhasewrapper/Datafiles/J0737-3039A_freq30_time10_ScintillationResults_UHF_Total_zapped.txt'
outdir = '/Users/jacobaskew/Desktop/NoWeights_NoCropping_NoPhasewrapper/Data/'
params = read_results(datafile)
# Read in arrays
mjd = float_array_from_dict(params, 'mjd')
df = float_array_from_dict(params, 'df')  # channel bandwidth
dnu = float_array_from_dict(params, 'dnu')  # scint bandwidth
# dnu_est = float_array_from_dict(params, 'dnu_est')
dnuerr = float_array_from_dict(params, 'dnuerr')
tau = float_array_from_dict(params, 'tau')
tauerr = float_array_from_dict(params, 'tauerr')
fse_tau = float_array_from_dict(params, 'fse_tau')
fse_dnu = float_array_from_dict(params, 'fse_dnu')
freq = float_array_from_dict(params, 'freq')
bw = float_array_from_dict(params, 'bw')
name = np.asarray(params['name'])
# scintle_num = float_array_from_dict(params, 'scintle_num')
tobs = float_array_from_dict(params, 'tobs')  # tobs in second
# rcvrs = np.array([rcvr[0] for rcvr in params['name']])
scint_param_method = np.asarray(params['scint_param_method'])
phasegrad = float_array_from_dict(params, 'phasegrad')
phasegraderr = float_array_from_dict(params, 'phasegraderr')
if weights_2dacf:
    acf_redchisqr = np.asarray(params['acf_redchisqr'])
    acf_redchisqr = acf_redchisqr.astype(np.float)

# Sort by MJD
sort_ind = np.argsort(mjd)

df = np.array(df[sort_ind]).squeeze()
dnu = np.array(dnu[sort_ind]).squeeze()
# dnu_est = np.array(dnu_est[sort_ind]).squeeze()
dnuerr = np.array(dnuerr[sort_ind]).squeeze()
tau = np.array(tau[sort_ind]).squeeze()
tauerr = np.array(tauerr[sort_ind]).squeeze()
fse_tau = np.array(fse_tau[sort_ind]).squeeze()
fse_dnu = np.array(fse_dnu[sort_ind]).squeeze()
mjd = np.array(mjd[sort_ind]).squeeze()
# rcvrs = np.array(rcvrs[sort_ind]).squeeze()
freq = np.array(freq[sort_ind]).squeeze()
tobs = np.array(tobs[sort_ind]).squeeze()
name = np.array(name[sort_ind]).squeeze()
phasegrad = np.array(phasegrad[sort_ind]).squeeze()
phasegraderr = np.array(phasegraderr[sort_ind]).squeeze()
# scintle_num = np.array(scintle_num[sort_ind]).squeeze()
bw = np.array(bw[sort_ind]).squeeze()
scint_param_method = np.array(scint_param_method[sort_ind]).squeeze()
if weights_2dacf:
    acf_redchisqr = np.array(acf_redchisqr[sort_ind]).squeeze()
# indicies = np.ones(100)
# for i in range(0, 100):
#     indicies[i] = [randrange(0, len(mjd)-1)]
# Used to filter the data
if weights_2dacf:
    indicies = np.argwhere((tauerr < 10*tau) *
                           (dnuerr < 10*dnu) *
                           (scint_param_method == "acf2d_approx") *
                           (acf_redchisqr < 1250))
else:
    indicies = np.argwhere((tauerr < 10*tau) *
                           (dnuerr < 10*dnu) *
                           (scint_param_method == "acf2d_approx"))  # *
    #                         (tauerr == fse_tau) *
    #                         (dnuerr == fse_dnu))
    # indicies = np.argwhere((tauerr == fse_tau) *
    #                         (dnuerr == fse_dnu))

df = df[indicies].squeeze()
dnu = dnu[indicies].squeeze()
# dnu_est = dnu_est[indicies].squeeze()
dnuerr = dnuerr[indicies].squeeze()
tau = tau[indicies].squeeze()
tauerr = tauerr[indicies].squeeze()
fse_tau = fse_tau[indicies].squeeze()
fse_dnu = fse_dnu[indicies].squeeze()
mjd = mjd[indicies].squeeze()
# rcvrs = rcvrs[indicies].squeeze()
freq = freq[indicies].squeeze()
tobs = tobs[indicies].squeeze()
name = name[indicies].squeeze()
# scintle_num = scintle_num[indicies].squeeze()
bw = bw[indicies].squeeze()
phasegrad = phasegrad[indicies].squeeze()
phasegraderr = phasegraderr[indicies].squeeze()
scint_param_method = scint_param_method[indicies].squeeze()
if weights_2dacf:
    acf_redchisqr = acf_redchisqr[indicies].squeeze()


# indicies = []
# for i in range(0, 100):
#     indicies.append(randrange(0, len(mjd)-1))
# indicies = np.asarray(indicies)
if not FullData:
    dnu_median = []
    tau_median = []
    freq_median = []
    dnuerr_std = []
    tauerr_std = []
    for i in range(0, len(np.unique(mjd))):
        dnu_median.append(np.median(dnu[np.argwhere(mjd == np.unique(mjd)[i])]))
        tau_median.append(np.median(tau[np.argwhere(mjd == np.unique(mjd)[i])]))
        freq_median.append(np.median(freq[np.argwhere(mjd == np.unique(mjd)[i])]))
        dnuerr_std.append(np.std(dnu[np.argwhere(mjd == np.unique(mjd)[i])]))
        tauerr_std.append(np.std(tau[np.argwhere(mjd == np.unique(mjd)[i])]))
    dnu = np.asarray(dnu_median)
    tau = np.asarray(tau_median)
    freq = np.asarray(freq_median)
    dnuerr = np.asarray(dnuerr_std)
    tauerr = np.asarray(tauerr_std)
    mjd = np.unique(mjd)
    pars = read_par(str(par_dir) + str(psrname) + '.par')
    ssb_delays = get_ssb_delay(mjd, pars['RAJ'], pars['DECJ'])
    mjd += np.divide(ssb_delays, 86400)  # add ssb delay
    """
    Model Viss
    """
    print('Getting Earth velocity')
    vearth_ra, vearth_dec = get_earth_velocity(mjd, pars['RAJ'],
                                               pars['DECJ'])
    print('Getting true anomaly')
    U = get_true_anomaly(mjd, pars)

    vearth_ra = vearth_ra.squeeze()
    vearth_dec = vearth_dec.squeeze()

    om = pars['OM'] + pars['OMDOT']*(mjd - pars['T0'])/365.2425
    # compute orbital phase
    phase = U*180/np.pi + om
    phase = phase % 360

# df = df[indicies]
# dnu = dnu[indicies]
# # dnu_est = dnu_est[indicies]
# dnuerr = dnuerr[indicies]
# tau = tau[indicies]
# tauerr = tauerr[indicies]
# fse_tau = fse_tau[indicies]
# fse_dnu = fse_dnu[indicies]
# mjd = mjd[indicies]
# # rcvrs = rcvrs[indicies]
# freq = freq[indicies]
# tobs = tobs[indicies]
# name = name[indicies]
# # scintle_num = scintle_num[indicies]
# bw = bw[indicies]
# phasegrad = phasegrad[indicies]
# phasegraderr = phasegraderr[indicies]
# scint_param_method = scint_param_method[indicies]
# if weights_2dacf:
#     acf_redchisqr = acf_redchisqr[indicies]


if compare:
    par_dir = '/Users/jacobaskew/Desktop/Swinburne_Daniel/ParFiles/'
    psrname = 'J0737-3039A'
    outdir = '/Users/jacobaskew/Desktop/NoWeights_NoCropping_NoPhasewrapper/Data/'
    params = read_results(compare_file)
    pars = read_par(str(par_dir) + str(psrname) + '.par')

    # Read in arrays
    diff_mjd = float_array_from_dict(params, 'mjd')
    diff_df = float_array_from_dict(params, 'df')  # channel bandwidth
    diff_dnu = float_array_from_dict(params, 'dnu')  # scint bandwidth
    # dnu_est = float_array_from_dict(params, 'dnu_est')
    diff_dnuerr = float_array_from_dict(params, 'dnuerr')
    diff_tau = float_array_from_dict(params, 'tau')
    diff_tauerr = float_array_from_dict(params, 'tauerr')
    diff_fse_tau = float_array_from_dict(params, 'fse_tau')
    diff_fse_dnu = float_array_from_dict(params, 'fse_dnu')
    diff_freq = float_array_from_dict(params, 'freq')
    diff_bw = float_array_from_dict(params, 'bw')
    diff_name = np.asarray(params['name'])
    # scintle_num = float_array_from_dict(params, 'scintle_num')
    diff_tobs = float_array_from_dict(params, 'tobs')  # tobs in second
    # rcvrs = np.array([rcvr[0] for rcvr in params['name']])
    diff_scint_param_method = np.asarray(params['scint_param_method'])
    diff_phasegrad = float_array_from_dict(params, 'phasegrad')
    diff_phasegraderr = float_array_from_dict(params, 'phasegraderr')
    if weights_2dacf:
        diff_acf_redchisqr = np.asarray(params['acf_redchisqr'])
        diff_acf_redchisqr = diff_acf_redchisqr.astype(np.float)

    # Sort by MJD
    sort_ind = np.argsort(diff_mjd)

    diff_df = np.array(diff_df[sort_ind]).squeeze()
    diff_dnu = np.array(diff_dnu[sort_ind]).squeeze()
    # dnu_est = np.array(dnu_est[sort_ind]).squeeze()
    diff_dnuerr = np.array(diff_dnuerr[sort_ind]).squeeze()
    diff_tau = np.array(diff_tau[sort_ind]).squeeze()
    diff_tauerr = np.array(diff_tauerr[sort_ind]).squeeze()
    diff_fse_tau = np.array(diff_fse_tau[sort_ind]).squeeze()
    diff_fse_dnu = np.array(diff_fse_dnu[sort_ind]).squeeze()
    diff_mjd = np.array(diff_mjd[sort_ind]).squeeze()
    # rcvrs = np.array(rcvrs[sort_ind]).squeeze()
    diff_freq = np.array(diff_freq[sort_ind]).squeeze()
    diff_tobs = np.array(diff_tobs[sort_ind]).squeeze()
    diff_name = np.array(diff_name[sort_ind]).squeeze()
    diff_phasegrad = np.array(diff_phasegrad[sort_ind]).squeeze()
    diff_phasegraderr = np.array(diff_phasegraderr[sort_ind]).squeeze()
    # scintle_num = np.array(scintle_num[sort_ind]).squeeze()
    diff_bw = np.array(diff_bw[sort_ind]).squeeze()
    diff_scint_param_method = \
        np.array(diff_scint_param_method[sort_ind]).squeeze()
    if weights_2dacf:
        diff_acf_redchisqr = np.array(diff_acf_redchisqr[sort_ind]).squeeze()

    # Used to filter the data
    if weights_2dacf:
        indicies = np.argwhere((diff_tauerr < 10*diff_tau) *
                               (diff_dnuerr < 10*diff_dnu) *
                               (diff_scint_param_method == "acf2d_approx") *
                               (diff_acf_redchisqr < 1250))
    else:
        indicies = np.argwhere((diff_tauerr < 10*diff_tau) *
                                (diff_dnuerr < 10*diff_dnu) *
                                (diff_scint_param_method == "acf2d_approx"))  # *
                                # (tauerr == fse_tau) *
                                # (dnuerr == fse_dnu))
        # indicies = np.argwhere((tauerr == fse_tau) *
        #                        (dnuerr == fse_dnu))

    diff_df = diff_df[indicies].squeeze()
    diff_dnu = diff_dnu[indicies].squeeze()
    # dnu_est = dnu_est[indicies].squeeze()
    diff_dnuerr = diff_dnuerr[indicies].squeeze()
    diff_tau = diff_tau[indicies].squeeze()
    diff_tauerr = diff_tauerr[indicies].squeeze()
    diff_fse_tau = diff_fse_tau[indicies].squeeze()
    diff_fse_dnu = diff_fse_dnu[indicies].squeeze()
    diff_mjd = diff_mjd[indicies].squeeze()
    # rcvrs = rcvrs[indicies].squeeze()
    diff_freq = diff_freq[indicies].squeeze()
    diff_tobs = diff_tobs[indicies].squeeze()
    diff_name = diff_name[indicies].squeeze()
    # scintle_num = scintle_num[indicies].squeeze()
    diff_bw = diff_bw[indicies].squeeze()
    diff_phasegrad = diff_phasegrad[indicies].squeeze()
    diff_phasegraderr = diff_phasegraderr[indicies].squeeze()
    diff_scint_param_method = diff_scint_param_method[indicies].squeeze()
    if weights_2dacf:
        diff_acf_redchisqr = diff_acf_redchisqr[indicies].squeeze()

    matching_indices = []
    for i in range(0, len(mjd)):
        for ii in range(0, len(diff_mjd)):
            if mjd[i] == diff_mjd[ii] and freq[i] == diff_freq[ii]:
                matching_indices.append((i, ii))
    matching_indices = np.asarray(matching_indices)
    matching_indices_array1 = matching_indices[:, 0]
    matching_indices_array2 = matching_indices[:, 1]

    compare_name = name[matching_indices_array1]
    compare_old_name = diff_name[matching_indices_array2]
    compare_dnu = dnu[matching_indices_array1]
    compare_dnuerr = dnuerr[matching_indices_array1]
    compare_old_dnu = diff_dnu[matching_indices_array2]
    compare_old_dnuerr = diff_dnuerr[matching_indices_array2]
    compare_tau = tau[matching_indices_array1]
    compare_tauerr = tauerr[matching_indices_array1]
    compare_old_tau = diff_tau[matching_indices_array2]
    compare_old_tauerr = diff_tauerr[matching_indices_array2]
    compare_phasegrad = phasegrad[matching_indices_array1]
    compare_phasegraderr = phasegraderr[matching_indices_array1]
    compare_old_phasegrad = diff_phasegrad[matching_indices_array2]
    compare_old_phasegraderr = diff_phasegraderr[matching_indices_array2]

    # comparing dnu
    fig = plt.figure(figsize=(20, 10))
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    sc = plt.scatter(compare_dnu, compare_old_dnu, c='C0', s=Size,
                     alpha=0.4)
    plt.errorbar(compare_dnu, compare_old_dnu, xerr=compare_dnuerr,
                 yerr=compare_old_dnuerr, fmt=' ',
                 ecolor='k', elinewidth=2, capsize=3, alpha=0.3)
    yl = plt.ylim()
    xl = plt.xlim()
    plt.plot([xl[0], yl[1]], [xl[0], yl[1]], 'k')
    plt.xlabel(str(Method_X)+r': $\Delta\nu_d$ (MHz)')
    plt.ylabel(str(Method_Y)+r': $\Delta\nu_d$ (MHz)')
    plt.xlim(xl)
    plt.ylim(yl)
    plt.savefig("/Users/jacobaskew/Desktop/Dnu_compare.png")
    # plt.savefig(plotdir+"Dnu_compare.pdf", dpi=400)
    plt.show()
    plt.close()
    # comparing tau
    fig = plt.figure(figsize=(20, 10))
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    sc = plt.scatter(compare_tau, compare_old_tau, c='C1', s=Size,
                     alpha=0.4)
    plt.errorbar(compare_tau, compare_old_tau, xerr=compare_tauerr,
                 yerr=compare_old_tauerr, fmt=' ',
                 ecolor='k', elinewidth=2, capsize=3, alpha=0.3)
    yl = plt.ylim()
    xl = plt.xlim()
    plt.plot([xl[0], yl[1]], [xl[0], yl[1]], 'k')
    plt.xlabel(str(Method_X)+r': $\tau_d$ (s)')
    plt.ylabel(str(Method_Y)+r': $\tau_d$ (s)')
    plt.xlim(xl)
    plt.ylim(yl)
    plt.savefig("/Users/jacobaskew/Desktop/Tau_compare.png")
    # plt.savefig(plotdir+"Tau_compare.pdf", dpi=400)
    plt.show()
    plt.close()
    # comparing phasegrad
    fig = plt.figure(figsize=(20, 10))
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    sc = plt.scatter(compare_phasegrad, compare_old_phasegrad, c='C2',
                     s=Size, alpha=0.4)
    plt.errorbar(compare_phasegrad, compare_old_phasegrad,
                 xerr=compare_phasegraderr,
                 yerr=compare_old_phasegraderr, fmt=' ',
                 ecolor='k', elinewidth=2, capsize=3, alpha=0.3)
    yl = plt.ylim()
    xl = plt.xlim()
    plt.plot([xl[0], yl[1]], [xl[0], yl[1]], 'k')
    plt.xlabel(str(Method_X)+r': $\phi$ (mins/MHz)')
    plt.ylabel(str(Method_Y)+r': $\phi$ (mins/MHz)')
    plt.xlim(xl)
    plt.ylim(yl)
    plt.savefig("/Users/jacobaskew/Desktop/phasegrad_compare.png")
    # plt.savefig(plotdir+"phasegrad_compare.pdf", dpi=400)
    plt.show()
    plt.close()

if not compare and FullData and GetPhaseNviss:
    phase_n_mjd_file = outdir + 'phase_data.txt'
    mjd_annual = mjd % 365.2425
    try:
        print("Opening the pickle jar ...")
        # Loading the phase array from the previously saved pickle
        pickle_in = open(datafile+"_phase.pickle", "rb")
        phase_array = pickle.load(pickle_in)
        phase = phase_array
        print("Found the orbital phase ...")

    except Exception as e:
        print(e)
        print("Nothing in this jar ...")
        print("Calculating the phase directly ...")
        # I want to make an array that saves the phase and mjd data
        ssb_delays = get_ssb_delay(mjd, pars['RAJ'], pars['DECJ'])
        mjd += np.divide(ssb_delays, 86400)  # add ssb delay
        """
        Model Viss
        """
        print('Getting Earth velocity')
        vearth_ra, vearth_dec = get_earth_velocity(mjd, pars['RAJ'],
                                                   pars['DECJ'])
        print('Getting true anomaly')
        U = get_true_anomaly(mjd, pars)

        vearth_ra = vearth_ra.squeeze()
        vearth_dec = vearth_dec.squeeze()

        om = pars['OM'] + pars['OMDOT']*(mjd - pars['T0'])/365.2425
        # compute orbital phase
        phase = U*180/np.pi + om
        phase = phase % 360

        # Saving the phase array as a pickle if it was already run in this console
        phase_array = phase
        pickle_out = open(datafile+"_phase.pickle", "wb")
        pickle.dump(phase_array, pickle_out)
        pickle_out.close()
    # PHASE and observation day #
    name_num = []
    for i in range(0, len(name)):
        for ii in range(0, len(np.unique(name))):
            if name[i] == np.unique(name)[ii]:
                name_num.append(ii)
    name_num = np.asarray(name_num)
    # Calculate Viss
    try:
        print("Opening the pickle jar ...")
        # Loading the phase array from the previously saved pickle
        pickle_in = open(datafile+"_viss.pickle", "rb")
        viss_array = pickle.load(pickle_in)
        viss, visserr = viss_array
        print("Found the scintillation velocity ...")

    except Exception as e:
        print(e)
        print("Nothing in this jar ...")
        print("Calculating the viss directly ...")
        d = 0.735  # kpc
        pars = read_par(str(par_dir) + str(psrname) + '.par')
        params = pars_to_params(pars)
        params.add('d', value=d, vary=False)
        params.add('derr', value=0.060, vary=False)
        params.add('s', value=0.72, vary=False)
        viss, visserr = scint_velocity_alternate(params, dnu, tau, freq,
                                                 dnuerr, tauerr)

        # Saving the phase array as a pickle if it was already run in this console
        viss_array = viss, visserr
        pickle_out = open(datafile+"_viss.pickle", "wb")
        pickle.dump(viss_array, pickle_out)
        pickle_out.close()

    ###############################################################################
    # Plotting

    # ORBITAL phase against bandwidth for each 'observation run'
    fig = plt.figure(figsize=(20, 10))
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    cm = plt.cm.get_cmap('viridis')
    z = mjd_annual
    sc = plt.scatter(phase, dnu, c=z, cmap=cm, s=Size, alpha=0.7)
    # plt.colorbar(sc)
    plt.errorbar(phase, dnu, yerr=dnuerr, fmt=' ', ecolor='k',
                 elinewidth=2, capsize=3, alpha=0.55)
    xl = plt.xlim()
    plt.plot(xl, (df[0], df[0]), color='C2')
    plt.xlabel('Orbital Phase (degrees)')
    plt.ylabel('Scintillation Bandwidth (MHz)')
    plt.title('Orbital Phase and "Annual Phase"')
    plt.xlim(xl)
    plt.savefig("/Users/jacobaskew/Desktop/Dnu_Orbital_Freq.png")
    plt.show()
    plt.close()

    # A plot showing the annual modulation if any?! ANNUAL #
    fig = plt.figure(figsize=(20, 10))
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    ax = fig.add_subplot(1, 1, 1)
    cm = plt.cm.get_cmap('viridis')
    z = mjd_annual
    sc = plt.scatter(freq, dnu, c=z, cmap=cm, s=Size, alpha=0.3)
    # plt.colorbar(sc)
    plt.errorbar(freq, dnu, yerr=dnuerr, fmt=' ', ecolor='k',
                 elinewidth=2, capsize=3, alpha=0.2)
    xl = plt.xlim()
    plt.plot(xl, (0.0332, 0.0332), color='C2')
    freq_range = np.linspace(xl[0], xl[1], 10000)
    freq_sort = np.argsort(freq_range)
    estimated_si = 0.05*(freq_range/800)**4
    plt.plot(freq_range[freq_sort], estimated_si[freq_sort], color='k',
             alpha=0.7)
    # predicted_si = 0.05*(freq_range/800)**2
    # plt.plot(freq_range[freq_sort], predicted_si[freq_sort], color='C1',
    #          alpha=0.7)
    plt.xlabel('Observation Frequency (MHz)')
    plt.ylabel('Scintillation Bandwidth (MHz)')
    plt.title('Spectral Index and "Annual Phase"')
    plt.xlim(xl)
    plt.savefig("/Users/jacobaskew/Desktop/Dnu_Freq_Observation.png")
    # ax.set_xscale('log')
    # ax.set_yscale('log')
    plt.savefig(
        "/Users/jacobaskew/Desktop/Dnu_Freq_annual_Observation_log.png")
    plt.show()
    plt.close()
    if weights_2dacf:

        # A plot showing the annual modulation if any?! ANNUAL #
        fig = plt.figure(figsize=(20, 10))
        fig.subplots_adjust(hspace=0.5, wspace=0.5)
        ax = fig.add_subplot(1, 1, 1)
        cm = plt.cm.get_cmap('viridis')
        z = acf_redchisqr
        sc = plt.scatter(freq, dnu, c=z, cmap=cm, s=Size, alpha=0.2)
        plt.colorbar(sc)
        plt.errorbar(freq, dnu, yerr=dnuerr, fmt=' ', ecolor='k',
                     elinewidth=2, capsize=3, alpha=0.1)
        xl = plt.xlim()
        plt.plot(xl, (0.0332, 0.0332), color='C2')
        freq_range = np.linspace(xl[0], xl[1], 10000)
        freq_sort = np.argsort(freq_range)
        estimated_si = 0.05*(freq_range/800)**4
        plt.plot(freq_range[freq_sort], estimated_si[freq_sort], color='k',
                 alpha=0.7)
        predicted_si = 0.05*(freq_range/800)**2
        plt.plot(freq_range[freq_sort], predicted_si[freq_sort],
                 color='C1', alpha=0.7)
        plt.xlabel('Observation Frequency (MHz)')
        plt.ylabel('Scintillation Bandwidth (MHz)')
        plt.title('Spectral Index and "red-chisqr"')
        plt.xlim(xl)
        plt.savefig(
            "/Users/jacobaskew/Desktop/Dnu_Freq_chisqr_Observation.png")
        ax.set_xscale('log')
        ax.set_yscale('log')
        plt.savefig(
            "/Users/jacobaskew/Desktop/Dnu_Freq_chisqr_Observation_log.png")
        plt.show()
        plt.close()

        fig, ax = plt.subplots(figsize=(15, 15))
        plt.hist(acf_redchisqr, color='C0')
        plt.xlabel("Reduced chi-sqr")
        plt.ylabel("Frequency")
        plt.show()
        plt.close()

    # ORBITAL phase against timescale
    fig = plt.figure(figsize=(20, 10))
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    cm = plt.cm.get_cmap('viridis')
    z = mjd_annual
    sc = plt.scatter(phase, tau, c=z, cmap=cm, s=Size, alpha=0.7)
    # plt.colorbar(sc)
    plt.errorbar(phase, tau, yerr=tauerr, fmt=' ', ecolor='k',
                 elinewidth=2, capsize=3, alpha=0.55)
    xl = plt.xlim()
    plt.xlabel('Orbital Phase (degrees)')
    plt.ylabel('Scintillation Timescale (s)')
    plt.title('Timescale and "Annual Phase"')
    plt.xlim(xl)
    plt.savefig("/Users/jacobaskew/Desktop/Dnu_Orbital_Freq.png")
    plt.show()
    plt.close()

    # TIMESCALE V DNU
    fig = plt.figure(figsize=(20, 10))
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    cm = plt.cm.get_cmap('viridis')
    z = phase
    sc = plt.scatter(tau, dnu, c=z, cmap=cm, s=Size, alpha=0.4)
    # plt.colorbar(sc)
    plt.errorbar(tau, dnu, xerr=tauerr, yerr=dnuerr, fmt=' ', ecolor='k',
                 elinewidth=2, capsize=3, alpha=0.3)
    xl = plt.xlim()
    plt.ylabel('Scintillation Bandwidth (MHz)')
    plt.xlabel('Scintillation Timescale (s)')
    plt.title('Tau V Dnu and "Orbital Phase"')
    # plt.xlim(30, 100)
    plt.savefig("/Users/jacobaskew/Desktop/Dnu_Orbital_Freq.png")
    plt.show()
    plt.close()

    # Viss against orbital phase with annual phase
    fig = plt.figure(figsize=(20, 10))
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    ax = fig.add_subplot(1, 1, 1)
    cm = plt.cm.get_cmap('viridis')
    z = mjd_annual
    sc = plt.scatter(phase, viss, c=z, cmap=cm, s=Size, alpha=0.7)
    plt.colorbar(sc)
    plt.errorbar(phase, viss, yerr=visserr, fmt=' ', ecolor='k',
                 elinewidth=2, capsize=3, alpha=0.55)
    xl = plt.xlim()
    plt.xlabel('Orbital Phase (degrees)')
    plt.ylabel(r'Scintillation Velocity (km$\,$s$^{-1}$)')
    plt.title('Velocity and Orbital/Annual Phase')
    plt.xlim(xl)
    plt.savefig("/Users/jacobaskew/Desktop/Viss_test.png", dpi=400)
    plt.savefig("/Users/jacobaskew/Desktop/Viss_test.pdf", dpi=400)
    plt.show()
    plt.close()
    np.savetxt(datadir + 'Full_VissData.txt', viss, delimiter=',')
    np.savetxt(datadir + 'Full_VisserrData.txt', visserr,
               delimiter=',')
    np.savetxt(datadir + 'Full_MJDData.txt', mjd, delimiter=',')
    np.savetxt(datadir + 'Full_FreqData.txt', freq, delimiter=',')
    np.savetxt(datadir + 'Full_DnuData.txt', dnu, delimiter=',')
    np.savetxt(datadir + 'Full_DnuerrData.txt', dnuerr, delimiter=',')
    np.savetxt(datadir + 'Full_TauData.txt', tau, delimiter=',')
    np.savetxt(datadir + 'Full_TauerrData.txt', tauerr, delimiter=',')
    np.savetxt(datadir + 'Full_PhaseData.txt', phase, delimiter=',')

###############################################################################

# np.savetxt(datadir + 'Test_VissData.txt', viss, delimiter=',')
# np.savetxt(datadir + 'Test_VisserrData.txt', visserr,
#            delimiter=',')

if FullData:
    viss = np.loadtxt(datadir + 'Full_VissData.txt', dtype='float')
    visserr = np.loadtxt(datadir + 'Full_VisserrData.txt',
                         dtype='float')
    mjd = np.loadtxt(datadir + 'Full_MJDData.txt', dtype='float')
    freq = np.loadtxt(datadir + 'Full_FreqData.txt', dtype='float')
    dnu = np.loadtxt(datadir + 'Full_DnuData.txt', dtype='float')
    dnuerr = np.loadtxt(datadir + 'Full_DnuerrData.txt', dtype='float')
    tau = np.loadtxt(datadir + 'Full_TauData.txt', dtype='float')
    tauerr = np.loadtxt(datadir + 'Full_TauerrData.txt', dtype='float')
    phase = np.loadtxt(datadir + 'Full_PhaseData.txt', dtype='float')
else:
    np.savetxt(datadir + 'Test_MJDData.txt', mjd, delimiter=',')
    np.savetxt(datadir + 'Test_FreqData.txt', freq, delimiter=',')
    np.savetxt(datadir + 'Test_DnuData.txt', dnu, delimiter=',')
    np.savetxt(datadir + 'Test_DnuerrData.txt', dnuerr, delimiter=',')
    np.savetxt(datadir + 'Test_TauData.txt', tau, delimiter=',')
    np.savetxt(datadir + 'Test_TauerrData.txt', tauerr, delimiter=',')
    np.savetxt(datadir + 'Test_PhaseData.txt', phase, delimiter=',')
    viss = np.loadtxt(datadir + 'Test_VissData.txt', dtype='float')
    visserr = np.loadtxt(datadir + 'Test_VisserrData.txt',
                         dtype='float')
    mjd = np.loadtxt(datadir + 'Test_MJDData.txt', dtype='float')
    freq = np.loadtxt(datadir + 'Test_FreqData.txt', dtype='float')
    dnu = np.loadtxt(datadir + 'Test_DnuData.txt', dtype='float')
    dnuerr = np.loadtxt(datadir + 'Test_DnuerrData.txt', dtype='float')
    tau = np.loadtxt(datadir + 'Test_TauData.txt', dtype='float')
    tauerr = np.loadtxt(datadir + 'Test_TauerrData.txt', dtype='float')
    phase = np.loadtxt(datadir + 'Test_PhaseData.txt', dtype='float')
pars = read_par(str(par_dir) + str(psrname) + '.par')
params = pars_to_params(pars)
U = get_true_anomaly(mjd, pars)
ve_ra, ve_dec = get_earth_velocity(mjd, pars['RAJ'], pars['DECJ'])
np.savetxt(datadir + 'Full_UData.txt', U, delimiter=',')
np.savetxt(datadir + 'Full_ve_raData.txt', ve_ra, delimiter=',')
np.savetxt(datadir + 'Full_ve_decData.txt', ve_dec, delimiter=',')


# plt.plot(viss)
# plt.plot(viss_model)
# plt.plot(viss - viss_model)
# plt.title('Outside Bibly')
# plt.show()
# U = dict(enumerate(U.flatten(), 1))
# ve_ra = dict(enumerate(ve_ra.flatten(), 1))
# ve_dec = dict(enumerate(ve_dec.flatten(), 1))
# X = dict(np.ndenumerate(U))
# Y = dict(np.ndenumerate(ve_ra, "ve_ra"))
# Z = dict(np.ndenumerate(ve_dec, "ve_dec"))
# X.update(Y)
# X.update(Z)
# indicies = []
# for i in range(0, 10):
#     indicies.append(randrange(0, len(mjd)-1))
# indicies = np.asarray(indicies)
# viss = viss[indicies]
# visserr = visserr[indicies]
# mjd = mjd[indicies]

# Modelling
if Modelling:
    label = "_".join(filter(None, [key if value else None
                                   for key, value in labellist.items()]))
    # label += '_nlive' + str(options['nlive']) + '_KIN' + str(input_KIN)
    wd0 = "/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/New/"
    outdir = wd0 + "Modelling"
# input_KIN
# xdata, D, s, k, vism_ra, vism_dec, psi, R, KIN, KOM, TAUEFAC, DNUEFAC, TAUESKEW, DNUESKEW, alpha
    if Anisotropy_Option:
        outdir += '/Anisotropic/' + str(label)
        try:
            os.mkdir(outdir)
        except OSError as error:
            print(error)
        priors = \
            dict(
                d=bilby.core.prior.analytical.DeltaFunction(0.735, 'd'),
                  # d=bilby.core.prior.Uniform(0, 10, 'd'),
                 s=bilby.core.prior.Uniform(0.001, 1, 's'),
                    # k=bilby.core.prior.Uniform(0, 10, 'k'),
                    k=bilby.core.prior.analytical.DeltaFunction(1, 'k'),
                 vism_ra=bilby.core.prior.Uniform(-300, 300, 'vism_ra'),
                 vism_dec=bilby.core.prior.Uniform(-300, 300, 'vism_dec'),
                 psi=bilby.core.prior.Uniform(0, 180, 'psi',
                                              boundary='periodic'),
                 R=bilby.core.prior.Uniform(0, 1, 'R'),
                    KIN=bilby.core.prior.analytical.DeltaFunction(input_KIN,
                                                                  'KIN'),
                    # KIN=bilby.core.prior.Uniform(0, 180, 'KIN',
                    #                              boundary='periodic'),
                    # KOM=bilby.core.prior.Uniform(0, 360, 'KOM'),
                    KOM=bilby.core.prior.analytical.DeltaFunction(65,
                                                                  name='KOM'),
                 #                              boundary='periodic'),
                    # TAUEFAC=bilby.core.prior.Uniform(-10, 10, 'TAUEFAC'),
                    TAUEFAC=bilby.core.prior.analytical.DeltaFunction(0,
                                                                      'TAUEFAC'),
                    # DNUEFAC=bilby.core.prior.Uniform(-10, 10, 'DNUEFAC'),
                    DNUEFAC=bilby.core.prior.analytical.DeltaFunction(0,
                                                                        'DNUEFAC'),
                    # TAUESKEW=bilby.core.prior.Uniform(-10, 10, 'TAUESKEW'),
                    TAUESKEW=bilby.core.prior.analytical.DeltaFunction(-100,
                                                                        'TAUESKEW'),
                    # DNUESKEW=bilby.core.prior.Uniform(-10, 10, 'DNUESKEW'),
                    DNUESKEW=bilby.core.prior.analytical.DeltaFunction(-100,
                                                                        'DNUESKEW'),
                  alpha=bilby.core.prior.analytical.DeltaFunction(4, 'alpha'))
                    # alpha=bilby.core.prior.Uniform(0, 5, 'alpha'))

        # index_array = np.linspace(0, len(mjd)-1, len(mjd))
        # index_array = np.array([int(i) for i in index_array])
        # ri = np.unique(np.random.choice(index_array, size=100))

        # likelihood = \
        #     VissGaussianLikelihood(
        #         x=mjd[ri], y=viss[ri],
        #         func=effective_velocity_annual_bilby,
        #         freq=freq[ri], tau=tau[ri], dnu=dnu[ri], tauerr=tauerr[ri],
        #         dnuerr=dnuerr[ri], sigma=visserr[ri],
        #         kwargs=(U[ri], ve_ra[ri], ve_dec[ri], params,
        # Anisotropy_Option))

        likelihood = \
            VissGaussianLikelihood(
                x=mjd, y=viss, func=effective_velocity_annual_bilby,
                freq=freq, tau=tau, dnu=dnu, tauerr=tauerr, dnuerr=dnuerr,
                sigma=visserr, kwargs=(U, ve_ra, ve_dec, params,
                                       Anisotropy_Option))

    else:
        outdir += '/Isotropic/' + str(label)
        try:
            os.mkdir(outdir)
        except OSError as error:
            print(error)
        priors = \
            dict(
                d=bilby.core.prior.analytical.DeltaFunction(0.735, 'd'),
                   # d=bilby.core.prior.Uniform(0, 10, 'd'),
                 s=bilby.core.prior.Uniform(0.001, 1, 's'),
                  # k=bilby.core.prior.Uniform(0, 10, 'k'),
                   k=bilby.core.prior.analytical.DeltaFunction(1, 'k'),
                 vism_ra=bilby.core.prior.Uniform(-300, 300, 'vism_ra'),
                 vism_dec=bilby.core.prior.Uniform(-300, 300, 'vism_dec'),
                    # KIN=bilby.core.prior.Uniform(0, 180, 'KIN',
                    #                              boundary='periodic'),
                    KIN=bilby.core.prior.analytical.DeltaFunction(input_KIN,
                                                                  'KIN'),
                    # KOM=bilby.core.prior.Uniform(0, 360, 'KOM',
                    #                              boundary='periodic'),
                    KOM=bilby.core.prior.analytical.DeltaFunction(65,
                                                                  name='KOM'),
                    # TAUEFAC=bilby.core.prior.Uniform(-10, 10, 'TAUEFAC'),
                    TAUEFAC=bilby.core.prior.analytical.DeltaFunction(0,
                                                                      'TAUEFAC'),
                    # DNUEFAC=bilby.core.prior.Uniform(-10, 10, 'DNUEFAC'),
                    DNUEFAC=bilby.core.prior.analytical.DeltaFunction(0,
                                                                      'DNUEFAC'),
                    # TAUESKEW=bilby.core.prior.Uniform(-10, 10, 'TAUESKEW'),
                    TAUESKEW=bilby.core.prior.analytical.DeltaFunction(-100,
                                                                        'TAUESKEW'),
                    # DNUESKEW=bilby.core.prior.Uniform(-10, 10, 'DNUESKEW'),
                    DNUESKEW=bilby.core.prior.analytical.DeltaFunction(-100,
                                                                        'DNUESKEW'),
                 psi=bilby.core.prior.analytical.DeltaFunction(1, 'psi'),
                 R=bilby.core.prior.analytical.DeltaFunction(0.5, 'R'),
                  alpha=bilby.core.prior.analytical.DeltaFunction(4, 'alpha'))
                    # alpha=bilby.core.prior.Uniform(0, 5, 'alpha'))

        # index_array = np.linspace(0, len(mjd)-1, len(mjd))
        # index_array = np.array([int(i) for i in index_array])
        # ri = np.unique(np.random.choice(index_array, size=100))

        # likelihood = \
        #     VissGaussianLikelihood(
        #         x=mjd[ri], y=viss[ri], func=effective_velocity_annual_bilby,
        #         freq=freq[ri],
        #         tau=tau[ri], dnu=dnu[ri], tauerr=tauerr[ri], dnuerr=dnuerr[ri],
        #         sigma=visserr[ri], kwargs=(U[ri], ve_ra[ri], ve_dec[ri],
        #                                    params, Anisotropy_Option))
        likelihood = \
            VissGaussianLikelihood(
                x=mjd, y=viss, func=effective_velocity_annual_bilby, freq=freq,
                tau=tau, dnu=dnu, tauerr=tauerr, dnuerr=dnuerr,
                sigma=visserr, kwargs=(U, ve_ra, ve_dec, params,
                                       Anisotropy_Option))

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
# Here we are going to make some fun plots
    pars = read_par(str(par_dir) + str(psrname) + '.par')
    params = pars_to_params(pars)
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
        psi_result = 1
        psierr_result = 0
        R_result = 0.5
        Rerr_result = 0
    # Recalculating the error bars
    alpha = 4
    Aiss = 2.78*10**4
    new_freq = freq / 1e3
    New_dnuerr = np.sqrt((dnuerr * 10**DNUEFAC_result)**2 +
                         ((10**(float(DNUESKEW_result)))*(new_freq/1)**alpha)**2)
    New_tauerr = np.sqrt((tauerr * 10**TAUEFAC_result)**2 +
                         ((10**float(TAUESKEW_result))*(new_freq/1)**(alpha/2))**2)
    New_viss, New_visserr = scint_velocity_alternate(params, dnu, tau, freq,
                                                     New_dnuerr, New_tauerr)
    mjd_year = Time(mjd, format='mjd').byear
    mjd_annual = mjd % 365.2425
    if os.path.exists(datadir + 'Model_mjdData.txt'):
        Model_mjd = np.loadtxt(datadir + 'Model_mjdData.txt',
                               dtype='float')
        Model_phase = np.loadtxt(datadir + 'Model_phaseData.txt',
                                 dtype='float')
        Model_U = np.loadtxt(datadir + 'Model_UData.txt',
                             dtype='float')
        Model_vearth_ra = np.loadtxt(
            datadir + 'Model_vearth_raData.txt', dtype='float')
        Model_vearth_dec = np.loadtxt(
            datadir + 'Model_vearth_decData.txt', dtype='float')
    else:
        BIGNUM = int(round((int((np.max(mjd) - np.min(mjd)) * 1440) / 10) * 11, -4))
        Model_mjd = np.linspace(np.min(mjd), np.max(mjd), BIGNUM)
        Model_ssb_delays = get_ssb_delay(Model_mjd, pars['RAJ'], pars['DECJ'])
        Model_mjd += np.divide(Model_ssb_delays, 86400)  # add ssb delay
        """
        Model Viss
        """
        print('Getting Earth velocity')
        Model_vearth_ra, Model_vearth_dec = get_earth_velocity(Model_mjd,
                                                               pars['RAJ'],
                                                               pars['DECJ'])
        print('Getting true anomaly')
        Model_U = get_true_anomaly(Model_mjd, pars)

        Model_vearth_ra = Model_vearth_ra.squeeze()
        Model_vearth_dec = Model_vearth_dec.squeeze()

        Model_om = pars['OM'] + pars['OMDOT']*(Model_mjd - pars['T0'])/365.2425
        # compute orbital phase
        Model_phase = Model_U*180/np.pi + Model_om
        Model_phase = Model_phase % 360
        np.savetxt(datadir + 'Model_mjdData.txt', Model_mjd,
                   delimiter=',')
        np.savetxt(datadir + 'Model_phaseData.txt', Model_phase,
                   delimiter=',')
        np.savetxt(datadir + 'Model_UData.txt', Model_U,
                   delimiter=',')
        np.savetxt(datadir + 'Model_vearth_raData.txt',
                   Model_vearth_ra, delimiter=',')
        np.savetxt(datadir + 'Model_vearth_decData.txt',
                   Model_vearth_dec, delimiter=',')
    #
    Model_mjd_year = Time(Model_mjd, format='mjd').byear
    Model_mjd_annual = Model_mjd % 365.2425

    # if Anisotropy_Option:
    U_tmp = U
    ve_ra_tmp = ve_ra
    ve_dec_tmp = ve_dec
    U = Model_U
    ve_ra = Model_vearth_ra
    ve_dec = Model_vearth_dec
    Model_veff = \
        effective_velocity_annual_bilby(
            Model_mjd, d_result, s_result, k_result, vism_ra_result,
            vism_dec_result, KIN_result, KOM_result,
            TAUEFAC_result, DNUEFAC_result, TAUESKEW_result,
            DNUESKEW_result, alpha_result, psi_result, R_result,
            kwargs=(U, ve_ra, ve_dec, params, Anisotropy_Option))
    ve_ra = np.zeros(np.shape(Model_mjd))
    ve_dec = np.zeros(np.shape(Model_mjd))
    Model_veff_VP = \
        effective_velocity_annual_bilby(
            Model_mjd, d_result, s_result, k_result, 0,
            0, KIN_result, KOM_result,
            TAUEFAC_result, DNUEFAC_result, TAUESKEW_result,
            DNUESKEW_result, alpha_result, psi_result, R_result,
            kwargs=(U, ve_ra, ve_dec, params, Anisotropy_Option))
    ve_ra = Model_vearth_ra
    ve_dec = Model_vearth_dec
    A1_temp = params['A1'].value
    params.add('A1', value=0, vary=False)
    Model_veff_VE = \
        effective_velocity_annual_bilby(
            Model_mjd, d_result, s_result, k_result, 0,
            0, KIN_result, KOM_result,
            TAUEFAC_result, DNUEFAC_result, TAUESKEW_result,
            DNUESKEW_result, alpha_result, psi_result, R_result,
            kwargs=(U, ve_ra, ve_dec, params, Anisotropy_Option))
    ve_ra = np.zeros(np.shape(Model_mjd))
    ve_dec = np.zeros(np.shape(Model_mjd))
    Model_veff_IISM = \
        effective_velocity_annual_bilby(
            Model_mjd, d_result, s_result, k_result, vism_ra_result,
            vism_dec_result, KIN_result, KOM_result,
            TAUEFAC_result, DNUEFAC_result, TAUESKEW_result,
            DNUESKEW_result, alpha_result, psi_result, R_result,
            kwargs=(U, ve_ra, ve_dec, params, Anisotropy_Option))
    params.add('A1', value=A1_temp, vary=False)

    U = U_tmp
    ve_ra = ve_ra_tmp
    ve_dec = ve_dec_tmp

    veff = \
        effective_velocity_annual_bilby(
            mjd, d_result, s_result, k_result, vism_ra_result,
            vism_dec_result, KIN_result, KOM_result,
            TAUEFAC_result, DNUEFAC_result, TAUESKEW_result,
            DNUESKEW_result, alpha_result, psi_result, R_result,
            kwargs=(U, ve_ra, ve_dec, params, Anisotropy_Option))
    ve_ra = np.zeros(np.shape(mjd))
    ve_dec = np.zeros(np.shape(mjd))
    veff_VP = \
        effective_velocity_annual_bilby(
            mjd, d_result, s_result, k_result, 0,
            0, KIN_result, KOM_result,
            TAUEFAC_result, DNUEFAC_result, TAUESKEW_result,
            DNUESKEW_result, alpha_result, psi_result, R_result,
            kwargs=(U, ve_ra, ve_dec, params, Anisotropy_Option))
    ve_ra = ve_ra_tmp
    ve_dec = ve_dec_tmp
    A1_temp = params['A1'].value
    params.add('A1', value=0, vary=False)
    veff_VE = \
        effective_velocity_annual_bilby(
            mjd, d_result, s_result, k_result, 0,
            0, KIN_result, KOM_result,
            TAUEFAC_result, DNUEFAC_result, TAUESKEW_result,
            DNUESKEW_result, alpha_result, psi_result, R_result,
            kwargs=(U, ve_ra, ve_dec, params, Anisotropy_Option))
    ve_ra = np.zeros(np.shape(mjd))
    ve_dec = np.zeros(np.shape(mjd))
    veff_IISM = \
        effective_velocity_annual_bilby(
            mjd, d_result, s_result, k_result, vism_ra_result,
            vism_dec_result, KIN_result, KOM_result,
            TAUEFAC_result, DNUEFAC_result, TAUESKEW_result,
            DNUESKEW_result, alpha_result, psi_result, R_result,
            kwargs=(U, ve_ra, ve_dec, params, Anisotropy_Option))
    params.add('A1', value=A1_temp, vary=False)
    # else:
    #     U_tmp = U
    #     ve_ra_tmp = ve_ra
    #     ve_dec_tmp = ve_dec
    #     U = Model_U
    #     ve_ra = Model_vearth_ra
    #     ve_dec = Model_vearth_dec

    #     Model_veff = \
    #         effective_velocity_annual_bilby(
    #             Model_mjd, d_result, s_result, k_result, vism_ra_result,
    #             vism_dec_result, KIN_result, KOM_result,
    #             TAUEFAC_result, DNUEFAC_result, TAUESKEW_result,
    #             DNUESKEW_result, alpha_result, psi=None, R=None,
    #             kwargs=(U, ve_ra, ve_dec, params, Anisotropy_Option))
    #     ve_ra = np.zeros(np.shape(Model_mjd))
    #     ve_dec = np.zeros(np.shape(Model_mjd))
    #     Model_veff_VP = \
    #         effective_velocity_annual_bilby(
    #             Model_mjd, d_result, s_result, k_result, vism_ra_result,
    #             vism_dec_result, KIN_result, KOM_result,
    #             TAUEFAC_result, DNUEFAC_result, TAUESKEW_result,
    #             DNUESKEW_result, alpha_result, psi=None, R=None,
    #             kwargs=(U, ve_ra, ve_dec, params, Anisotropy_Option))
    #     ve_ra = Model_vearth_ra
    #     ve_dec = Model_vearth_dec
    #     A1_temp = params['A1'].value
    #     params.add('A1', value=0, vary=False)
    #     Model_veff_VE = \
    #         effective_velocity_annual_bilby(
    #             Model_mjd, d_result, s_result, k_result, vism_ra_result,
    #             vism_dec_result, KIN_result, KOM_result,
    #             TAUEFAC_result, DNUEFAC_result, TAUESKEW_result,
    #             DNUESKEW_result, alpha_result, psi=None, R=None,
    #             kwargs=(U, ve_ra, ve_dec, params, Anisotropy_Option))
    #     params.add('A1', value=A1_temp, vary=False)

    #     U = U_tmp
    #     ve_ra = ve_ra_tmp
    #     ve_dec = ve_dec_tmp

    #     veff = \
    #         effective_velocity_annual_bilby(
    #             mjd, d_result, s_result, k_result, vism_ra_result,
    #             vism_dec_result, KIN_result, KOM_result,
    #             TAUEFAC_result, DNUEFAC_result, TAUESKEW_result,
    #             DNUESKEW_result, alpha_result, psi=None, R=None,
    #             kwargs=(U, ve_ra, ve_dec, params, Anisotropy_Option))
    #     ve_ra = np.zeros(np.shape(mjd))
    #     ve_dec = np.zeros(np.shape(mjd))
    #     veff_VP = \
    #         effective_velocity_annual_bilby(
    #             mjd, d_result, s_result, k_result, vism_ra_result,
    #             vism_dec_result, KIN_result, KOM_result,
    #             TAUEFAC_result, DNUEFAC_result, TAUESKEW_result,
    #             DNUESKEW_result, alpha_result, psi=None, R=None,
    #             kwargs=(U, ve_ra, ve_dec, params, Anisotropy_Option))
    #     ve_ra = ve_ra_tmp
    #     ve_dec = ve_dec_tmp
    #     A1_temp = params['A1'].value
    #     params.add('A1', value=0, vary=False)
    #     veff_VE = \
    #         effective_velocity_annual_bilby(
    #             mjd, d_result, s_result, k_result, vism_ra_result,
    #             vism_dec_result, KIN_result, KOM_result,
    #             TAUEFAC_result, DNUEFAC_result, TAUESKEW_result,
    #             DNUESKEW_result, alpha_result, psi=None, R=None,
    #             kwargs=(U, ve_ra, ve_dec, params, Anisotropy_Option))
    #     params.add('A1', value=A1_temp, vary=False)

    # Model_VP_veff_only = Model_veff_VP - Model_veff - Model_veff_VE
    # Model_VE_veff_only = Model_veff - Model_veff_VP

    viss_VP_only = New_viss - veff_VE  # - veff_IISM
    viss_VE_only = New_viss - veff_VP  # - veff_IISM
    #
    Model_mjd_sort = np.argsort(Model_mjd)
    mjd_sort = np.argsort(mjd)
    Model_phase_sort = np.argsort(Model_phase)
    phase_sort = np.argsort(phase)
    Model_mjd_annual_sort = np.argsort(Model_mjd_annual)
    mjd_annual_sort = np.argsort(mjd_annual)

    # Plotting; New_viss, New_visserr
    Size = 80*np.pi  # Determines the size of the datapoints used
    font = {'size': 28}
    matplotlib.rc('font', **font)
    # Year against corrected data with model

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

    # # Residuals for only phase and only annual; New_viss, New_visserr
    # # phase
    # phase_sort = np.argsort(phase)
    # fig = plt.figure(figsize=(20, 10))
    # fig.subplots_adjust(hspace=0.5, wspace=0.5)
    # ax = fig.add_subplot(1, 1, 1)
    # ax.errorbar(phase[phase_sort],
    #             (New_viss[phase_sort] - (veff - VE_veff_only)[phase_sort]),
    #             yerr=New_visserr[phase_sort],
    #             fmt='o', ecolor='k', elinewidth=2, capsize=3, alpha=0.55)
    # ax.plot([np.min(phase), np.max(phase)], [0, 0], c='C3')
    # xl = plt.xlim()
    # ax.set_xlabel('Orbital Phase (degrees)')
    # ax.set_ylabel(r'Scintillation Velocity (km$\,$s$^{-1}$)')
    # ax.set_xlim(xl)
    # plt.tight_layout()
    # plt.savefig(str(outdir) + "/NewViss+Residuals_phase.png", dpi=400)
    # plt.savefig(str(outdir) + "/NewViss+Residuals_phase.pdf", dpi=400)
    # plt.show()
    # plt.close()

    # # # Annual
    # fig = plt.figure(figsize=(20, 10))
    # fig.subplots_adjust(hspace=0.5, wspace=0.5)
    # ax = fig.add_subplot(1, 1, 1)
    # ax.errorbar(mjd_annual[mjd_annual_sort],
    #             New_viss[mjd_annual_sort] - (veff[mjd_annual_sort] -
    #                                          VE_veff_only[mjd_annual_sort]),
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

    # Frequency variations

    freq_sort = np.argsort(freq)
    fig = plt.figure(figsize=(20, 10))
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    ax = fig.add_subplot(1, 1, 1)
    ax.errorbar(freq[freq_sort], dnu[freq_sort],
                yerr=New_dnuerr[freq_sort],
                fmt='o', ecolor='k', elinewidth=2, capsize=3, alpha=0.55)
    # ax.plot([np.min(phase), np.max(phase)], [0, 0], c='C3')
    xl = plt.xlim()
    ax.set_xlabel('Frequency (MHz)')
    ax.set_ylabel(r'Scintillation Bandwidth (MHz)')
    ax.set_xlim(xl)
    plt.tight_layout()
    plt.savefig(str(outdir) + "/NewDnuFreq.png", dpi=400)
    plt.savefig(str(outdir) + "/NewDnuFreq.pdf", dpi=400)
    plt.show()
    plt.close()

    freq_sort = np.argsort(freq)
    fig = plt.figure(figsize=(20, 10))
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    ax = fig.add_subplot(1, 1, 1)
    ax.errorbar(freq[freq_sort], tau[freq_sort],
                yerr=New_tauerr[freq_sort],
                fmt='o', ecolor='k', elinewidth=2, capsize=3, alpha=0.55)
    # ax.plot([np.min(phase), np.max(phase)], [0, 0], c='C3')
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

if distance:

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

# ax = plt.axes(projection='3d')

# # # Data for a three-dimensional line
# # zline = np.linspace(0, 15, 1000)
# # xline = np.sin(zline)
# # yline = np.cos(zline)
# # ax.plot3D(xline, yline, zline, 'gray')

# # Data for three-dimensional scattered points
# zdata = viss
# xdata = mjd_annual
# ydata = phase
# ax.scatter3D(xdata, ydata, zdata, c='C0', alpha=0.5)

###############################################################################
    # # This section runs across all time to plot a model given some variables

    # mjd_full = np.linspace(np.min(mjd), np.max(mjd), 5000)
    # ssb_delays_full = get_ssb_delay(mjd_full, pars['RAJ'], pars['DECJ'])
    # mjd_full += np.divide(ssb_delays_full, 86400)  # add ssb delay
    # vearth_ra_full, vearth_dec_full = get_earth_velocity(mjd_full, pars['RAJ'],
    #                                                       pars['DECJ'])
    # true_anomaly_full = get_true_anomaly(mjd_full, pars)

    # vearth_ra_full = vearth_ra_full.squeeze()
    # vearth_dec_full = vearth_dec_full.squeeze()

    # om_full = pars['OM'] + pars['OMDOT']*(mjd_full - pars['T0'])/365.2425
    # # compute orbital phase
    # phase_full = true_anomaly_full*180/np.pi + om_full
    # phase_full = phase_full % 360
    # for i in range(0, 36):
    #     params.add('KOM', value=i*10, vary=False)
    #     params.add('KIN', value=89.35, vary=False)  # 89.35 or 90.65
    #     # params.add('vism_psi', value=86.47, vary=False)
    #     # params.add('psi', value=86.47, vary=False)
    #     params.add('d', value=0.735, vary=False)
    #     params.add('derr', value=0.060, vary=False)
    #     params.add('s', value=0.7, vary=False)
    #     params.add('serr', value=0.03, vary=False)

    #     veff2 = np.sqrt(veff_ra**2 + veff_dec**2)
    #     phase_full_sort = np.argsort(phase_full)

    #     # Viss against orbital phase with annual phase
    #     fig = plt.figure(figsize=(20, 10))
    #     fig.subplots_adjust(hspace=0.5, wspace=0.5)
    #     ax = fig.add_subplot(1, 1, 1)
    #     cm = plt.cm.get_cmap('viridis')
    #     z = mjd_annual
    #     sc = plt.scatter(phase, viss, c=z, cmap=cm, s=Size, alpha=0.7)
    #     plt.colorbar(sc)
    #     plt.errorbar(phase, viss, yerr=visserr, fmt=' ', ecolor='k',
    #                  elinewidth=2, capsize=3, alpha=0.55)
    #     xl = plt.xlim()
    #     plt.plot(phase_full[phase_full_sort], veff2[phase_full_sort], c='k',
    #              alpha=0.3)
    #     plt.xlabel('Orbital Phase (degrees)')
    #     plt.ylabel(r'Scintillation Velocity (km$\,$s$^{-1}$)')
    #     plt.title('Velocity and Orbital/Annual Phase')
    #     plt.xlim(xl)
    #     plt.savefig("/Users/jacobaskew/Desktop/Viss_test.png", dpi=400)
    #     plt.savefig("/Users/jacobaskew/Desktop/Viss_test.pdf", dpi=400)
    #     plt.show()
    #     plt.close()

# Calculating the uncertainty in the mean orbital velocity uncertainty
# INC = 89.35*np.pi/180
# v_c = 299792.458
# A1err = 1e-6
# INCerr = 0.05*np.pi/180
# PBerr = (5e-11)*86400
# ECCerr = 9e-7
# dAdV = (2 * np.pi * 1 * v_c) / (np.sin(89.35*np.pi/180) * params['PB'] *
#                                 86400 * np.sqrt(1 - params['ECC']**2))
# didV = (-2*(np.sin(INC) * np.sin(INC*2)) /
#         (-1+np.cos(2*INC)**2)) * ((2 * np.pi * params['A1'] * v_c) /
#                                   (params['PB'] *
#                                    86400 * np.sqrt(1 - params['ECC']**2)))
# dPdV = -(2 * np.pi * params['A1'] * v_c) / (np.sin(89.35*np.pi/180) *
#                                             (params['PB'] * 86400)**2 *
#                                             np.sqrt(1 - params['ECC']**2))

# dEdV = -(2 * np.pi * params['A1'] * v_c * params['ECC']) / \
#     (np.sin(89.35*np.pi/180) * (params['PB'] * 86400)**2 *
#      np.sqrt(1 - params['ECC']**2)**(3/2))

# sigma_V_0 = np.sqrt((dAdV)**2*(A1err)**2 + (didV)**2*(INCerr)**2 +
#                     (dPdV)**2*(PBerr)**2 + (dEdV)**2*(ECCerr)**2)
###############################################################################
# This is testing the validity of a 10min window

# mjd_test_model = np.linspace(np.min(mjd), np.max(mjd), 10000)
# U = get_true_anomaly(mjd_test_model, pars)
# ve_ra, ve_dec = get_earth_velocity(mjd_test_model, pars['RAJ'], pars['DECJ'])
# Anisotropy_Option = True
# veff_model_test = effective_velocity_annual_bilby(
#     xdata=mjd_test_model, d=0.735, s=0.72, k=1, vism_ra=-20, vism_dec=-20,
#     KIN=89.35, KOM=65, TAUEFAC=0, DNUEFAC=0, TAUESKEW=-100, DNUESKEW=-100,
#     alpha=4, psi=65, R=0.01, kwargs=(U, ve_ra, ve_dec, params,
#                                       Anisotropy_Option))

# om = pars['OM'] + pars['OMDOT']*(mjd_test_model - pars['T0'])/365.2425
# # compute orbital phase
# phase = U*180/np.pi + om
# phase = phase % 360
# phase_sort = np.argwhere(phase)
# plt.scatter(phase[phase_sort], veff_model_test[phase_sort])

# #

# Anisotropy_Option = True
# end_mjd = 0
# step_mjd = 0
# start_mjd = np.min(mjd)
# veff_mean = []
# veff_median = []
# veff_middle = []
# for i in range(0, 5000):
#     start_mjd += step_mjd
#     step_mjd = 10/1440
#     end_mjd = start_mjd + step_mjd
#     if end_mjd > np.max(mjd):
#         break
#     mjd_range = np.linspace(start_mjd, end_mjd, 100)
#     U = get_true_anomaly(mjd_range, pars)
#     ve_ra, ve_dec = get_earth_velocity(mjd_range, pars['RAJ'], pars['DECJ'])
#     veff_test = effective_velocity_annual_bilby(
#         xdata=mjd_range, d=0.735, s=0.72, k=1, vism_ra=-20, vism_dec=20,
#         KIN=89.35, KOM=65, TAUEFAC=0, DNUEFAC=0, TAUESKEW=-100, DNUESKEW=-100,
#         alpha=4, psi=45, R=0.4, kwargs=(U, ve_ra, ve_dec, params,
#                                         Anisotropy_Option))
#     veff_mean.append(np.mean(veff_test))
#     veff_median.append(np.median(veff_test))
#     veff_middle.append(veff_test[len(veff_test) // 2])

# veff_mean = np.asarray(veff_mean)
# veff_median = np.asarray(veff_median)
# veff_middle = np.asarray(veff_middle)

# mjd_range = np.linspace(np.max(mjd), np.min(mjd), len(veff_middle))
# U = get_true_anomaly(mjd_range, pars)
# om = pars['OM'] + pars['OMDOT']*(mjd_range - pars['T0'])/365.2425
# # compute orbital phase
# phase = U*180/np.pi + om
# phase = phase % 360
# phase_sort = np.argwhere(phase)

# plt.scatter(phase[phase_sort], abs(veff_mean - veff_middle)[phase_sort])
# plt.scatter(phase[phase_sort], abs(veff_median - veff_middle)[phase_sort])
# plt.show()
# plt.close()

# #

# Anisotropy_Option = True
# veff_mean = []
# veff_median = []
# veff_middle = []
# for i in range(0, len(mjd)):
#     min_mjd = mjd[i] - 5/1440
#     max_mjd = mjd[i] + 5/1440
#     mjd_range = np.linspace(min_mjd, max_mjd, 100)
#     U = get_true_anomaly(mjd_range, pars)
#     ve_ra, ve_dec = get_earth_velocity(mjd_range, pars['RAJ'], pars['DECJ'])
#     veff_test = effective_velocity_annual_bilby(
#         xdata=mjd_range, d=0.735, s=0.72, k=1, vism_ra=-20, vism_dec=20,
#         KIN=89.35, KOM=65, TAUEFAC=0, DNUEFAC=0, TAUESKEW=-100, DNUESKEW=-100,
#         alpha=4, psi=45, R=0.4, kwargs=(U, ve_ra, ve_dec, params,
#                                         Anisotropy_Option))
#     veff_mean.append(np.mean(veff_test))
#     veff_median.append(np.median(veff_test))
#     veff_middle.append(veff_test[len(veff_test) // 2])

# veff_mean = np.asarray(veff_mean)
# veff_median = np.asarray(veff_median)
# veff_middle = np.asarray(veff_middle)

# mjd_range = np.linspace(np.max(mjd), np.min(mjd), len(veff_middle))
# U = get_true_anomaly(mjd_range, pars)
# om = pars['OM'] + pars['OMDOT']*(mjd_range - pars['T0'])/365.2425
# # compute orbital phase
# phase = U*180/np.pi + om
# phase = phase % 360
# phase_sort = np.argwhere(phase)

# plt.scatter(phase[phase_sort], abs(veff_mean - veff_middle)[phase_sort])
# plt.scatter(phase[phase_sort], abs(veff_median - veff_middle)[phase_sort])
# plt.show()
# plt.close()

#

# min_mjd = 59790.38797054979 - 75/1440
# max_mjd = 59790.38797054979 + 75/1440
# mjd_range = np.linspace(min_mjd, max_mjd, 1500)
# U = get_true_anomaly(mjd_range, pars)
# ve_ra, ve_dec = get_earth_velocity(mjd_range, pars['RAJ'], pars['DECJ'])
# veff_test = effective_velocity_annual_bilby(
#          xdata=mjd_range, d=0.735, s=0.72, k=1, vism_ra=-20, vism_dec=20,
#          KIN=89.35, KOM=65, TAUEFAC=0, DNUEFAC=0, TAUESKEW=-100, DNUESKEW=-100,
#          alpha=4, psi=45, R=0.4, kwargs=(U, ve_ra, ve_dec, params,
#                                          Anisotropy_Option))
# U = get_true_anomaly(mjd_range, pars)
# om = pars['OM'] + pars['OMDOT']*(mjd_range - pars['T0'])/365.2425
# # compute orbital phase
# phases = U*180/np.pi + om
# phases = phases % 360

# means = np.asarray([np.mean(veff_test[i:i+100]) for i in range(0, len(veff_test), 100)])
# # medians = np.asarray([np.median(veff_test[i:i+100]) for i in range(0, len(veff_test), 100)])
# phases2 = np.asarray([phases[i] + (360/150)*5 for i in range(0, len(veff_test), 100)])
# veffs = []
# for i in range(0, len(phases2)):
#     veffs.append(veff_test[np.argmin(abs(phases - phases2[i]))])
# veffs = np.asarray(veffs)

# plt.plot(phases, veff_test, c='C0')
# yl = plt.ylim()
# plt.scatter(phases2, means, c='C1')
# # plt.plot(phases2, means, c='C1')
# # plt.scatter(phases2, medians, c='C2', alpha=0.7)
# # plt.plot(phases2, medians, c='C2', alpha=0.7)
# plt.xlabel("Orbital Phase (degrees)")
# plt.ylabel(r"Scintillation Velocity (km$\,$s$^{-1}$)")
# plt.show()
# plt.close()

# plt.scatter(phases2, means - veffs, c='C1')
# # plt.plot(phases2, abs(means - veffs), c='C1')
# # plt.scatter(phases2, abs(medians - veffs), c='C2', alpha=0.7)
# # plt.plot(phases2, abs(medians - veffs), c='C2', alpha=0.7)
# yl = plt.ylim()
# plt.xlabel("Orbital Phase (degrees)")
# plt.ylabel(r"Scintillation Velocity (km$\,$s$^{-1}$)")
# plt.show()
# plt.close()

# #


# def veldiff_orbital(xdata):

#     Anisotropy_Option = True
#     veff_mean = []
#     veff_middle = []
#     for i in range(0, len(xdata)):
#         min_mjd = xdata[i] - 5/1440
#         max_mjd = xdata[i] + 5/1440
#         mjd_range = np.linspace(min_mjd, max_mjd, 100)
        # U = get_true_anomaly(mjd_range, pars)
        # ve_ra, ve_dec = get_earth_velocity(mjd_range, pars['RAJ'],
        #                                     pars['DECJ'])
        # kwargs = {"U": U, "ve_ra": ve_ra, "ve_dec": ve_dec, "params": params,
        #           "Anisotropy_Option": Anisotropy_Option}
#         veff_test = effective_velocity_annual_bilby(
#             xdata=mjd_range, d=0.735, s=0.72, k=1, vism_ra=-20, vism_dec=20,
#             KIN=89.35, KOM=65, TAUEFAC=0, DNUEFAC=0, TAUESKEW=-100,
#             DNUESKEW=-100, alpha=4, psi=45, R=0.4, **kwargs)
#         veff_mean.append(np.mean(veff_test))
#         veff_middle.append(veff_test[len(veff_test) // 2])
#     veff_middle = np.asarray(veff_middle)
#     veff_mean = np.asarray(veff_mean)
#     veldiffs = veff_middle - veff_mean
#     return veldiffs


# #
###############################################################################
# Creating some datafiles to feed into the likelihood function


mjd_ranges = []
U_ranges = []
ve_ra_ranges = []
ve_dec_ranges = []
for i in range(0, len(mjd)):
    mjd_step = 5/1440
    mjd_range = np.linspace(mjd[i] - mjd_step, mjd[i] + mjd_step, 11)
    mjd_ranges.append(mjd_range)
    U_ranges.append(get_true_anomaly(mjd_range, pars))
    ve_ra, ve_dec = get_earth_velocity(mjd_range, pars['RAJ'], pars['DECJ'])
    ve_ra_ranges.append(ve_ra)
    ve_dec_ranges.append(ve_dec)
mjd_ranges = np.asarray(mjd_ranges).flatten()  
U_ranges = np.asarray(U_ranges).flatten()  
ve_ra_ranges = np.asarray(ve_ra_ranges).flatten()  
ve_dec_ranges = np.asarray(ve_dec_ranges).flatten()  
np.savetxt(datadir + '10minMJD.txt', mjd_ranges, delimiter=',')
np.savetxt(datadir + '10minU.txt', U_ranges, delimiter=',')
np.savetxt(datadir + '10minVE_RA.txt', ve_ra_ranges, delimiter=',')
np.savetxt(datadir + '10minVE_DEC.txt', ve_dec_ranges, delimiter=',')

###############################################################################
