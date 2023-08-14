#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 14:23:33 2023

@author: jacobaskew
"""

###############################################################################
# Importing neccessary things
from scintools.scint_utils import read_par, get_ssb_delay, \
    get_earth_velocity, get_true_anomaly, scint_velocity, pars_to_params
from scintools.scint_models import effective_velocity_annual
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
import bilby
from astropy.time import Time
###############################################################################


def effective_velocity_annual_alternate(params, true_anomaly, vearth_ra,
                                        vearth_dec, mjd=None):
    """
    Effective velocity with annual and pulsar terms
        Note: Does NOT include IISM velocity, but returns veff in IISM frame
    """
    # Define some constants
    v_c = 299792.458  # km/s
    kmpkpc = 3.085677581e16
    secperyr = 86400*365.2425
    masrad = np.pi/(3600*180*1000)

    # tempo2 parameters from par file in capitals
    if 'PB' in params.keys():
        A1 = params['A1']  # projected semi-major axis in lt-s
        PB = params['PB']  # orbital period in days
        ECC = params['ECC']  # orbital eccentricity
        OM = params['OM'] * np.pi/180  # longitude of periastron rad
        if 'OMDOT' in params.keys():
            if mjd is None:
                print('Warning, OMDOT present but no mjd for calculation')
                omega = OM
            else:
                omega = OM + \
                    params['OMDOT']*np.pi/180*(mjd-params['T0'])/365.2425
        else:
            omega = OM
        # Note: fifth Keplerian param T0 used in true anomaly calculation
        if 'KIN' in params.keys():
            INC = params['KIN']*np.pi/180  # inclination
        elif 'COSI' in params.keys():
            INC = np.arccos(params['COSI'])
        elif 'SINI' in params.keys():
            INC = np.arcsin(params['SINI'])
        else:
            print('Warning: inclination parameter (KIN, COSI, or SINI) ' +
                  'not found')

        if 'sense' in params.keys():
            sense = params['sense']
            if sense < 0.5:  # KIN < 90
                if INC > np.pi/2:
                    INC = np.pi - INC
            if sense >= 0.5:  # KIN > 90
                if INC < np.pi/2:
                    INC = np.pi - INC

        KOM = params['KOM']*np.pi/180  # longitude ascending node

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

    if 'PMRA' in params.keys():
        PMRA = params['PMRA']  # proper motion in RA
        PMDEC = params['PMDEC']  # proper motion in DEC
    else:
        PMRA = 0
        PMDEC = 0

    if 'vism_ra' in params.keys():
        vism_ra = params['vism_ra']  # proper motion in RA
        vism_dec = params['vism_dec']  # proper motion in DEC
    else:
        vism_ra = 0
        vism_dec = 0

    # other parameters in lower-case
    s = params['s']  # fractional screen distance
    d = params['d']  # pulsar distance in kpc
    d = d * kmpkpc  # distance in km

    pmra_v = PMRA * masrad * d / secperyr
    pmdec_v = PMDEC * masrad * d / secperyr

    # Rotate pulsar velocity into RA/DEC
    vp_ra = np.sin(KOM) * vp_x + np.cos(KOM) * vp_y
    vp_dec = np.cos(KOM) * vp_x - np.sin(KOM) * vp_y

    # find total effective velocity in RA and DEC, defined at earth
    veff_ra = (s * vearth_ra + (1 - s) * (vp_ra + pmra_v) - vism_ra) / s
    veff_dec = (s * vearth_dec + (1 - s) * (vp_dec + pmdec_v) - vism_dec) / s

    return veff_ra, veff_dec, vp_ra, vp_dec


###############################################################################


def effective_velocity_annual_bilby(xdata, D, s, vism_ra, vism_dec, KIN, KOM,
                                    EFAC, ESKEW, **kwargs):
    """
    Effective velocity with annual and pulsar terms
        Note: Does NOT include IISM velocity, but returns veff in IISM frame
    """
    # Define the initial parameters
    params_ = dict(params)
    params_['d'] = D
    params_['s'] = s
    params_['KOM'] = KOM
    params_['KIN'] = KIN
    params_['vism_ra'] = vism_ra
    params_['vism_dec'] = vism_dec

    mjd = xdata
    mjd_sort = np.argsort(mjd)
    true_anomaly = U[mjd_sort]
    vearth_ra = ve_ra[mjd_sort]
    vearth_dec = ve_dec[mjd_sort]

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
    d = d * kmpkpc  # distance in km

    pmra_v = PMRA * masrad * d / secperyr
    pmdec_v = PMDEC * masrad * d / secperyr

    # Rotate pulsar velocity into RA/DEC
    vp_ra = np.sin(KOM) * vp_x + np.cos(KOM) * vp_y
    vp_dec = np.cos(KOM) * vp_x - np.sin(KOM) * vp_y

    # find total effective velocity in RA and DEC
    veff_ra = s * vearth_ra + (1 - s) * (vp_ra + pmra_v) - vism_ra
    veff_dec = s * vearth_dec + (1 - s) * (vp_dec + pmdec_v) - vism_dec

    model = np.sqrt(veff_ra**2 + veff_dec**2) / s
    model = np.float64(model)

    return model


###############################################################################


def effective_velocity_annual_anisotropy_bilby(xdata, D, s, k, vism_psi, psi,
                                               R, KIN, KOM, TAUEFAC, DNUEFAC,
                                               TAUESKEW, DNUESKEW,
                                               **kwargs):
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
    params_['vism_psi'] = vism_psi
    params_['psi'] = psi
    params_['R'] = R
    params_['KOM'] = KOM
    params_['KIN'] = KIN

    mjd = xdata
    mjd_sort = np.argsort(mjd)
    true_anomaly = U[mjd_sort]
    vearth_ra = ve_ra[mjd_sort]
    vearth_dec = ve_dec[mjd_sort]

    veff_ra, veff_dec, vp_ra, vp_dec = \
        effective_velocity_annual(params_, true_anomaly, vearth_ra, vearth_dec,
                                  mjd=mjd)

    r = params_['R']  # axial ratio parameter, some complicated relationship
    psi = params_['psi'] * np.pi / 180  # anisotropy angle
    kappa = params_['kappa']

    cosa = np.cos(2 * psi)
    sina = np.sin(2 * psi)

    # quadratic coefficients
    a = (1 - r * cosa) / np.sqrt(1 - r**2)
    b = (1 + r * cosa) / np.sqrt(1 - r**2)
    c = -2 * r * sina / np.sqrt(1 - r**2)

    # other parameters in lower-case
    kmpkpc = 3.085677581e16
    s = params_['s']  # fractional screen distance
    d = params_['d']  # pulsar distance in kpc
    d = d * kmpkpc  # distance in km

    # coefficient to match model with data
    coeff = 1 / np.sqrt(2 * d * (1 - s) / s)

    veff = kappa * (np.sqrt(a*veff_dec**2 + b*veff_ra**2 +
                            c*veff_ra*veff_dec))
    model = coeff * veff / s
    model = np.float64(model)

    return model


###############################################################################
weights_2dacf = None
Modelling = True
FullData = False
Anisotropy_Option = True
GetPhaseNviss = False
nlive = 200
resume = False
sense = False  # True = Flipped, False = not-flipped
input_KIN = 89.35  # Kramer et al. 2021
input_KIN_err = 0.05
# Other values ... earlier timing 88.69 + 0.5 - 0.76 ... eclipse 89.3 +- 0.1
#  earlier scintillation 88.1 +- 0.5
if sense:
    input_KIN = 180 - input_KIN
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
U = get_true_anomaly(mjd, pars)
ve_ra, ve_dec = get_earth_velocity(mjd, pars['RAJ'], pars['DECJ'])

# Modelling
if Modelling:

    label = 'test'
    wd0 = "/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/New/"
    outdir = wd0 + "Modelling"
    if Anisotropy_Option:
        outdir += '/Anisotropic'
        priors = dict(D=bilby.core.prior.analytical.DeltaFunction(0.735, 'D'),
                      s=bilby.core.prior.Uniform(0, 1, 's'),
                      k=bilby.core.prior.Uniform(0, 10, 'k'),
                      vism_psi=bilby.core.prior.Uniform(-200, 200,
                                                        'vism_psi'),
                      psi=bilby.core.prior.Uniform(0, 180, 'psi',
                                                   boundary='periodic'),
                      R=bilby.core.prior.Uniform(0, 1, 'R'),
                      KIN=bilby.core.prior.analytical.DeltaFunction(input_KIN,
                                                                    'KIN'),
                      # KIN=bilby.core.prior.Uniform(0, 180, 'KIN',
                      #                              boundary='periodic'),
                      KOM=bilby.core.prior.Uniform(0, 360, 'KOM',
                                                   boundary='periodic'),
                      # KOM=bilby.core.prior.analytical.DeltaFunction(53,
                      #                                  name='KOM'),
                      TAUESKEW=bilby.core.prior.Uniform(-2, 2, 'TAUESKEW'),
                      DNUESKEW=bilby.core.prior.Uniform(-2, 2, 'DNUESKEW'),
                      TAUEFAC=bilby.core.prior.Uniform(-2, 2, 'TAUEFAC'),
                      DNUEFAC=bilby.core.prior.Uniform(-2, 2, 'DNUEFAC'))
        likelihood = \
            bilby.likelihood.GaussianLikelihood(
                x=mjd, y=viss, func=effective_velocity_annual_anisotropy_bilby,
                freq=freq, tau=tau, dnu=dnu, tauerr=tauerr, dnuerr=dnuerr,
                sigma=visserr, kwargs=(U, ve_ra, ve_dec, params))
    else:
        outdir += '/Isotropic'
        priors = dict(D=bilby.core.prior.analytical.DeltaFunction(0.735, 'D'),
                      s=bilby.core.prior.Uniform(0, 1, 's'),
                      vism_ra=bilby.core.prior.Uniform(-200, 200, 'vism_ra'),
                      vism_dec=bilby.core.prior.Uniform(-200, 200, 'vism_dec'),
                      # KIN=bilby.core.prior.Uniform(0, 180, 'KIN',
                      #                              boundary='periodic'),
                      KIN=bilby.core.prior.analytical.DeltaFunction(input_KIN,
                                                                    'KIN'),
                      KOM=bilby.core.prior.Uniform(0, 360, 'KOM',
                                                   boundary='periodic'),
                      # KOM=bilby.core.prior.analytical.DeltaFunction(53,
                      #                                  name='KOM'),
                      TAUESKEW=bilby.core.prior.Uniform(-2, 2, 'TAUESKEW'),
                      DNUESKEW=bilby.core.prior.Uniform(-2, 2, 'DNUESKEW'),
                      TAUEFAC=bilby.core.prior.Uniform(-2, 2, 'TAUEFAC'),
                      DNUEFAC=bilby.core.prior.Uniform(-2, 2, 'DNUEFAC'))
        likelihood = \
            bilby.likelihood.GaussianLikelihood(
                x=mjd, y=viss, func=effective_velocity_annual_bilby, freq=freq,
                tau=tau, dnu=dnu, tauerr=tauerr, dnuerr=dnuerr,
                sigma=visserr, kwargs=(U, ve_ra, ve_dec, params))

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
    d_result = result.posterior['D'][NUM]
    derr_result = np.std(result.posterior['D'].values)
    params.add('d', value=d_result, vary=False)
    params.add('derr', value=derr_result, vary=False)
    s_result = result.posterior['s'][NUM]
    serr_result = np.std(result.posterior['s'].values)
    params.add('s', value=s_result, vary=False)
    params.add('serr', value=serr_result, vary=False)
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
    if Anisotropy_Option:
        k_result = result.posterior['k'][NUM]
        kerr_result = np.std(result.posterior['k'].values)
        params.add('k', value=k_result, vary=False)
        params.add('kerr', value=kerr_result, vary=False)
        vism_psi_result = result.posterior['vism_psi'][NUM]
        vism_psierr_result = np.std(result.posterior['vism_psi'].values)
        params.add('vism_psi', value=vism_psi_result, vary=False)
        params.add('vism_psierr', value=vism_psierr_result, vary=False)
        psi_result = result.posterior['psi'][NUM]
        psierr_result = np.std(result.posterior['psi'].values)
        params.add('psi', value=psi_result, vary=False)
        params.add('psierr', value=psierr_result, vary=False)
        R_result = result.posterior['R'][NUM]
        Rerr_result = np.std(result.posterior['R'].values)
        params.add('R', value=R_result, vary=False)
        params.add('Rerr', value=Rerr_result, vary=False)
    else:
        vism_ra_result = result.posterior['vism_ra'][NUM]
        vism_raerr_result = np.std(result.posterior['vism_ra'].values)
        params.add('vism_ra', value=vism_ra_result, vary=False)
        params.add('vism_raerr', value=vism_raerr_result, vary=False)
        vism_dec_result = result.posterior['vism_dec'][NUM]
        vism_decerr_result = np.std(result.posterior['vism_dec'].values)
        params.add('vism_dec', value=vism_dec_result, vary=False)
        params.add('vism_decerr', value=vism_decerr_result, vary=False)
    # Recalculating the error bars
    alpha = 4
    Aiss = 2.78*10**4
    new_freq = freq / 1e3
    New_dnuerr = np.sqrt((dnuerr * 10**DNUEFAC_result)**2 +
                         ((10**DNUESKEW_result)*(new_freq/1)**alpha)**2)
    New_tauerr = np.sqrt((tauerr * 10**TAUEFAC_result)**2 +
                         ((10**TAUESKEW_result)*(new_freq/1)**(alpha/2))**2)
    New_viss, New_visserr = scint_velocity(params, dnu, tau, freq, New_dnuerr,
                                           New_tauerr, a=Aiss)
    mjd_year = Time(mjd, format='mjd').byear
    mjd_annual = mjd % 365.2425
    if os.path.exists('/Users/jacobaskew/Desktop/Model_mjdData.txt'):
        Model_mjd = np.loadtxt('/Users/jacobaskew/Desktop/Model_mjdData.txt',
                               dtype='float')
        Model_phase = \
            np.loadtxt('/Users/jacobaskew/Desktop/Model_phaseData.txt',
                       dtype='float')
        Model_U = np.loadtxt('/Users/jacobaskew/Desktop/Model_UData.txt',
                             dtype='float')
        Model_vearth_ra = np.loadtxt(
            '/Users/jacobaskew/Desktop/Model_vearth_raData.txt', dtype='float')
        Model_vearth_dec = np.loadtxt(
            '/Users/jacobaskew/Desktop/Model_vearth_decData.txt',
            dtype='float')
    else:
        Model_mjd = np.linspace(np.min(mjd), np.max(mjd), 10000)
        Model_mjd_year = Time(Model_mjd, format='mjd').byear
        Model_mjd_annual = Model_mjd % 365.2425
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
        np.savetxt('/Users/jacobaskew/Desktop/Model_mjdData.txt', Model_mjd,
                   delimiter=',')
        np.savetxt('/Users/jacobaskew/Desktop/Model_phaseData.txt',
                   Model_phase, delimiter=',')
        np.savetxt('/Users/jacobaskew/Desktop/Model_UData.txt', Model_U,
                   delimiter=',')
        np.savetxt('/Users/jacobaskew/Desktop/Model_vearth_raData.txt',
                   Model_vearth_ra, delimiter=',')
        np.savetxt('/Users/jacobaskew/Desktop/Model_vearth_decData.txt',
                   Model_vearth_dec, delimiter=',')
    #
    Model_veff_ra, Model_veff_dec, Model_vp_ra, Model_vp_dec = \
        effective_velocity_annual_alternate(params, true_anomaly=Model_U,
                                            vearth_ra=Model_vearth_ra,
                                            vearth_dec=Model_vearth_dec,
                                            mjd=Model_mjd)
    Model_VP_ra, Model_VP_dec, Model_VPp_ra, Model_VPp_dec = \
        effective_velocity_annual_alternate(params, true_anomaly=Model_U,
                                            vearth_ra=0,
                                            vearth_dec=0,
                                            mjd=Model_mjd)
    params.add('A1', value=0, vary=False)
    Model_VE_ra, Model_VE_dec, Model_VEp_ra, Model_VEp_dec = \
        effective_velocity_annual_alternate(params, true_anomaly=Model_U,
                                            vearth_ra=Model_vearth_ra,
                                            vearth_dec=Model_vearth_dec,
                                            mjd=Model_mjd)
    params.add('A1', value=1.415032, vary=False)

    Model_VP_veff = np.sqrt((Model_VP_ra)**2 + (Model_VP_dec)**2)
    Model_VE_veff = np.sqrt((Model_VE_ra)**2 + (Model_VE_dec)**2)
    Model_veff = np.sqrt((Model_veff_ra)**2 + (Model_veff_dec)**2)

    Model_VP_veff_only = Model_veff / -(Model_VE_veff)
    Model_VE_veff_only = Model_veff / -(Model_VP_veff)
    #
    veff_ra, veff_dec, vp_ra, vp_dec = \
        effective_velocity_annual_alternate(params, true_anomaly=U,
                                            vearth_ra=ve_ra,
                                            vearth_dec=ve_dec,
                                            mjd=mjd)
    VP_ra, VP_dec, VPp_ra, VPp_dec = \
        effective_velocity_annual_alternate(params, true_anomaly=U,
                                            vearth_ra=0,
                                            vearth_dec=0,
                                            mjd=mjd)
    params.add('A1', value=0, vary=False)
    VE_ra, VE_dec, VEp_ra, VEp_dec = \
        effective_velocity_annual_alternate(params, true_anomaly=U,
                                            vearth_ra=ve_ra,
                                            vearth_dec=ve_dec,
                                            mjd=mjd)
    params.add('A1', value=1.415032, vary=False)

    VP_veff = np.sqrt((VP_ra)**2 + (VP_dec)**2)
    VE_veff = np.sqrt((VE_ra)**2 + (VE_dec)**2)
    veff = np.sqrt((veff_ra)**2 + (veff_dec)**2)

    VP_veff_only = veff / -(VE_veff)
    VE_veff_only = veff / -(VP_veff)
    #
    Model_mjd_sort = np.argsort(Model_mjd)
    mjd_sort = np.argsort(mjd)
    Model_phase_sort = np.argsort(Model_phase)
    phase_sort = np.argsort(phase)
    Model_mjd_annual_sort = np.argsort(Model_mjd_annual)
    mjd_annual_sort = np.argsort(mjd_annual)

    # Plotting

    Size = 80*np.pi  # Determines the size of the datapoints used
    font = {'size': 28}
    matplotlib.rc('font', **font)

    fig = plt.figure(figsize=(20, 10))
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    ax = fig.add_subplot(1, 1, 1)
    # ax.scatter(mjd_year[mjd_sort], viss[mjd_sort], c='C0', s=Size, alpha=0.7)
    ax.errorbar(mjd_year[mjd_sort], viss[mjd_sort], yerr=visserr[mjd_sort],
                fmt='o', ecolor='k', elinewidth=2, capsize=3, alpha=0.55)
    ax.plot(Model_mjd_year[Model_mjd_sort], Model_veff[Model_mjd_sort], c='k',
            alpha=0.2)
    xl = plt.xlim()
    ax.set_xlabel('Year')
    ax.set_ylabel(r'Scintillation Velocity (km$\,$s$^{-1}$)')
    ax.set_xlim(xl)
    plt.savefig("/Users/jacobaskew/Desktop/Viss_test.png", dpi=400)
    plt.savefig("/Users/jacobaskew/Desktop/Viss_test.pdf", dpi=400)
    plt.tight_layout()
    plt.show()
    plt.close()
    # #
    fig = plt.figure(figsize=(20, 10))
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    #
    ax = fig.add_subplot(1, 1, 1)
    # ax.scatter(phase[phase_sort], VP_veff_only[phase_sort], c='C0', s=Size,
    #            alpha=0.7)
    ax.errorbar(phase[phase_sort], VP_veff_only[phase_sort],
                yerr=visserr[phase_sort],
                fmt='o', ecolor='k', elinewidth=2, capsize=3, alpha=0.55)
    ax.plot(Model_phase[Model_phase_sort],
            Model_VP_veff_only[Model_phase_sort], c='k', alpha=0.2)
    xl = plt.xlim()
    ax.set_xlabel('Orbital Phase (degrees)')
    ax.set_ylabel(r'Scintillation Velocity (km$\,$s$^{-1}$)')
    ax.set_xlim(xl)
    plt.savefig("/Users/jacobaskew/Desktop/Viss_orbital_test.png", dpi=400)
    plt.savefig("/Users/jacobaskew/Desktop/Viss_orbital_test.pdf", dpi=400)
    plt.tight_layout()
    plt.show()
    plt.close()
    # #
    fig = plt.figure(figsize=(20, 10))
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    #
    ax = fig.add_subplot(1, 1, 1)
    # ax.scatter(mjd_annual[mjd_annual_sort], VE_veff_only[mjd_annual_sort],
    #            c='C0', s=Size, alpha=0.7)
    ax.errorbar(mjd_annual[mjd_annual_sort], VE_veff_only[mjd_annual_sort],
                yerr=visserr[mjd_annual_sort], fmt='o', ecolor='k',
                elinewidth=2, capsize=3, alpha=0.55)
    ax.plot(Model_mjd_annual[Model_mjd_annual_sort],
            Model_VE_veff_only[Model_mjd_annual_sort], c='k', alpha=0.2)
    xl = plt.xlim()
    ax.set_xlabel('Annual Phase (days)')
    ax.set_ylabel(r'Scintillation Velocity (km$\,$s$^{-1}$)')
    ax.set_xlim(xl)
    plt.savefig("/Users/jacobaskew/Desktop/Viss_annual_test.png", dpi=400)
    plt.savefig("/Users/jacobaskew/Desktop/Viss_annual_test.pdf", dpi=400)
    plt.tight_layout()
    plt.show()
    plt.close()

    # Residuals
    fig = plt.figure(figsize=(20, 10))
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    ax = fig.add_subplot(1, 1, 1)
    ax.errorbar(phase[phase_sort], (viss - veff - VE_veff_only)[phase_sort],
                yerr=visserr[phase_sort],
                fmt='o', ecolor='k', elinewidth=2, capsize=3, alpha=0.55)
    ax.plot([np.min(phase), np.max(phase)], [0, 0], c='C3')
    xl = plt.xlim()
    ax.set_xlabel('Orbital Phase (degrees)')
    ax.set_ylabel(r'Scintillation Velocity (km$\,$s$^{-1}$)')
    ax.set_xlim(xl)
    plt.savefig("/Users/jacobaskew/Desktop/Viss_residual_orbital_test.png",
                dpi=400)
    plt.savefig("/Users/jacobaskew/Desktop/Viss_residual_orbital_test.pdf",
                dpi=400)
    plt.tight_layout()
    plt.show()
    plt.close()
    # #
    fig = plt.figure(figsize=(20, 10))
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    ax = fig.add_subplot(1, 1, 1)
    ax.errorbar(mjd_annual[mjd_annual_sort],
                (viss - veff - VP_veff_only)[mjd_annual_sort],
                yerr=visserr[mjd_annual_sort],
                fmt='o', ecolor='k', elinewidth=2, capsize=3, alpha=0.55)
    ax.plot([np.min(mjd_annual), np.max(mjd_annual)], [0, 0], c='C3')
    xl = plt.xlim()
    ax.set_xlabel('Annual Phase (days)')
    ax.set_ylabel(r'Scintillation Velocity (km$\,$s$^{-1}$)')
    ax.set_xlim(xl)
    plt.savefig("/Users/jacobaskew/Desktop/Viss_residual_annual_test.png",
                dpi=400)
    plt.savefig("/Users/jacobaskew/Desktop/Viss_residual_annual_test.pdf",
                dpi=400)
    plt.tight_layout()
    plt.show()
    plt.close()
    # #
    fig = plt.figure(figsize=(20, 10))
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    ax = fig.add_subplot(1, 1, 1)
    ax.errorbar(mjd_year[mjd_sort], (viss - veff)[mjd_sort],
                yerr=visserr[mjd_sort],
                fmt='o', ecolor='k', elinewidth=2, capsize=3, alpha=0.55)
    ax.plot([np.min(mjd_year), np.max(mjd_year)], [0, 0], c='C3')
    xl = plt.xlim()
    ax.set_xlabel('Year')
    ax.set_ylabel(r'Scintillation Velocity (km$\,$s$^{-1}$)')
    ax.set_xlim(xl)
    plt.savefig("/Users/jacobaskew/Desktop/Viss_residual_test.png",
                dpi=400)
    plt.savefig("/Users/jacobaskew/Desktop/Viss_residual_test.pdf",
                dpi=400)
    plt.tight_layout()
    plt.show()
    plt.close()

###############################################################################
