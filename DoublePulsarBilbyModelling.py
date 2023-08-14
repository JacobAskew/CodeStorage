#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 14:31:51 2023

@author: jacobaskew
"""
###############################################################################
import bilby
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from scintools.scint_utils import read_par, get_earth_velocity, \
    get_true_anomaly, pars_to_params, scint_velocity, read_results, \
    float_array_from_dict, get_ssb_delay
from scintools.scint_models import effective_velocity_annual
###############################################################################
nlive = 200
resume = False
freq_bin = 30
time_bin = 10
filtered_tau = 10
filtered_dnu = 10
par_dir = '/Users/jacobaskew/Desktop/Swinburne_Daniel/ParFiles/'
psrname = 'J0737-3039A'
pars = read_par(str(par_dir) + str(psrname) + '.par')
params = pars_to_params(pars)
###############################################################################


def effective_velocity_annual_bilby(xdata, D, s, KOM, KIN, vism_ra, vism_dec):
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
    true_anomaly = get_true_anomaly(mjd, pars)
    vearth_ra, vearth_dec = get_earth_velocity(mjd, pars['RAJ'],
                                               pars['DECJ'])

    # Define some constants
    v_c = 299792.458  # km/s
    kmpkpc = 3.085677581e16
    secperyr = 86400*365.2425
    masrad = np.pi/(3600*180*1000)

# tempo2 parameters from par file in capitals
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
    if 'PMRA' in params_.keys():
        PMRA = params_['PMRA']  # proper motion in RA
        PMDEC = params_['PMDEC']  # proper motion in DEC
    else:
        PMRA = 0
        PMDEC = 0

    # other parameters in lower-case
    D = D * kmpkpc  # distance in km

    pmra_v = PMRA * masrad * D / secperyr
    pmdec_v = PMDEC * masrad * D / secperyr

    # Rotate pulsar velocity into RA/DEC
    vp_ra = np.sin(KOM) * vp_x + np.cos(KOM) * vp_y
    vp_dec = np.cos(KOM) * vp_x - np.sin(KOM) * vp_y

    # find total effective velocity in RA and DEC
    veff_ra = s * vearth_ra + (1 - s) * (vp_ra + pmra_v)
    veff_dec = s * vearth_dec + (1 - s) * (vp_dec + pmdec_v)

    model = np.sqrt(veff_ra**2 + veff_dec**2)
    model = model.astype('float64')

    return model


###############################################################################
wd0 = '/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/'
wd = wd0+'New/'
outdir = wd + 'DataFiles/'
filedir2 = outdir
psrname = 'J0737-3039A'
outfile_total = str(filedir2)+str(psrname)+'_freq' + \
    str(freq_bin)+'_time'+str(time_bin) + \
    '_ScintillationResults_UHF_Total.txt'
# file1 = "J0737-3039A_2022-12-30_freq30_time10_ScintillationResults_UHF.txt"
# outfile_total = wd + "DataFiles/2022-12-30/" + file1
params = read_results(outfile_total)

pars = read_par(str(par_dir) + str(psrname) + '.par')

# Read in arrays
mjd = float_array_from_dict(params, 'mjd')
df = float_array_from_dict(params, 'df')  # channel bandwidth
dnu = float_array_from_dict(params, 'dnu')  # scint bandwidth
# dnu_est = float_array_from_dict(params, 'dnu_est')
dnuerr = float_array_from_dict(params, 'dnuerr')
tau = float_array_from_dict(params, 'tau')
tauerr = float_array_from_dict(params, 'tauerr')
freq = float_array_from_dict(params, 'freq')
bw = float_array_from_dict(params, 'bw')
name = np.asarray(params['name'])
# scintle_num = float_array_from_dict(params, 'scintle_num')
tobs = float_array_from_dict(params, 'tobs')  # tobs in second
# rcvrs = np.array([rcvr[0] for rcvr in params['name']])
scint_param_method = np.asarray(params['scint_param_method'])

# Sort by MJD
sort_ind = np.argsort(mjd)

df = np.array(df[sort_ind]).squeeze()
dnu = np.array(dnu[sort_ind]).squeeze()
# dnu_est = np.array(dnu_est[sort_ind]).squeeze()
dnuerr = np.array(dnuerr[sort_ind]).squeeze()
tau = np.array(tau[sort_ind]).squeeze()
tauerr = np.array(tauerr[sort_ind]).squeeze()
mjd = np.array(mjd[sort_ind]).squeeze()
# rcvrs = np.array(rcvrs[sort_ind]).squeeze()
freq = np.array(freq[sort_ind]).squeeze()
tobs = np.array(tobs[sort_ind]).squeeze()
name = np.array(name[sort_ind]).squeeze()
# scintle_num = np.array(scintle_num[sort_ind]).squeeze()
bw = np.array(bw[sort_ind]).squeeze()
scint_param_method = np.array(scint_param_method[sort_ind]).squeeze()

# Used to filter the data
indicies = np.argwhere((tauerr < filtered_tau*tau) *
                       (dnuerr < filtered_dnu*dnu) *
                       (scint_param_method == "acf2d_approx"))

df = df[indicies].squeeze()
dnu = dnu[indicies].squeeze()
# dnu_est = dnu_est[indicies].squeeze()
dnuerr = dnuerr[indicies].squeeze()
tau = tau[indicies].squeeze()
tauerr = tauerr[indicies].squeeze()
mjd = mjd[indicies].squeeze()
# rcvrs = rcvrs[indicies].squeeze()
freq = freq[indicies].squeeze()
tobs = tobs[indicies].squeeze()
name = name[indicies].squeeze()
# scintle_num = scintle_num[indicies].squeeze()
bw = bw[indicies].squeeze()
scint_param_method = scint_param_method[indicies].squeeze()

# # Making the size of the new array very small
# jump = int(len(mjd) / 10)
# indicies = np.arange(0, len(mjd), jump)

# df = df[indicies].squeeze()
# dnu = dnu[indicies].squeeze()
# # dnu_est = dnu_est[indicies].squeeze()
# dnuerr = dnuerr[indicies].squeeze()
# tau = tau[indicies].squeeze()
# tauerr = tauerr[indicies].squeeze()
# mjd = mjd[indicies].squeeze()
# # rcvrs = rcvrs[indicies].squeeze()
# freq = freq[indicies].squeeze()
# tobs = tobs[indicies].squeeze()
# name = name[indicies].squeeze()
# # scintle_num = scintle_num[indicies].squeeze()
# bw = bw[indicies].squeeze()
# scint_param_method = scint_param_method[indicies].squeeze()

#
unique_mjd = np.unique(mjd)
unique_mjd_len = len(unique_mjd)
median_dnu = np.zeros((unique_mjd_len, 1))
median_dnuerr = np.zeros((unique_mjd_len, 1))
median_tau = np.zeros((unique_mjd_len, 1))
median_tauerr = np.zeros((unique_mjd_len, 1))
median_freq = np.zeros((unique_mjd_len, 1))
# for i in range(0, unique_mjd_len):
#     for ii in range(0, len(mjd)):
#         if unique_mjd[i] == mjd[ii]:
#             median_dnu = np.median()

for i in range(0, unique_mjd_len):
    median_dnu[i, :] = np.median(dnu[np.argwhere(mjd == unique_mjd[i])])
    median_dnuerr[i, :] = np.median(dnuerr[np.argwhere(mjd == unique_mjd[i])])
    median_tau[i, :] = np.median(tau[np.argwhere(mjd == unique_mjd[i])])
    median_tauerr[i, :] = np.median(tauerr[np.argwhere(mjd == unique_mjd[i])])
    median_freq[i, :] = np.median(freq[np.argwhere(mjd == unique_mjd[i])])

#

mjd_annual = mjd % 365.2425
print('Getting SSB delays')
ssb_delays = get_ssb_delay(mjd, pars['RAJ'], pars['DECJ'])
mjd += np.divide(ssb_delays, 86400)  # add ssb delay

median_mjd_annual = unique_mjd % 365.2425
print('Getting SSB delays')
median_ssb_delays = get_ssb_delay(unique_mjd, pars['RAJ'], pars['DECJ'])
unique_mjd += np.divide(median_ssb_delays, 86400)  # add ssb delay

"""
Model Viss
"""
print('Getting Earth velocity')
vearth_ra, vearth_dec = get_earth_velocity(mjd, pars['RAJ'],
                                           pars['DECJ'])
print('Getting true anomaly')
true_anomaly = get_true_anomaly(mjd, pars)

vearth_ra = vearth_ra.squeeze()
vearth_dec = vearth_dec.squeeze()

om = pars['OM'] + pars['OMDOT']*(mjd - pars['T0'])/365.2425
# compute orbital phase
phase = true_anomaly*180/np.pi + om
phase = phase % 360

print('Getting Earth velocity')
median_vearth_ra, median_vearth_dec = get_earth_velocity(unique_mjd,
                                                         pars['RAJ'],
                                                         pars['DECJ'])
print('Getting true anomaly')
median_true_anomaly = get_true_anomaly(unique_mjd, pars)

median_vearth_ra = median_vearth_ra.squeeze()
median_vearth_dec = median_vearth_dec.squeeze()

median_om = pars['OM'] + pars['OMDOT']*(unique_mjd - pars['T0'])/365.2425
# compute orbital phase
median_phase = median_true_anomaly*180/np.pi + median_om
median_phase = median_phase % 360

# PHASE and observation day #
name_num = []
for i in range(0, len(name)):
    for ii in range(0, len(np.unique(name))):
        if name[i] == np.unique(name)[ii]:
            name_num.append(ii)
name_num = np.asarray(name_num)
# Calculate Viss

Aiss = 2.78*10**4  # thin screen,table 2 of Cordes & Rickett (1998)
D = 0.735  # kpc
pars = read_par(str(par_dir) + str(psrname) + '.par')
params = pars_to_params(pars)
params.add('d', value=D, vary=False)
params.add('derr', value=0.060, vary=False)
params.add('s', value=0.7, vary=False)
params.add('serr', value=0.03, vary=False)
params.add('KOM', value=65, vary=False)
# params.add('KIN', value=89.35, vary=False)
viss, visserr = scint_velocity(params, dnu, tau, freq, dnuerr,
                               tauerr, a=Aiss)
median_viss, median_visserr = scint_velocity(params, median_dnu, median_tau,
                                             median_freq, median_dnuerr,
                                             median_tauerr, a=Aiss)
median_viss = median_viss.flatten()
median_visserr = median_visserr.flatten()
median_viss = np.asarray(median_viss)
median_visserr = np.asarray(median_visserr)

veff_ra, veff_dec, vp_ra, vp_dec \
    = effective_velocity_annual(params, true_anomaly, vearth_ra, vearth_dec,
                                mjd)
veff = np.sqrt(veff_ra**2 + veff_dec**2)

phase_sort = np.argsort(phase)
median_phase_sort = np.argsort(median_phase)

plt.scatter(phase[phase_sort], viss[phase_sort])
plt.plot(phase[phase_sort], veff[phase_sort], 'C3')
plt.xlabel("orbital phase (degrees)")
plt.ylabel("Veff and Viss (km s-1)")
plt.show()
plt.close()

plt.scatter(median_phase[median_phase_sort], median_viss[median_phase_sort])
plt.plot(phase[phase_sort], veff[phase_sort], 'C3')
plt.xlabel("orbital phase (degrees)")
plt.ylabel("Veff and Viss (km s-1)")
plt.show()
plt.close()

par_dir = '/Users/jacobaskew/Desktop/Swinburne_Daniel/ParFiles/'
psrname = 'J0737-3039A'
pars = read_par(str(par_dir) + str(psrname) + '.par')
params = pars_to_params(pars)

# This is a test script for modelling our data

label = 'test'
wd0 = "/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/New/"
outdir = wd0 + "Modelling/"

# if not Anisotropy_Option:
priors = dict(D=bilby.core.prior.Uniform(0, 2, 'D'),
              s=bilby.core.prior.Uniform(0, 1, 's'),
              vism_ra=bilby.core.prior.Uniform(-200, 200, 'vism_ra'),
              vism_dec=bilby.core.prior.Uniform(-200, 200, 'vism_dec'),
              KIN=bilby.core.prior.Uniform(0, 180, 'KIN',
                                           boundary='periodic'),
              KOM=bilby.core.prior.Uniform(0, 360, 'KOM',
                                           boundary='periodic'))

# if Anisotropy_Option:
#     priors = dict(s=bilby.core.prior.Uniform(0, 1, 's'),
#                   psi=bilby.core.prior.Uniform(0, 180, 'psi',
#                                                boundary='periodic'),
#                   vism_psi=bilby.core.prior.Uniform(-200, 200,
# 'vism_psi'),
#                   # KIN=bilby.core.prior.Uniform(0, 180, 'KIN',
#                   #                              boundary=
# 'periodic'),
#                   KOM=bilby.core.prior.Uniform(0, 360, 'KOM',
#                                                boundary='periodic'),
#                   # KOM=bilby.prior.Gaussian(mu=KOM_mu,
# sigma=KOM_sigma,
#                   #                          name='KOM'),
#                   # OM=bilby.core.prior.Uniform(0, 360, 'OM',
#                   #                             boundary='periodic'),
#                   # T0=bilby.core.prior.Uniform(minT0, maxT0, 'T0'),
#                   efac=bilby.core.prior.Uniform(-2, 2, 'efac'),
#                   equad=bilby.core.prior.Uniform(-2, 2, 'equad'))
# if not Anisotropy_Option:
likelihood = bilby.likelihood.GaussianLikelihood(
    x=unique_mjd, y=median_viss, func=effective_velocity_annual_bilby,
    sigma=median_visserr)
# if Anisotropy_Option:
#     likelihood = \
# bilby.likelihood.GaussianLikelihood(
#     x=mjd, func=effective_velocity_annual_anisotropy_bilby,
# sigma=sigma)

# And run sampler
result = bilby.core.sampler.run_sampler(
        likelihood, priors=priors, sampler='dynesty', label=label,
        nlive=nlive, verbose=True, resume=resume,
        outdir=outdir)

font = {'size': 16}
matplotlib.rc('font', **font)
result.plot_corner()
