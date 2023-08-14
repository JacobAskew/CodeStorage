#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 12:04:23 2022

@author: jacobaskew
"""

# I want to write a python script that can process my new data much like old

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
from matplotlib.axes import Axes
import os

psrname = 'J0737-3039A'
pulsar = '0737-3039A'

wd = '/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/New/'
###############################################################################


def arc_curvature_alternate(params, U, vearth_ra, vearth_dec,
                            anisotropy=False):

    params_ = dict(params)
    if anisotropy:
        psi = params_['psi'] * np.pi / 180
        vism_psi = params_['vism_psi']
    else:
        vism_ra = params_['vism_ra']
        vism_dec = params_['vism_dec']
    s = params_['s']

    if params_['sense'].value > 0.5:
        params_['KIN'].value = 180 - params_['KIN'].value

    kmpkpc = 3.085677581e16

    # Other parameters in lower-case
    dkm = params_['d'].value * kmpkpc  # kms

    veff_ra, veff_dec, vp_ra, vp_dec = \
        effective_velocity_annual(params_, U, vearth_ra, vearth_dec)
    if anisotropy:
        veff2 = (veff_ra*np.sin(psi) + veff_dec*np.cos(psi) - vism_psi)**2
    else:
        veff2 = (veff_ra - vism_ra)**2 + (veff_dec - vism_dec)**2

    # Calculate curvature model
    model = dkm * s * (1 - s)/(2 * veff2)  # in 1/(km * Hz**2)
    # Convert to 1/(m * mHz**2) for beta in 1/m and fdop in mHz
    model = model/1e9
    # Covert from eta back into normalised fdop, using a reference eta=100
    # model = np.sqrt((100)/(model))

    return model


###############################################################################


def remove_eclipse(start_mjd, tobs, dyn, fluxes):
    end_mjd = start_mjd + tobs/86400
    Eclipse_mjd = np.genfromtxt(eclipsefile, delimiter=',',
                                encoding=None, dtype=float)
    Eclipse_events = np.array(np.where((Eclipse_mjd > start_mjd) *
                              (Eclipse_mjd < end_mjd)))
    if Eclipse_events.size == 0:
        median_flux_list = []
        for i in range(0, np.shape(fluxes)[1]):
            median_flux_list.append(np.median(fluxes[:, i]))
        median_fluxes = np.asarray(median_flux_list)
        Eclipse_index = int(np.argmin(median_fluxes))
        if Eclipse_index is not None:
            fluxes[:, Eclipse_index-3:Eclipse_index+3] = 0
            if linear:
                dyn.refill(method='linear')
            else:
                dyn.refill()
            print("Eclispe in dynspec")

        else:
            print("No Eclispe in dynspec")
    else:
        Eclipse_events_mjd = Eclipse_mjd[Eclipse_events]
        mjds = start_mjd + dyn.times/86400
        Eclipse_index = np.argmin(abs(mjds - Eclipse_events_mjd))
        if Eclipse_index is not None:
            print("Eclispe in dynspec")
            fluxes[:, Eclipse_index-3:Eclipse_index+3] = 0
            if linear:
                dyn.refill(method='linear')
            elif median:
                dyn.refill(method='median')
            else:
                dyn.refill()
        else:
            print("No Eclispe in dynspec")
    return Eclipse_index


##############################################################################
datadir = wd + 'Dynspec/'
outdir = wd + 'DataFiles/'
spectradir = wd + 'SpectraPlots/'
dynspecs = sorted(glob.glob(datadir + '/*.dynspec'))
outfile = str(outdir) + str(psrname) + '_ScintillationResults_UHF.txt'
par_dir = '/Users/jacobaskew/Desktop/Swinburne_Daniel/ParFiles/'

Font = 30
Size = 80*np.pi  # Determines the size of the datapoints used
font = {'size': 28}
matplotlib.rc('font', **font)


time_bin = 10
freq_bin = 10
zap = True
linear = True

dyn1 = Dynspec(filename=dynspecs[0], process=False)
dyn2 = Dynspec(filename=dynspecs[1], process=False)
dyn3 = Dynspec(filename=dynspecs[2], process=False)
dyn = dyn1 + dyn2 + dyn3
# dyn.trim_edges()
# dyn_crop = cp(dyn)
# dyn_crop.crop_dyn(tmin=20, tmax=50)
# dyn_crop.zap()
# dyn_crop.refill(linear=True)

dyn_crop1 = cp(dyn)
dyn_crop1.crop_dyn(tmin=0, tmax=10)
dyn_crop1.zap()
dyn_crop1.refill(linear=True)

dyn_crop2 = cp(dyn)
dyn_crop2.crop_dyn(tmin=10, tmax=20)
dyn_crop2.zap()
dyn_crop2.refill(linear=True)

dyn_crop3 = cp(dyn)
dyn_crop3.crop_dyn(tmin=20, tmax=30)
dyn_crop3.zap()
dyn_crop3.refill(linear=True)

dyn_crop4 = cp(dyn)
dyn_crop4.crop_dyn(tmin=30, tmax=40)
dyn_crop4.zap()
dyn_crop4.refill(linear=True)

dyn_crop5 = cp(dyn)
dyn_crop5.crop_dyn(tmin=40, tmax=50)
dyn_crop5.zap()
dyn_crop5.refill(linear=True)

dyn_crop6 = cp(dyn)
dyn_crop6.crop_dyn(tmin=50, tmax=60)
dyn_crop6.zap()
dyn_crop6.refill(linear=True)

dyn_crop7 = cp(dyn)
dyn_crop7.crop_dyn(tmin=60, tmax=70)
dyn_crop7.zap()
dyn_crop7.refill(linear=True)

dyn_crop8 = cp(dyn)
dyn_crop8.crop_dyn(tmin=70, tmax=80)
dyn_crop8.zap()
dyn_crop8.refill(linear=True)

dyn_crop9 = cp(dyn)
dyn_crop9.crop_dyn(tmin=80, tmax=90)
dyn_crop9.zap()
dyn_crop9.refill(linear=True)

dyn_crop10 = cp(dyn)
dyn_crop10.crop_dyn(tmin=90, tmax=100)
dyn_crop10.zap()
dyn_crop10.refill(linear=True)

dyn_crop11 = cp(dyn)
dyn_crop11.crop_dyn(tmin=100, tmax=110)
dyn_crop11.zap()
dyn_crop11.refill(linear=True)

dyn_crop12 = cp(dyn)
dyn_crop12.crop_dyn(tmin=110, tmax=120)
dyn_crop12.zap()
dyn_crop12.refill(linear=True)

dyn_crop13 = cp(dyn)
dyn_crop13.crop_dyn(tmin=120, tmax=130)
dyn_crop13.zap()
dyn_crop13.refill(linear=True)

dyn_crop14 = cp(dyn)
dyn_crop14.crop_dyn(tmin=130, tmax=140)
dyn_crop14.zap()
dyn_crop14.refill(linear=True)

dyn_crop15 = cp(dyn)
dyn_crop15.crop_dyn(tmin=140, tmax=150)
dyn_crop15.zap()
dyn_crop15.refill(linear=True)

dyn_crop16 = cp(dyn)
dyn_crop16.crop_dyn(tmin=150, tmax=160)
dyn_crop16.zap()
dyn_crop16.refill(linear=True)

dyn_crop17 = cp(dyn)
dyn_crop17.crop_dyn(tmin=160, tmax=170)
dyn_crop17.zap()
dyn_crop17.refill(linear=True)

dyn_crop18 = cp(dyn)
dyn_crop18.crop_dyn(tmin=170, tmax=180)
dyn_crop18.zap()
dyn_crop18.refill(linear=True)

dyn_crop19 = cp(dyn)
dyn_crop19.crop_dyn(tmin=180, tmax=190)
dyn_crop19.zap()
dyn_crop19.refill(linear=True)
###############################################################################

dyn = dyn_crop1 + dyn_crop2 + dyn_crop3 + dyn_crop4 + dyn_crop5 + dyn_crop6 + \
    dyn_crop7 + dyn_crop8 + dyn_crop9 + dyn_crop10 + dyn_crop11 + \
    dyn_crop12 + dyn_crop13 + dyn_crop14 + dyn_crop15 + dyn_crop16 + \
    dyn_crop17 + dyn_crop18 + dyn_crop19
# dyn.write_file(filename=wd+"Observation_MeerKAT.processed")
Fmax = np.max(dyn.freqs) - 48
Fmin = np.min(dyn.freqs) + 48
dyn.trim_edges()
dyn.crop_dyn(fmin=Fmin, fmax=Fmax)  # tmin=50
dyn.plot_dyn(filename=str(spectradir)+'FullSpectra.pdf', dpi=300)

dyn.plot_sspec(filename=str(spectradir)+'Rawsspec.pdf', lamsteps=True,
               maxfdop=15, dpi=300)  # plot secondary spectrum

# Rescale the dynamic spectrum in time, using a velocity model
dyn.scale_dyn(scale='lambda+velocity', s=0.7, d=0.9, inc=88.7, Omega=65,
              parfile=par_dir+'J0737-3039A.par')
# Plot the velocity-rescaled dynamic spectrum
dyn.plot_dyn(filename=str(spectradir)+'FullSpectra_velocity.pdf',
             velocity=True, dpi=300)
# Fit the arc using the new spectrum, and plot
dyn.plot_sspec(filename=str(spectradir)+'Rawsspec_velocity.pdf', lamsteps=True,
               velocity=True, maxfdop=15, prewhite=True,
               dpi=300)
dyn.fit_arc(filename=str(spectradir)+'DopplerProfile_velocity.pdf',
            velocity=True, cutmid=5, plot=True, weighted=False,
            lamsteps=True, subtract_artefacts=True, log_parabola=True,
            constraint=[100, 1000], high_power_diff=-0.05, low_power_diff=-4,
            etamin=1, startbin=5)
dyn.plot_sspec(filename=str(spectradir)+'Rawsspec_velocity_fit.pdf',
               lamsteps=True, velocity=True, maxfdop=15, plotarc=True,
               prewhite=True, dpi=300)
###############################################################################
# Here I want to loop over the above code and instead of plotting save the data
# Defining some basic parameters of our arc fitting
Cutmid = 5
Startbin = 5
# Maxnormfac = 10
Etamin = 1
#
dyn = dyn_crop1 + dyn_crop2 + dyn_crop3 + dyn_crop4 + dyn_crop5 + dyn_crop6 + \
    dyn_crop7 + dyn_crop8 + dyn_crop9 + dyn_crop10 + dyn_crop11 + \
    dyn_crop12 + dyn_crop13 + dyn_crop14 + dyn_crop15 + dyn_crop16 + \
    dyn_crop17 + dyn_crop18 + dyn_crop19
Fmax = np.max(dyn.freqs) - 48
Fmin = np.min(dyn.freqs) + 48
dyn.trim_edges()
dyn.crop_dyn(fmin=Fmin, fmax=Fmax, tmin=50)

# A potential for loop to run over several parameters.
Snum = 4
Omeganum = 4
Incnum = 2
Vism_ranum = 4
Vism_decnum = 4
S = np.linspace(0, 1, Snum)
Omega = np.linspace(0, 180, Omeganum)
Inc = np.asarray([89.35, 90.65])
Vism_ra = np.linspace(-50, 100, Vism_ranum)
Vism_dec = np.linspace(-50, 100, Vism_decnum)

S_list = []
Omega_list = []
Inc_list = []
Vism_ra_list = []
Vism_dec_list = []
MaxPower_list = []
Eta_list = []
Power_Average = []
fdop_Range = []
arc_curvature_list = []
Sigma_list = []

# i = False
# ii = False
# iii = False
# iv = False
# v = False

for i in range(0, Snum):
    for ii in range(0, Omeganum):
        for iii in range(0, Incnum):
            for iv in range(0, Vism_ranum):
                for v in range(0, Vism_decnum):
                    dyn_crop = cp(dyn)
                # Rescale the dynamic spectrum in time, using a velocity model
                    dyn_crop.scale_dyn(scale='lambda+velocity',
                                       d=0.9,
                                       s=S[i],
                                       # s=0.7,
                                       #
                                       Omega=Omega[ii],
                                       # Omega=65,
                                       #
                                       inc=Inc[iii],
                                       # inc=Inc[0],
                                       #
                                       vism_ra=Vism_ra[iv],
                                       # vism_ra=0,
                                       #
                                       vism_dec=Vism_dec[v],
                                       # vism_dec=0,
                                       #
                                       parfile=par_dir+'J0737-3039A.par')
                    dyn_crop.calc_sspec(lamsteps=True, velocity=True,
                                        prewhite=True)
                    dyn_crop.fit_arc(velocity=True, cutmid=Cutmid, plot=False,
                                     weighted=False, lamsteps=True,
                                     subtract_artefacts=True,
                                     log_parabola=True,
                                     constraint=[100, 1000],
                                     high_power_diff=-0.05,
                                     low_power_diff=-4, etamin=Etamin,
                                     startbin=Startbin, display=False)

                    pars = read_par(str(par_dir) + str(psrname) + '.par')
                    params = pars_to_params(pars)
                    params.add('d', value=0.9, vary=False)
                    params.add('sense', value=0.2, vary=False)
                    #
                    params.add('s', value=S[i], vary=False)
                    # params.add('s', value=0.7, vary=False)
                    #
                    params.add('KOM', value=Omega[ii], vary=False)
                    # params.add('KOM', value=65, vary=False)
                    #
                    params.add('KIN', value=Inc[iii], vary=False)
                    # params.add('KIN', value=Inc[0], vary=False)
                    #
                    # params.add('vism_ra', value=0, vary=False)
                    params.add('vism_ra', value=Vism_ra[iv], vary=False)
                    #
                    # params.add('vism_dec', value=0, vary=False)
                    params.add('vism_dec', value=Vism_dec[v], vary=False)
                    #
                    mjd = [dyn_crop.mjd, dyn_crop.mjd+(dyn_crop.tobs)/86400]
                    ssb_delays = get_ssb_delay(mjd, pars['RAJ'], pars['DECJ'])
                    mjd -= np.divide(ssb_delays, 86400)
                    vearth_ra, vearth_dec = get_earth_velocity(mjd,
                                                               pars['RAJ'],
                                                               pars['DECJ'])
                    U = get_true_anomaly(mjd, params)
                # Finding the maxima in the Doppler profile associated with the fit
                    ind_pos = np.argwhere(dyn_crop.normsspec_fdop >= 0)
                    ind_neg = np.argwhere(dyn_crop.normsspec_fdop < 0)
                    ydata_pos = dyn_crop.normsspecavg[ind_pos]
                    ydata_neg = np.flip(dyn_crop.normsspecavg[ind_neg])
                    power_pos_average = (ydata_neg + ydata_pos)/2
                    xdata_pos = dyn_crop.normsspec_fdop[ind_pos]
                    fdop_estimation = \
                        xdata_pos[np.argmax(power_pos_average)]
                    eta_estimation = Etamin/(fdop_estimation**2)
                    power_estimation = \
                        power_pos_average[np.argmax(power_pos_average)]
                    if i:
                        S_list.append(S[i])
                    else:
                        S_list.append(0.7)
                    if ii:
                        Omega_list.append(Omega[ii])
                    else:
                        Omega_list.append(65)
                    if iii:
                        Inc_list.append(Inc[iii])
                    else:
                        Inc_list.append(89.35)
                    if iv:
                        Vism_ra_list.append(Vism_ra[iv])
                    else:
                        Vism_ra_list.append(0)
                    if v:
                        Vism_dec_list.append(Vism_dec[v])
                    else:
                        Vism_dec_list.append(0)
                    MaxPower_list.append(power_estimation)
                    Eta_list.append(eta_estimation)
                    Power_Average.append(power_pos_average)
                    fdop_Range.append(xdata_pos)
                    arc_curvature_list.append(
                        arc_curvature_alternate(params, U, vearth_ra,
                                                vearth_dec))
                    Sigma_list.append(dyn_crop.noise)
S_list = np.asarray(S_list)
Omega_list = np.asarray(Omega_list)
Inc_list = np.asarray(Inc_list)
Vism_ra_list = np.asarray(Vism_ra_list)
Vism_dec_list = np.asarray(Vism_dec_list)
MaxPower_list = np.asarray(MaxPower_list)
Eta_list = np.asarray(Eta_list)
arc_curvature_array = np.asarray(arc_curvature_list)[:, 0]
Sigma_list = np.asarray(Sigma_list)
Power_Average = np.asarray(Power_Average)

eta_Range = []
power_at_expected_eta = []
for i in range(0, len(fdop_Range)):
    eta_Range.append(1/((fdop_Range[i])**2))
    power_at_expected_eta.append(np.max(Power_Average[i][(eta_Range[i] >
                                                          800)*(eta_Range[i] <
                                                                1200)]))
xdata = np.arange(0, len(power_at_expected_eta))
xdata2 = np.arange(0, Power_Average.shape[1])
power_at_expected_eta = np.asarray(power_at_expected_eta)
eta_Range = np.asarray(eta_Range)

power_at_predicted_eta = []
power_around_predicted_eta = []
for i in range(0, len(arc_curvature_array)):
    if arc_curvature_array[i] <= 0:
        power_around_predicted_eta.append(np.max(Power_Average[i, np.argwhere((eta_Range[i, :] > 0) * (eta_Range[i, :] < 10))]))
        power_at_predicted_eta.append(np.max(Power_Average[i, np.argmin(abs(eta_Range[i, :] - arc_curvature_array[i]))]))
    else:
        power_around_predicted_eta.append(np.max(Power_Average[i, np.argwhere((eta_Range[i, :] > arc_curvature_array[i]*0.9) * (eta_Range[i, :] < arc_curvature_array[i]*1.1))]))
        power_at_predicted_eta.append(np.max(Power_Average[i, np.argmin(abs(eta_Range[i, :] - arc_curvature_array[i]))]))
power_at_predicted_eta = np.asarray(power_at_predicted_eta)
power_around_predicted_eta = np.asarray(power_around_predicted_eta)

# xdata3 = np.arange(0, len(power_at_predicted_eta))
# Inc_list3 = Inc_list[np.argsort(power_at_predicted_eta)]
# Omega_list3 = Omega_list[np.argsort(power_at_predicted_eta)]
# S_list3 = S_list[np.argsort(power_at_predicted_eta)]

fig = plt.figure(figsize=(60, 30))
fig.subplots_adjust(hspace=0.25, wspace=0.25)
ax1 = fig.add_subplot(2, 3, 1)
# ax1.set_ylim(10, 25)
ax1.scatter(Inc_list, power_at_predicted_eta)
ax1.set_ylabel('Predicted Power (dB)', fontsize=Font,  ha='center')
ax1.set_xlabel(r'$i$ (degrees)', fontsize=Font, ha='center')
#
ax2 = fig.add_subplot(2, 3, 2)
# ax2.set_ylim(10, 25)
ax2.scatter(Omega_list, power_at_predicted_eta)
ax2.set_ylabel('Predicted Power (dB)', fontsize=Font,  ha='center')
ax2.set_xlabel(r'$\Omega$ (degrees)', fontsize=Font, ha='center')
#
ax3 = fig.add_subplot(2, 3, 3)
# ax3.set_xlim(0.075, 0.925)
# ax3.set_ylim(10, 25)
ax3.scatter(S_list, power_at_predicted_eta)
ax3.set_ylabel('Predicted Power (dB)', fontsize=Font,  ha='center')
ax3.set_xlabel(r'$s$ (arb)', fontsize=Font, ha='center')
#
ax4 = fig.add_subplot(2, 3, 4)
# ax4.set_xlim(0.075, 0.925)
# ax4.set_ylim(10, 25)
ax4.scatter(Vism_ra_list, power_at_predicted_eta)
ax4.set_ylabel('Predicted Power (dB)', fontsize=Font,  ha='center')
ax4.set_xlabel(r'$V_{ISM},\alpha$ ($km$\,$s^{-1}$)', fontsize=Font, ha='center')
#
ax5 = fig.add_subplot(2, 3, 5)
# ax5.set_xlim(0.075, 0.925)
# ax5.set_ylim(10, 25)
ax5.scatter(Vism_dec_list, power_at_predicted_eta)
ax5.set_ylabel('Predicted Power (dB)', fontsize=Font,  ha='center')
ax5.set_xlabel(r'$V_{ISM},\delta$ ($km$\,$s^{-1}$)', fontsize=Font, ha='center')
#
ax6 = fig.add_subplot(2, 3, 6)
# ax6.set_xlim(0.075, 0.925)
# ax6.set_ylim(10, 25)
cm = plt.cm.get_cmap('viridis')
z = Omega_list[Inc_list > 90]
sc = ax6.scatter(S_list[Inc_list > 90], power_at_predicted_eta[Inc_list > 90],
                 marker='v', c=z, cmap=cm, s=Size, alpha=0.5,
                 label=r'$i>90^\circ$')
z = Omega_list[Inc_list < 90]
sc = ax6.scatter(S_list[Inc_list < 90], power_at_predicted_eta[Inc_list < 90],
                 marker='s', c=z, cmap=cm, s=Size, alpha=0.5,
                 label=r'$i<90^\circ$')
ax6.set_ylabel('Predicted Power (dB)', fontsize=Font,  ha='center')
ax6.set_xlabel(r'$s$ (arb)', fontsize=Font, ha='center')
cbar = plt.colorbar(sc)
cbar.ax.set_ylabel(r'$\Omega$ (degrees)', rotation=90, labelpad=40, y=0.5)
ax6.legend()
#
plt.savefig("/Users/jacobaskew/Desktop/parameters_curvature.pdf")
plt.show()
plt.close()

# TESTING SOMETHING CRAZY, I want these to be on the same axes/plot
fig = plt.figure(figsize=(30, 15))
fig.subplots_adjust(hspace=0.5, wspace=0.5)
#
ax1 = fig.add_subplot(1, 2, 1)
#
cm = plt.cm.get_cmap('viridis')
z = power_at_predicted_eta
sc = ax1.scatter(S_list, Omega_list, c=z, cmap=cm, s=Size, alpha=0.5)  # Put your speed/power plot here
cm = plt.cm.get_cmap('viridis')
z = power_at_predicted_eta
sc = ax1.scatter(S_list, Omega_list, c=z, cmap=cm, s=Size, alpha=0.5)  # Put your speed/power plot here
ax1.set_ylabel(r'$\Omega$ (degrees)', fontsize=Font,  ha='center')
ax1.set_xlabel(r'$s$ (arb)', fontsize=Font, ha='center')
#
cbar = plt.colorbar(sc)
cbar.ax.set_ylabel('Predicted Power (dB)', rotation=90, labelpad=40, y=0.5)
#
ax2 = fig.add_subplot(1, 2, 2)
#
cm = plt.cm.get_cmap('viridis')
z = power_at_predicted_eta
sc = ax2.scatter(Vism_ra_list, Vism_dec_list, c=z, cmap=cm, s=Size, alpha=0.5)  # Put your speed/power plot here
cm = plt.cm.get_cmap('viridis')
z = power_at_predicted_eta
sc = ax2.scatter(Vism_ra_list, Vism_dec_list, c=z, cmap=cm, s=Size, alpha=0.5)  # Put your speed/power plot here
ax2.set_ylabel(r'$V_{ISM},\delta$ ($km$\,$s^{-1}$)', fontsize=Font,  ha='center')
ax2.set_xlabel(r'$V_{ISM},\alpha$ ($km$\,$s^{-1}$)', fontsize=Font, ha='center')
#
cbar = plt.colorbar(sc)
cbar.ax.set_ylabel('Predicted Power (dB)', rotation=90, labelpad=40, y=0.5)
#
plt.show()


# This is a different plot ##
# z = Omega_list[Inc_list > 90]
# sc = plt.scatter(S_list[Inc_list > 90], power_at_expected_eta[Inc_list > 90],
#                   marker='v', c=z, cmap=cm, s=Size, alpha=0.5,
#                   label=r'$i>90^\circ$')
# z = Omega_list[Inc_list < 90]
# sc = plt.scatter(S_list[Inc_list < 90], power_at_expected_eta[Inc_list < 90],
#                   marker='s', c=z, cmap=cm, s=Size, alpha=0.5,
#                   label=r'$i<90^\circ$')
# # This is a different plot ##
# # z = Omega_list[Inc_list > 90]
# # sc = plt.scatter(xdata[Inc_list > 90], power_at_expected_eta[Inc_list > 90],
# #                  marker='v', c=z, cmap=cm, s=Size, alpha=0.5,
# #                  label=r'$i>90^\circ$')
# # z = Omega_list[Inc_list < 90]
# # sc = plt.scatter(xdata[Inc_list < 90], power_at_expected_eta[Inc_list < 90],
# #                  marker='s', c=z, cmap=cm, s=Size, alpha=0.5,
# #                  label=r'$i<90^\circ$')
# cbar = plt.colorbar(sc)
# cbar.ax.set_ylabel(r'$\Omega$ (degrees)', rotation=90, labelpad=40, y=0.5)
# plt.xlabel(r'Relative distance to the scattering screen $s$ (arb)')
# plt.ylabel('Power Predicted (dB)')
# sc = plt.scatter(Eta_list[Inc_list < 90], MaxPower_list[Inc_list < 90],
#                  c=z, cmap=cm, s=Size, alpha=0.5)
# plt.colorbar(sc)
# plt.scatter()
# plt.scatter(Eta_list[Inc_list < 90], MaxPower_list[Inc_list < 90], c='C0')

###############################################################################
# # This method uses the power in the doppler profile to fit the arc #

# # Calculating a velocity and then plotting and saving the secondary spectrum
# # for a range of values of i, d, s and omega.
# dyn = dyn_crop1 + dyn_crop2 + dyn_crop3 + dyn_crop4 + dyn_crop5 + dyn_crop6 + \
#     dyn_crop7 + dyn_crop8 + dyn_crop9 + dyn_crop10 + dyn_crop11 + \
#     dyn_crop12 + dyn_crop13 + dyn_crop14 + dyn_crop15 + dyn_crop16 + \
#     dyn_crop17 + dyn_crop18 + dyn_crop19
# # dyn.write_file(filename=wd+"Observation_MeerKAT.processed")
# Fmax = np.max(dyn.freqs) - 48
# Fmin = np.min(dyn.freqs) + 48
# dyn.trim_edges()
# dyn.crop_dyn(fmin=Fmin, fmax=Fmax)  # tmin=50
# # Finished caclulating the starting dynspec!
# # Defining some basic parameters of our arc fitting
# Cutmid = 5
# Startbin = 5
# Maxnormfac = 10
# # Rescaling the dynamic spectra
# dyn.scale_dyn(scale='lambda+velocity', s=0.7, d=0.9, inc=88.7, Omega=65,
#               parfile=par_dir+'J0737-3039A.par')
# #
# dyn.plot_sspec(maxfdop=Maxnormfac, velocity=True)
# dyn.calc_sspec(prewhite=False, lamsteps=True, velocity=True)
# dyn.norm_sspec(eta=100, maxnormfac=Maxnormfac, plot=False,
#                startbin=Startbin, cutmid=Cutmid, plot_fit=False,
#                display=False, numsteps=10000, lamsteps=True, velocity=True)

# ind_pos = np.argwhere(dyn.normsspec_fdop >= 0)
# ind_neg = np.argwhere(dyn.normsspec_fdop < 0)
# xdata_pos = dyn.normsspec_fdop[ind_pos]
# ydata_pos = dyn.normsspecavg[ind_pos]
# ydata_neg = np.flip(dyn.normsspecavg[ind_neg])
# # Creating the Power v fdop average
# power_pos_average = (ydata_neg + ydata_pos)/2
# max_norm_power = np.max(power_pos_average)

# dyn.estimate_noise(cutmid=Cutmid, lamsteps=False, startbin=5)
# # Inputs to the pdf
# Sigma = dyn.noise
# Power = power_pos_average
# MaxPower = max_norm_power
# # Creating a pdf of power
# pdf_log = (np.log(1/(Sigma * np.sqrt(2 * np.pi))) - 0.5 *
#            ((Power-MaxPower)/(Sigma))**2)
# pdf_log_norm_component = \
#     np.log(np.sum(np.exp((np.log(1/(Sigma * np.sqrt(2 * np.pi))) -
#                           0.5 * ((Power-MaxPower)/(Sigma))**2))))
# pdf_log_norm = pdf_log - pdf_log_norm_component
# pdf_norm = np.exp(pdf_log_norm)

# maxima = float(xdata_pos[np.argmax(pdf_norm), 0])
# maxima_eta = float(100/(maxima**2))

# print("The estimated norm_fdop:", maxima)
# print("The estimated curvauture from norm_fdop:", maxima_eta)
# peak_index = np.argmax(pdf_norm[:, 0])
# itr = 100
# probability_values = np.linspace(0, np.max(pdf_norm[:, 0]), itr)
# Sigma = []
# Ceiling_values = []
# for ii in range(itr-1, 0, -1):
#     pdfvalues = []
#     ceiling = probability_values[ii]
#     for i in range(0, pdf_norm.shape[0]):
#         if pdf_norm[:, 0][i] > ceiling:
#             pdfvalues.append(pdf_norm[:, 0][i])
#     if np.sum(pdfvalues) > 0.7:
#         break
#     # print("This is the sum:", np.sum(pdfvalues))
#     Sigma.append(np.sum(pdfvalues))
#     Ceiling_values.append(ceiling)
# Sigma = np.asarray(Sigma)
# Ceiling = Ceiling_values[np.argmin(abs(Sigma - 0.6827))]
# uncertainty_right = xdata_pos[np.argmin(abs(pdf_norm[peak_index:, 0] -
#                                         Ceiling))] + maxima
# # uncertainty_left = xdata_pos[np.argwhere(abs(pdf_norm[:peak_index,
# # 0] - Ceiling) < 0.001)[0][0]]
# uncertainty_left = xdata_pos[np.argmin(abs(pdf_norm[:peak_index, 0] -
#                                        Ceiling))]

# uncertainty_left_eta = float(100/(uncertainty_left**2))
# uncertainty_left_eta_diff = uncertainty_left_eta - maxima_eta
# uncertainty_right_eta = float(100/(uncertainty_right**2))
# uncertainty_right_eta_diff = maxima_eta - uncertainty_right_eta
# uncertainty_mean_eta = (uncertainty_left_eta_diff +
#                         uncertainty_right_eta_diff)/2
# print("The estimated curvauture error from norm_fdop:",
#       uncertainty_mean_eta)

# cutmid_normsspec_fdop = dyn.normsspec_fdop
# cutmid_normsspecavg = dyn.normsspecavg
# dyn.norm_sspec(eta=100, maxnormfac=Maxnormfac, plot=False,
#                startbin=Startbin, cutmid=0, plot_fit=False,
#                display=False, numsteps=10000, velocity=True)

# # dyn.fit_arc(filename=str(spectradir)+'DopplerProfile_velocity.pdf',
# #             velocity=True, cutmid=Cutmid, startbin=Startbin, plot=True,
# #             weighted=False, lamsteps=True, subtract_artefacts=True,
# #             log_parabola=True, constraint=[100, 1000], high_power_diff=-0.05,
# #             low_power_diff=-4, etamin=1)

# # dyn.get_arc_snr(startbin=5, maxfdop=15,
# #                 prewhite=False, betaeta=maxima_eta,
# #                 betaetaerr=uncertainty_mean_eta)

# # Plotting the raw normalised fdop space
# plt.figure(figsize=(20, 15))
# plt.plot(dyn.normsspec_fdop, dyn.normsspecavg, c='C0')
# plt.plot(cutmid_normsspec_fdop, cutmid_normsspecavg, c='C1')
# yl = plt.ylim()
# plt.plot([-maxima, -maxima], yl, 'C3', alpha=0.4)
# plt.plot([maxima, maxima], yl, 'C3', alpha=0.4)
# plt.plot([-uncertainty_left, -uncertainty_left], yl, '--C3', alpha=0.4)
# plt.plot([-uncertainty_right, -uncertainty_right], yl, '--C3',
#          alpha=0.4)
# plt.plot([uncertainty_left, uncertainty_left], yl, '--C3', alpha=0.4)
# plt.plot([uncertainty_right, uncertainty_right], yl, '--C3', alpha=0.4)
# # plt.ylim(-10, 100)
# # plt.xlim(-0.25, 0.25)
# plt.ylim(yl)
# plt.xlim(np.min(dyn.normsspec_fdop), np.max(dyn.normsspec_fdop))
# plt.ylabel("Power (dB)")
# plt.xlabel(r'Normalised $f_t$')
# # plt.savefig(str(wd) + 'fdopspace.png')
# plt.show()
# plt.close()

# # A potential for loop to run over several parameters.
# S = np.linspace(0, 1, 11)
# Omega = np.arange(0, 360, 10)
# Inc = np.asarray([89.35, 90.65])
# vism_ra
# vism_dec

# for i in range(0, 10):
#     for ii in range(0, 36):
#         for iii in range(0, 2):
# #             dyn_crop = cp(dyn)
# #             dyn_crop.scale_dyn(scale='lambda+velocity', s=S[i], d=0.735,
# #                                inc=Inc[iii], Omega=Omega[ii],
# #                                parfile=par_dir+'J0737-3039A.par')
# #             dyn_crop.plot_sspec(filename='/Users/jacobaskew/Desktop/sspecs/i' +
# #                                 str(int(Inc[iii]))+'/S'+str(S[i]) +
# #                                 '_Omega'+str(Omega[ii]) +
# #                                 '_Rawsspec_velocity.pdf', lamsteps=True,
# #                                 velocity=True, maxfdop=15)

# ###############################################################################
psrname = 'J0737-3039A'
pulsar = '0737-3039A'

wd0 = '/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/'
eclipsefile = wd0+'Datafiles/Eclipse_mjd.txt'
wd = wd0+'New/'
datadir = wd + 'Dynspec/'
outdir = wd + 'DataFiles/'
par_dir = '/Users/jacobaskew/Desktop/Swinburne_Daniel/ParFiles/'
dynspecs = sorted(glob.glob(datadir + '/*.dynspec'))
observations = []
for i in range(0, len(dynspecs)):
    observations.append(dynspecs[i].split(datadir)[1].split('-')[0]+'-' +
                        dynspecs[i].split(datadir)[1].split('-')[1]+'-' +
                        dynspecs[i].split(datadir)[1].split('-')[2])
observations = np.unique(np.asarray(observations))
for i in range(0, len(observations)):
    observation_date = observations[i]+'/'
    observation_date2 = observations[i]
    dynspecs = sorted(glob.glob(datadir + str(observation_date.split('/')[0])
                                + '*.XPp.dynspec'))
    dynspecfile = \
        outdir+'DynspecPlotFiles/'+observation_date2+'_CompleteDynspec.dynspec'
    if os.path.exists(dynspecfile):
        continue
    # Settings #
    time_bin = 30
    freq_bin = 30
    #
    zap = True
    linear = False
    var = False
    median = True
    #
    dynspecplotfiledir = str(outdir)+"DynspecPlotFiles/"
    try:
        os.mkdir(dynspecplotfiledir)
    except OSError as error:
        print(error)
    outfile = str(dynspecplotfiledir)+str(observation_date2) + \
        '_CompleteDynspec.dynspec'

    if len(dynspecs) == 3:
        dyn1 = Dynspec(filename=dynspecs[0], process=False)
        remove_eclipse(dyn1.mjd, dyn1.tobs, dyn1, dyn1.dyn)
        dyn2 = Dynspec(filename=dynspecs[1], process=False)
        dyn3 = Dynspec(filename=dynspecs[2], process=False)
        remove_eclipse(dyn3.mjd, dyn3.tobs, dyn3, dyn3.dyn)
        dyn = dyn1 + dyn2 + dyn3
    elif len(dynspecs) == 2:
        dyn1 = Dynspec(filename=dynspecs[0], process=False)
        remove_eclipse(dyn1.mjd, dyn1.tobs, dyn1, dyn1.dyn)
        dyn2 = Dynspec(filename=dynspecs[1], process=False)
        remove_eclipse(dyn2.mjd, dyn2.tobs, dyn2, dyn2.dyn)
        dyn = dyn1 + dyn2
    else:
        dyn = Dynspec(filename=dynspecs[0], process=False)
        remove_eclipse(dyn, dyn.dyn)
    dyn.plot_dyn()
    dyn.trim_edges()
    Fmax = np.max(dyn.freqs) - 48
    Fmin = np.min(dyn.freqs) + 48
    dyn_crop0 = cp(dyn)
    dyn_crop0.crop_dyn(fmin=Fmin, fmax=Fmax)
    f_min_init = Fmax - freq_bin
    f_max = Fmax
    f_init = int((f_max + f_min_init)/2)
    bw_init = int(f_max - f_min_init)

    # # For this observation the 15min 30-45 are evil and must be cropped.
    # if observation_date2 == '2022-06-30':
    #     dyn_crop1 = cp(dyn_crop0)
    #     dyn_crop1.crop_dyn(tmin=0, tmax=30)
    #     dyn_crop2 = cp(dyn_crop0)
    #     dyn_crop2.crop_dyn(tmin=50)
    #     dyn_crop = dyn_crop1 + dyn_crop2
    #     dyn_crop.trim_edges()
    # else:
    dyn_crop = cp(dyn_crop0)

    len_min = dyn_crop.tobs/60
    len_min_chunk = len_min/20

    len_minimum = 0
    len_maximum = int(len_min_chunk)
    dyn_crop1 = cp(dyn_crop)
    dyn_crop1.crop_dyn(tmin=0, tmax=len_maximum)
    dyn_crop1.zap()
    dyn_crop1.refill()

    len_minimum += int(len_min_chunk)
    len_maximum += int(len_min_chunk)
    dyn_crop2 = cp(dyn_crop)
    dyn_crop2.crop_dyn(tmin=len_minimum, tmax=len_maximum)
    dyn_crop2.zap()
    dyn_crop2.refill(method='linear')

    len_minimum += int(len_min_chunk)
    len_maximum += int(len_min_chunk)
    dyn_crop3 = cp(dyn_crop)
    dyn_crop3.crop_dyn(tmin=len_minimum, tmax=len_maximum)
    dyn_crop3.zap()
    dyn_crop3.refill(method='linear')

    len_minimum += int(len_min_chunk)
    len_maximum += int(len_min_chunk)
    dyn_crop4 = cp(dyn_crop)
    dyn_crop4.crop_dyn(tmin=len_minimum, tmax=len_maximum)
    dyn_crop4.zap()
    dyn_crop4.refill(method='linear')

    len_minimum += int(len_min_chunk)
    len_maximum += int(len_min_chunk)
    dyn_crop5 = cp(dyn_crop)
    dyn_crop5.crop_dyn(tmin=len_minimum, tmax=len_maximum)
    dyn_crop5.zap()
    dyn_crop5.refill(method='linear')

    len_minimum += int(len_min_chunk)
    len_maximum += int(len_min_chunk)
    dyn_crop6 = cp(dyn_crop)
    dyn_crop6.crop_dyn(tmin=len_minimum, tmax=len_maximum)
    dyn_crop6.zap()
    dyn_crop6.refill(method='linear')

    len_minimum += int(len_min_chunk)
    len_maximum += int(len_min_chunk)
    dyn_crop7 = cp(dyn_crop)
    dyn_crop7.crop_dyn(tmin=len_minimum, tmax=len_maximum)
    dyn_crop7.zap()
    dyn_crop7.refill(method='linear')

    len_minimum += int(len_min_chunk)
    len_maximum += int(len_min_chunk)
    dyn_crop8 = cp(dyn_crop)
    dyn_crop8.crop_dyn(tmin=len_minimum, tmax=len_maximum)
    dyn_crop8.zap()
    dyn_crop8.refill(method='linear')

    len_minimum += int(len_min_chunk)
    len_maximum += int(len_min_chunk)
    dyn_crop9 = cp(dyn_crop)
    dyn_crop9.crop_dyn(tmin=len_minimum, tmax=len_maximum)
    dyn_crop9.zap()
    dyn_crop9.refill(method='linear')

    len_minimum += int(len_min_chunk)
    len_maximum += int(len_min_chunk)
    dyn_crop10 = cp(dyn_crop)
    dyn_crop10.crop_dyn(tmin=len_minimum, tmax=len_maximum)
    dyn_crop10.zap()
    dyn_crop10.refill(method='linear')

    len_minimum += int(len_min_chunk)
    len_maximum += int(len_min_chunk)
    dyn_crop11 = cp(dyn_crop)
    dyn_crop11.crop_dyn(tmin=len_minimum, tmax=len_maximum)
    dyn_crop11.zap()
    dyn_crop11.refill(method='linear')

    len_minimum += int(len_min_chunk)
    len_maximum += int(len_min_chunk)
    dyn_crop12 = cp(dyn_crop)
    dyn_crop12.crop_dyn(tmin=len_minimum, tmax=len_maximum)
    dyn_crop12.zap()
    dyn_crop12.refill(method='linear')

    len_minimum += int(len_min_chunk)
    len_maximum += int(len_min_chunk)
    dyn_crop13 = cp(dyn_crop)
    dyn_crop13.crop_dyn(tmin=len_minimum, tmax=len_maximum)
    dyn_crop13.zap()
    dyn_crop13.refill(method='linear')

    len_minimum += int(len_min_chunk)
    len_maximum += int(len_min_chunk)
    dyn_crop14 = cp(dyn_crop)
    dyn_crop14.crop_dyn(tmin=len_minimum, tmax=len_maximum)
    dyn_crop14.zap()
    dyn_crop14.refill(method='linear')

    len_minimum += int(len_min_chunk)
    len_maximum += int(len_min_chunk)
    dyn_crop15 = cp(dyn_crop)
    dyn_crop15.crop_dyn(tmin=len_minimum, tmax=len_maximum)
    dyn_crop15.zap()
    dyn_crop15.refill(method='linear')

    len_minimum += int(len_min_chunk)
    len_maximum += int(len_min_chunk)
    dyn_crop16 = cp(dyn_crop)
    dyn_crop16.crop_dyn(tmin=len_minimum, tmax=len_maximum)
    dyn_crop16.zap()
    dyn_crop16.refill(method='linear')

    len_minimum += int(len_min_chunk)
    len_maximum += int(len_min_chunk)
    dyn_crop17 = cp(dyn_crop)
    dyn_crop17.crop_dyn(tmin=len_minimum, tmax=len_maximum)
    dyn_crop17.zap()
    dyn_crop17.refill(method='linear')

    len_minimum += int(len_min_chunk)
    len_maximum += int(len_min_chunk)
    dyn_crop18 = cp(dyn_crop)
    dyn_crop18.crop_dyn(tmin=len_minimum, tmax=len_maximum)
    dyn_crop18.zap()
    dyn_crop18.refill(method='linear')

    len_minimum += int(len_min_chunk)
    len_maximum += int(len_min_chunk)
    dyn_crop19 = cp(dyn_crop)
    dyn_crop19.crop_dyn(tmin=len_minimum, tmax=len_maximum)
    dyn_crop19.zap()
    dyn_crop19.refill(method='linear')

    len_minimum += int(len_min_chunk)
    len_maximum += int(len_min_chunk)
    dyn_crop20 = cp(dyn_crop)
    dyn_crop20.crop_dyn(tmin=len_minimum, tmax=np.inf)
    dyn_crop20.zap()
    dyn_crop20.refill(method='linear')

    dyn_all = dyn_crop1 + dyn_crop2 + dyn_crop3 + dyn_crop4 + dyn_crop5 + \
        dyn_crop6 + dyn_crop7 + dyn_crop8 + dyn_crop9 + dyn_crop10 + \
        dyn_crop11 + dyn_crop12 + dyn_crop13 + dyn_crop14 + dyn_crop15 + \
        dyn_crop16 + dyn_crop17 + dyn_crop18 + dyn_crop19 + dyn_crop20

    dyn_all.plot_dyn(filename='/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/New/SpectraPlots/'+str(observation_date2)+'_FullSpectra.pdf', dpi=400)
    dyn_all.write_file(filename=str(outfile))
