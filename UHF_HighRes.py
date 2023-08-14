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
        get_true_anomaly, scint_velocity
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import glob
from copy import deepcopy as cp
# import os

psrname = 'J0737-3039A'
pulsar = '0737-3039A'

wd = '/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/'

eclipsefile = wd + 'Datafiles/Eclipse_mjd.txt'

##############################################################################


def SearchEclipse(start_mjd, tobs):
    end_mjd = start_mjd + tobs/86400
    Eclipse_mjd = np.genfromtxt(eclipsefile, delimiter=',',
                                encoding=None, dtype=float)
    Eclipse_events = np.array(np.where((Eclipse_mjd > start_mjd) *
                              (Eclipse_mjd < end_mjd)))
    if Eclipse_events.size == 0:
        Eclipse_index = None
        print("No Eclispe in dynspec")
    else:
        Eclipse_events_mjd = Eclipse_mjd[Eclipse_events]
        mjds = start_mjd + dyn.times/86400
        Eclipse_index = np.argmin(abs(mjds - Eclipse_events_mjd))
    return Eclipse_index


##############################################################################
datadir = wd + 'NewDynspec/'
outdir = wd + 'NewDataFiles/'
spectradir = wd + 'NewSpectraPlots/'
dynspecs = sorted(glob.glob(datadir + '/*.dynspec'))
outfile = str(outdir) + str(psrname) + '_ScintillationResults_UHF.txt'
par_dir = '/Users/jacobaskew/Desktop/Swinburne_Daniel/ParFiles/'

time_bin = 30
freq_bin = 30
zap = True
linear = True

# First I want to sort the incoming data into groups of three observations.
# This is because each observing run has 3 components short, long, short.
# If we take the observations in grouos of three it should be fine.

for dynspec in dynspecs:

    # This code gets rid of the eclipse if we have an updated list #
    # dyn = Dynspec(filename=dynspec, process=False)
    # start_mjd = dyn.mjd
    # tobs = dyn.tobs
    # Eclipse_index = SearchEclipse(start_mjd, tobs)
    # if Eclipse_index != None:
    #     dyn.dyn[:, Eclipse_index-3:Eclipse_index+3] = 0

    File1 = dynspec.split(str(datadir))[1]
    Filename = str(File1.split('.')[0])

    try:
        # Option 1 #
        # dyn = Dynspec(filename=dynspec, process=False)
        # Option 2 #
        dyn1 = Dynspec(filename=dynspecs[0], process=False)
        dyn2 = Dynspec(filename=dynspecs[1], process=False)
        dyn3 = Dynspec(filename=dynspecs[2], process=False)
        dyn = dyn1 + dyn2 + dyn3
        dyn.trim_edges()
        Fmin = np.min(dyn.freqs)+48
        Fmax = np.max(dyn.freqs)-48
        Fmin = 650
        Fmax = 850
        dyn.crop_dyn(fmin=650, fmax=850, tmin=50, tmax=150)
        # start_mjd = dyn.mjd
        # tobs = dyn.tobs
        # Eclipse_index = SearchEclipse(start_mjd, tobs)
        # if Eclipse_index != None:
        #     dyn.dyn[:, Eclipse_index-3:Eclipse_index+3] = 0
        dyn.plot_dyn(filename=str(spectradir) + str(Filename) + '_Spectra.pdf')
    except Exception as e:
        print(e)
        continue

    dyn_crop = cp(dyn)
    dyn_crop.crop_dyn(fmin=Fmin, fmax=Fmax)
    f_min_init = Fmax - freq_bin
    f_max = Fmax
    f_init = int((f_max + f_min_init)/2)
    bw_init = int(f_max - f_min_init)

    for istart_t in range(0, int(dyn.tobs/60), int(time_bin)):
        try:
            dyn_new = cp(dyn_crop)
            dyn_new.crop_dyn(fmin=f_min_init, fmax=f_min_init+bw_init,
                             tmin=istart_t, tmax=istart_t + time_bin)
            dyn_new.trim_edges()
            if dyn_new.tobs/60 < (time_bin - 5):
                print("Spectra rejected: Not enough time!")
                print(str(dyn_new.tobs/60) + " < " + str(time_bin - 5))
                continue
            if (np.max(dyn_new.freqs) - np.min(dyn_new.freqs)) < freq_bin - 5:
                print("Spectra rejected: Not enough frequency!")
                print(str((np.max(dyn_new.freqs) - np.min(dyn_new.freqs))) +
                      " < " + str(freq_bin - 5))
                continue
            print("Spectra Time accepeted: ")
            print(str(dyn_new.tobs/60) + " > " + str(time_bin - 5))
            print("Spectra Frequency accepeted: ")
            print(str((np.max(dyn_new.freqs) - np.min(dyn_new.freqs))) + " > "
                  + str(freq_bin - 5))
            if zap:
                dyn_new.zap()
            if linear:
                dyn_new.refill(linear=True)
            else:
                dyn_new.refill(linear=False)
            dyn_new.get_acf_tilt(plot=False, display=False)
            dyn_new.get_scint_params(method='acf2d_approx',
                                     flux_estimate=True,
                                     plot=False, display=False)
            # dyn_new.plot_dyn(filename=str(HighFreqDir) + str(Filename) + '/' +
            #                  str(Filename) + '_' + str(freq_bin) + 'MHz_' +
            #                  str(time_bin) + '_fmin' + str(f_min_init) +
            #                  '_tmin' + str(istart_t) + '_Spectra.png')
            # dyn_new.plot_dyn(filename=str(HighFreqDir) + 'Spectra_' +
            #                  str(freq_bin) + 'MHz_' + str(time_bin) + 'min.txt'
            #                  + '/' + str(Filename) + '_' + str(freq_bin) +
            #                  'MHz_' + str(time_bin) + '_fmin' + str(f_min_init)
            #                  + '_tmin' + str(istart_t) + '_Spectra.png')
            write_results(outfile, dyn=dyn_new)
        except Exception as e:
            print(e)
            continue
    for x in range(0, 100):
        if x == 0:
            f_min_new = f_min_init - freq_bin
            bw_new = (f_min_new / f_init)**2 * bw_init
            if f_min_new < Fmin:
                continue
            for istart_t in range(0, int(dyn.tobs/60), int(time_bin)):
                try:
                    dyn_new = cp(dyn_crop)
                    dyn_new.crop_dyn(fmin=f_min_new, fmax=f_min_new+bw_new,
                                     tmin=istart_t, tmax=istart_t + time_bin)
                    dyn_new.trim_edges()
                    if dyn_new.tobs/60 < (time_bin - 5):
                        print("Spectra rejected: Not enough time!")
                        print(str(dyn_new.tobs/60) + " < " + str(time_bin - 5))
                        continue
                    if (np.max(dyn_new.freqs) - np.min(dyn_new.freqs)) < bw_new - 5:
                        print("Spectra rejected: Not enough frequency!")
                        print(str((np.max(dyn_new.freqs) -
                                   np.min(dyn_new.freqs))) + " < " +
                              str(bw_new - 5))
                        continue
                    print("Spectra Time accepeted: ")
                    print(str(dyn_new.tobs/60) + " > " + str(time_bin - 5))
                    print("Spectra Frequency accepeted: ")
                    print(str((np.max(dyn_new.freqs) - np.min(dyn_new.freqs)))
                          + " > " + str(bw_new - 5))
                    if zap:
                        dyn_new.zap()
                    if linear:
                        dyn_new.refill(linear=True)
                    else:
                        dyn_new.refill(linear=False)
                    dyn_new.get_acf_tilt(plot=False, display=False)
                    dyn_new.get_scint_params(method='acf2d_approx',
                                             flux_estimate=True,
                                             plot=False, display=False)
                    # dyn_new.plot_dyn(filename=str(HighFreqDir) + str(Filename) + '/' + str(Filename) + '_' + str(freq_bin) + 'MHz_' + str(time_bin) + '_fmin' + str(f_min_new) + '_tmin' + str(istart_t) + '_Spectra.png')  #
                    # dyn_new.plot_dyn(filename=str(HighFreqDir) + 'Spectra_' + str(freq_bin) + 'MHz_' + str(time_bin) + 'min.txt' + '/' + str(Filename) + '_' + str(freq_bin) + 'MHz_' + str(time_bin) + '_fmin' + str(f_min_new) + '_tmin' + str(istart_t) + '_Spectra.png')  #
                    write_results(outfile, dyn=dyn_new)
                except Exception as e:
                    print(e)
                    continue
        elif x > 0:
            f_min_new = f_min_new - freq_bin
            bw_new = (f_min_new / f_init)**2 * bw_init
            if f_min_new < Fmin:
                continue
            for istart_t in range(0, int(dyn.tobs/60), int(time_bin)):
                try:
                    dyn_new = cp(dyn_crop)
                    dyn_new.crop_dyn(fmin=f_min_new, fmax=f_min_new+bw_new,
                                     tmin=istart_t, tmax=istart_t + time_bin)
                    dyn_new.trim_edges()
                    if dyn_new.tobs/60 < (time_bin - 5):
                        print("Spectra rejected: Not enough time!")
                        print(str(dyn_new.tobs/60) + " < " + str(time_bin - 5))
                        continue
                    if (np.max(dyn_new.freqs) - np.min(dyn_new.freqs)) < bw_new - 5:
                        print("Spectra rejected: Not enough frequency!")
                        print(str((np.max(dyn_new.freqs) -
                                   np.min(dyn_new.freqs))) + " < " +
                              str(bw_new - 5))
                        continue
                    print("Spectra Time accepeted: ")
                    print(str(dyn_new.tobs/60) + " > " + str(time_bin - 5))
                    print("Spectra Frequency accepeted: ")
                    print(str((np.max(dyn_new.freqs) - np.min(dyn_new.freqs)))
                          + " > " + str(bw_new - 5))
                    if zap:
                        dyn_new.zap()
                    if linear:
                        dyn_new.refill(linear=True)
                    else:
                        dyn_new.refill(linear=False)
                    dyn_new.get_acf_tilt(plot=False, display=False)
                    dyn_new.get_scint_params(method='acf2d_approx',
                                             flux_estimate=True,
                                             plot=False, display=False)
                    # dyn_new.plot_dyn(filename=str(HighFreqDir) + str(Filename) + '/' + str(Filename) + '_' + str(freq_bin) + 'MHz_' + str(time_bin) + '_fmin' + str(f_min_new) + '_tmin' + str(istart_t) + '_Spectra.png')  #
                    # dyn_new.plot_dyn(filename=str(HighFreqDir) + 'Spectra_' + str(freq_bin) + 'MHz_' + str(time_bin) + 'min.txt' + '/' + str(Filename) + '_' + str(freq_bin) + 'MHz_' + str(time_bin) + '_fmin' + str(f_min_new) + '_tmin' + str(istart_t) + '_Spectra.png')  #
                    write_results(outfile, dyn=dyn_new)
                except Exception as e:
                    print(e)
                    continue

results_dir = outdir
params = read_results(outfile)

pars = read_par(str(par_dir) + str(psrname) + '.par')

# Read in arrays
mjd = float_array_from_dict(params, 'mjd')  # MJD for observation start
df = float_array_from_dict(params, 'df')  # channel bandwidth
dnu = float_array_from_dict(params, 'dnu')  # scint bandwidth
dnu_est = float_array_from_dict(params, 'dnu_est')  # estimated bandwidth
dnuerr = float_array_from_dict(params, 'dnuerr')
tau = float_array_from_dict(params, 'tau')
tauerr = float_array_from_dict(params, 'tauerr')
freq = float_array_from_dict(params, 'freq')
bw = float_array_from_dict(params, 'bw')
scintle_num = float_array_from_dict(params, 'scintle_num')
tobs = float_array_from_dict(params, 'tobs')  # tobs in second
rcvrs = np.array([rcvr[0] for rcvr in params['name']])

# Sort by MJD
sort_ind = np.argsort(mjd)

df = np.array(df[sort_ind]).squeeze()
dnu = np.array(dnu[sort_ind]).squeeze()
dnu_est = np.array(dnu_est[sort_ind]).squeeze()
dnuerr = np.array(dnuerr[sort_ind]).squeeze()
tau = np.array(tau[sort_ind]).squeeze()
tauerr = np.array(tauerr[sort_ind]).squeeze()
mjd = np.array(mjd[sort_ind]).squeeze()
rcvrs = np.array(rcvrs[sort_ind]).squeeze()
freq = np.array(freq[sort_ind]).squeeze()
tobs = np.array(tobs[sort_ind]).squeeze()
scintle_num = np.array(scintle_num[sort_ind]).squeeze()
bw = np.array(bw[sort_ind]).squeeze()

indicies = np.argwhere((tauerr < 0.6*tau) * (dnuerr < 0.6*dnu))

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


mjd_annual = mjd % 365.2425
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
phase[phase > 360] = phase[phase > 360] - 360

Font = 30
Size = 80*np.pi  # Determines the size of the datapoints used
font = {'size': 28}
matplotlib.rc('font', **font)

fig = plt.figure(figsize=(20, 10))
fig.subplots_adjust(hspace=0.5, wspace=0.5)
cm = plt.cm.get_cmap('viridis')
z = freq
sc = plt.scatter(phase, dnu, c=z, cmap=cm, s=Size, alpha=0.6)
plt.colorbar(sc)
plt.errorbar(phase, dnu, yerr=dnuerr, fmt=' ', ecolor='k',
             elinewidth=2, capsize=3, alpha=0.55)
xl = plt.xlim()
plt.plot(xl, (df[0], df[0]), color='C2')
plt.xlabel('Orbital Phase (degrees)')
plt.ylabel('Scintillation Bandwidth (MHz)')
plt.title(psrname + r', $\Delta\nu$')
plt.grid(True, which="both", ls="-", color='0.65')
plt.xlim(xl)
# plt.savefig(str(Dnudir) + "Dnu_Orbital_Freq.png")
plt.show()
plt.close()
