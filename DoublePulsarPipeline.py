#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 13:52:49 2021

@author: jacobaskew
"""
##############################################################################
#This is the start of the Double Pulsar Project Pipeline
#onescripttorulethemall

# Objectives:
# Flag out RFI by cutting into windows of frequency
# Measure, dnu, tau and eta
# Include de-velocity function to de-broaden arcs in sspec
# Include Flagging of Eclispe in J0737-3039A
##############################################################################
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
import os
# import bilby
# from bilby.core import result
# import pdb
from astropy.time import Time

psrname = '0737-3039A'
datadir = '/Users/jacobaskew/Desktop/DoublePulsar_Project/' + str(psrname) + '/Data/'
plotdir = '/Users/jacobaskew/Desktop/DoublePulsar_Project/' + str(psrname) + '/Plots/'
spectradir = '/Users/jacobaskew/Desktop/DoublePulsar_Project/' + str(psrname) + '/Spectra/'
outdir = '/Users/jacobaskew/Desktop/DoublePulsar_Project/' + str(psrname) + '/Datafiles/'
outfile = str(outdir) + str(psrname) + '_Results.txt'

Log_Option = 0
Eclipse_mjd = np.genfromtxt(str(outdir) + 'Eclipse_mjd.txt', delimiter=',',encoding=None, dtype=float)
# dynspecs = sorted(glob.glob(datadir + psrname + '/*2019-12-14-19:44*.dynspec'))
dynspecs = sorted(glob.glob(datadir + '*.dynspec'))
#############################################################################
# Predicting the times of the eclipses
def Date2Mjd():
    Eclipse_Date = np.genfromtxt(str(outdir) + 'EclipseDate.txt', delimiter=',',encoding=None, dtype=str)
    Eclipse_Date_New = []
    for x in range(0, len(Eclipse_Date)):
        Eclipse_Date_New.append(Eclipse_Date[x].split('-')[0] + '-' + Eclipse_Date[x].split('-')[1] + '-' + Eclipse_Date[x].split('-')[2] + ' ' + Eclipse_Date[x].split('-')[3] + '.00')
    t = Time(Eclipse_Date_New, format='iso')
    Eclipse_mjd = t.mjd  # observation year date and time
    Eclipse_yday = t.yday
    np.savetxt(str(outdir) + "Eclipse_mjd.txt", Eclipse_mjd, fmt='%s')
    np.savetxt(str(outdir) + "Eclipse_yday.txt", Eclipse_yday, fmt='%s')

def SearchEclipse(start_mjd, tobs):
        
    end_mjd = start_mjd + tobs/86400
    Eclipse_mjd = np.genfromtxt(str(outdir) + 'Eclipse_mjd.txt', delimiter=','
                                ,encoding=None, dtype=float)
    Eclipse_events = np.where((Eclipse_mjd > start_mjd) * 
                              (Eclipse_mjd < end_mjd))
    Eclipse_events_mjd = Eclipse_mjd[Eclipse_events]
    mjds = start_mjd + dyn.times/86400
    Eclipse_index = np.argmin(abs(mjds - Eclipse_events_mjd))

    return Eclipse_index
# Eclipse_mjd = np.genfromtxt(str(outdir) + 'Eclipse_mjd.txt', delimiter=',',encoding=None, dtype=float)

# start_mjd = 59558.60644675926
# end_mjd = 59658.70873842593
# np.where((Eclipse_mjd > start_mjd) * (Eclipse_mjd < end_mjd))

# SearchEclipse(start_mjd, end_mjd)
# #############################################################################
# Begin by looking at a single spectra then automating on the rest and see what happens

# dyn = Dynspec(filename='/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/Data/J0737-3039A_2019-07-20-06:52:56_ch5.0_sub5.0.ar.dynspec', process=False)

# dyn.trim_edges()
# dyn.zap
# for x in range(0, dyn.dyn.shape[1]):
#     for y in range(0, dyn.dyn.shape[0]):
#         if dyn.dyn[y,x] == 0:
#             print()
#             print(x)
#             print(y)
#             dyn.dyn[y,x] = 50
# dyn.refill(linear=False)
# dyn.calc_acf()
# dyn.plot_dyn(filename='/Users/jacobaskew/Desktop/Spectra.png')
# # Maximum frequency # 1671.039062
# #Top of first RFI zone
# dyn.dyn[876,:] = 100 # 1628.40625
# #Bottom of first RFI zone
# dyn.dyn[861,:] = 100 # 1615.867188
# #Top of second RFI zone
# dyn.dyn[849,:] = 100 # 1605.835938
# #Bottom of second RFI zone
# dyn.dyn[839,:] = 100 # 1597.476562
# #Top of third RFI zone
# dyn.dyn[816,:] = 100 # 1578.25
# #Bottom of third RFI zone
# dyn.dyn[809,:] = 100 # 1572.398438
# #Top of fourth RFI zone
# dyn.dyn[799,:] = 100 # 1564.039062
# #Bottom of fourth RFI zone
# dyn.dyn[750,:] = 100 # 1523.078125
# #Top of fifth RFI zone
# dyn.dyn[629,:] = 100 # 1421.929688
# #Bottom of fifth RFI zone
# dyn.dyn[625,:] = 100 # 1418.585938
# #Top of sixth RFI zone
# dyn.dyn[462,:] = 100 # 1282.328125
# #Bottom of sixth RFI zone
# dyn.dyn[435,:] = 100 # 1259.757812
# #Top of seventh RFI zone
# dyn.dyn[399,:] = 100 # 1229.664062
# #Bottom of seventh RFI zone
# dyn.dyn[394,:] = 100 # 1225.484375
# #Top of eighth RFI zone
# dyn.dyn[381,:] = 100 # 1214.617188
# #Bottom of eighth RFI zone
# dyn.dyn[364,:] = 100 # 1200.40625
# #Top of ninth RFI zone
# dyn.dyn[345,:] = 100 # 1184.523438
# #Bottom of ninth RFI zone
# dyn.dyn[326,:] = 100 # 1168.640625
# #Top of tenth RFI zone
# dyn.dyn[296,:] = 100 # 1143.5625
# #Bottom of tenth RFI zone
# dyn.dyn[218,:] = 100 # 1078.359375
# #Top of eleventh RFI zone
# dyn.dyn[189,:] = 100 # 1054.117188
# #Bottom of eleventh RFI zone
# dyn.dyn[153,:] = 100 # 1024.023438
# #Top of twelfth RFI zone
# dyn.dyn[78,:] = 100 # 961.328125
# #Bottom of twelfth RFI zone
# dyn.dyn[45,:] = 100 # 933.742188
# # Minimum frequency # 896.125

# dyn.plot_dyn(filename='/Users/jacobaskew/Desktop/Spectra.png')
##############################################################################
# Here I am going to try an automatic version of what is above
# AcceptedFractionalRFI = 0.2

# dyn = Dynspec(filename='/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/Data/J0737-3039A_2019-07-20-06:52:56_ch5.0_sub5.0.ar.dynspec', process=False)

# dyn.trim_edges()
# dyn.zap
# # dyn.refill(linear=False)
# dyn.calc_acf()
# dyn.plot_dyn(filename='/Users/jacobaskew/Desktop/Spectra.png')
# Spectra = dyn.dyn
# Flagged = np.where(Spectra == 0)
# BadFreq = []
# for y in range(0, Spectra.shape[0]):
#     if int(len(Spectra[y,:]) * AcceptedFractionalRFI) < int(len(np.array(np.where(Spectra[y,:] == 0)).flatten())):
#         BadFreq.append(y)
# Spectra[BadFreq,:] = 0.001
# dyn.plot_dyn(filename='/Users/jacobaskew/Desktop/Spectra.png')
# SpectraZoneStart = []
# SpectraZoneEnd = []
# RFIZoneStart = []
# RFIZoneEnd = []
# for x in range(0, len(BadFreq)):
#     for t in range(0, Spectra.shape[0]):
#         if t != BadFreq[x] and len(RFIZoneStart) == len(RFIZoneEnd):
#             RFIZoneStart.append(t)
#             continue
#         elif t != BadFreq[x] and len(RFIZoneStart) > len(RFIZoneEnd) and t+1 == BadFreq[x]:
#             RFIZoneEnd.append(t)
#             continue
#         elif t == BadFreq[x] and len(SpectraZoneStart) == len(SpectraZoneEnd):
#             SpectraZoneStart.append(t)
#             continue   
#         elif t != BadFreq[x] and len(SpectraZoneStart) > len(SpectraZoneEnd) and t+1 != BadFreq[x]:
#             SpectraZoneEnd.append(t)
#             continue
#         else:
#             # print("Middle Point")
#             continue
            
        # if BadFreq[x] != BadFreq[x] + 1:
        #     if BadFreq[x+1] == BadFreq[x] + 1:
        #         continue
#############################################################################
# Begin by looking at a single spectra then automating on the rest and see what happens
for dynspec in dynspecs:
    
    File1=dynspec.split(str(datadir))[1]
    Filename=str(File1.split('.')[0])

    dyn = Dynspec(filename=dynspec, process=False)
    start_mjd=dyn.mjd
    tobs = dyn.tobs
    SearchEclipse(start_mjd, tobs)

    if dyn.freq > 1000 and dyn.freq < 1500:
        
        dyn = Dynspec(filename=dynspec, process=False)

        FreqRange = 'High'
        Bandwidth = 50
        dyn.crop_dyn(fmin=1650, fmax=1700)
        dyn.trim_edges()
        dyn.zap
        dyn.refill(linear=False)
        dyn.calc_acf(method='acf2d_approx',flux_estimate=True)
        dyn.plot_dyn(filename=str(spectradir) + str(Filename) + '_' + str(FreqRange) + '_Spectra.png')
        dyn.plot_acf(fit=True) # filename=str(plotdir) + str(Filename) + '_' + str(FreqRange) + '_ACF.png',
        write_results(str(outdir) + 'J' + str(psrname) + '_ScintillationResults.txt', dyn=dyn)
        
        dyn = Dynspec(filename=dynspec, process=False)

        FreqRange = 'Mid'
        Bandwidth = 200
        
        dyn.crop_dyn(fmin=1300, fmax=1500)
        dyn.trim_edges()
        dyn.zap
        dyn.refill(linear=False)
        dyn.calc_acf(method='acf2d_approx',flux_estimate=True)
        dyn.plot_dyn(filename=str(spectradir) + str(Filename) + '_' + str(FreqRange) + '_Spectra.png')
        dyn.plot_acf(fit=True) # filename=str(plotdir) + str(Filename) + '_' + str(FreqRange) + '_ACF.png',
        write_results(str(outdir) + 'J' + str(psrname) + '_ScintillationResults.txt', dyn=dyn)
        
        dyn = Dynspec(filename=dynspec, process=False)

        FreqRange = 'Low'
        Bandwidth = 100
        dyn.crop_dyn(fmin=975, fmax=1075)
        dyn.trim_edges()
        dyn.zap
        dyn.refill(linear=False)
        dyn.calc_acf(method='acf2d_approx',flux_estimate=True)
        dyn.plot_dyn(filename=str(spectradir) + str(Filename) + '_' + str(FreqRange) + '_Spectra.png')
        dyn.plot_acf(fit=True) # filename=str(plotdir) + str(Filename) + '_' + str(FreqRange) + '_ACF.png',
        write_results(str(outdir) + 'J' + str(psrname) + '_ScintillationResults.txt', dyn=dyn)
    

##############################################################################
# Me attmepting to determine the relative change in bandwidth

# fc1 = ( 1671.039062 + 1628.40625 ) / 2
# fc2 = ( 1523.078125 + 1421.929688 ) / 2
# fdiff = fc1-fc2
# fbw1 = 1671.039062 - 1628.40625
# fbw2 = 1523.078125 - 1421.929688

# fbw2_scaled = fbw1 * (1 - (fc2**(1/2) / fdiff))


# fc1 = 1675
# fc2 = 1400

# fdiff = fc1-fc2
# fbw1 = 50
# # fbw2 = k*(1400**(-4))

# Xdata = fbw1 + fdiff
# Ydata = fdiff**4
# plt.plot(Xdata, Ydata)

# Power = 10
# Number = 1/(5**(5/3))
# Power2 = Power * Number

Eclipse_mjd = np.genfromtxt(str(outdir) + 'Eclipse_mjd.txt', delimiter=',',encoding=None, dtype=float)

dyn = Dynspec(filename='/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/Data/J0737-3039A_2020-03-20-20:18:32_ch5.0_sub5.0.ar.dynspec', process=False)
start_mjd=dyn.mjd
tobs = dyn.tobs

Eclipse_index = SearchEclipse(start_mjd, tobs)

dyn.dyn[:,Eclipse_index-3:Eclipse_index+3] = 0



dyn.plot_dyn(filename='/Users/jacobaskew/Desktop/Spectra.png')



# Eclipse_events_mjd = Eclipse_mjd[SearchEclipse(start_mjd, tobs)]
# dyn.crop_dyn(fmin=1650, fmax=1750)
dyn.trim_edges()
dyn.zap
dyn.refill(linear=False)
dyn.calc_acf()
dyn.get_acf_tilt()
dyn.get_scint_params(method='acf2d_approx',
                         flux_estimate=True)
dyn.plot_dyn(filename='/Users/jacobaskew/Desktop/Spectra.png')

k = (50)/(dyn.freq**(-4))
bw_mid = k*(1400**(-4))

write_results("/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/Result.txt", dyn=dyn)

# freq,bw,tobs,dt,df,tau,tauerr,dnu,dnuerr,dnu_est,acf_tilt,acf_tilt_err,phasegrad,phasegraderr
# 1660.59,21.73,8,0.83594,66.90090400387383,4.531984657684668,0.5357861272498707,0.0519485594952074
# 1400.2,199.79,8,0.83594,78.06331448106293,2.299158745997239,0.46582077197364136,0.01820632486883046
# 1024.86,99.48,8,0.83594,65.20188483307574,1.6441149642913528,0.2639341711818677,0.009667785106979214


k_list = []
for x in range(0, dyn.dyn.shape[0]):
    k_list.append(50*((x)**2))

k = np.mean(k_list)
fbw2 = k*(1600**(-2))
print(fbw2)

##############################################################################
# Looking at modelling the eclipse of J0737

# dyn = Dynspec(filename='/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/Data/J0737-3039A_2019-07-20-06:22:48_ch5.0_sub5.0.ar.dynspec', process=False)

# dyn.trim_edges()
# dyn.zap
# dyn.refill(linear=False)
# dyn.calc_acf()
# dyn.plot_dyn(filename='/Users/jacobaskew/Desktop/Spectra.png')

# t = Time(dyn.mjd, format='mjd')
# start_time = t.decimalyear  # observation year date and time

# # HrsMinSec_start = str(start_time.split(':')[2] + ':' + start_time.split(':')[3] + ':' + start_time.split(':')[4])
# print()
# print(start_time)

# dyn = Dynspec(filename='/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/Data/J0737-3039A_2019-07-20-08:38:56_ch5.0_sub5.0.ar.dynspec', process=False)

# dyn.trim_edges()
# dyn.zap
# dyn.refill(linear=False)
# dyn.calc_acf()
# dyn.plot_dyn(filename='/Users/jacobaskew/Desktop/Spectra.png')

# BinaryPhase = 2.4
# Event = 2.4/4

# t = Time(dyn.mjd, format='mjd')
# start_time = t.decimalyear  # observation year date and time

# # HrsMinSec_start = str(start_time.split(':')[2] + ':' + start_time.split(':')[3] + ':' + start_time.split(':')[4])
# print()
# print(start_time)

# 2019.5486736419139
# 2019.5489327571372

# 86400
# HrsMinSec_end = '06:52:56.371'

# k = 54*((1600)**4)



# BinaryPhase = 2.4
# Event = 2.4/4

# #Time between observations
# 2.271404047829492
# #How many hours passed since start of ob1 and end of ob2
# 2.7738484922739364
# #Number of events seen
# 2
# #Relative time of events where 0 = start ob1
# Num1 = 0.11666666666666667
# Num2 = 2.271404047829492 + (18/60) #2.5714040478294917

# Ydata = np.arange(11,1000, 25)
# Xdata = (12, 257)

# # plt.fig()
# plt.hist(Ydata, bins=150, color='r', alpha=0.1)
# plt.hist(Xdata, bins=50, color='g', alpha=0.75)
# plt.show()
##############################################################################
# Calculations #
ScintillationData = read_results(str(outdir) + 'J' + str(psrname) + '_ScintillationResults.txt')


# Read in arrays
mjd = float_array_from_dict(ScintillationData, 'mjd')  # MJD for observation start
dnu = float_array_from_dict(ScintillationData, 'dnu')  # scint bandwidth
dnuerr = float_array_from_dict(ScintillationData, 'dnuerr')
tau = float_array_from_dict(ScintillationData, 'tau')
tauerr = float_array_from_dict(ScintillationData, 'tauerr')
freq = float_array_from_dict(ScintillationData, 'freq')
bw = float_array_from_dict(ScintillationData, 'bw')
df = float_array_from_dict(ScintillationData, 'df')
dt = float_array_from_dict(ScintillationData, 'dt')
tobs = float_array_from_dict(ScintillationData, 'tobs')  # tobs in seconds
rcvrs = np.array([rcvr[0] for rcvr in ScintillationData['name']])

# # Make MJD centre of observation, instead of start
mjd = mjd + tobs/86400/2

# Form Viss from the data
Aiss = 2.78*10**4  # thin screen
# D = pars['DIST_A']  # kpc

D = 1.15 # kpc
Derr = 0.22  # kpc

#0737-3039A: D = 1.150 +/- 0.220 kpc Parrallax/VLBI ^+220 _−160 (Deller 2009)

X = 1
#Aisserr = Aiss*0.2

freqs = freq*10**(-3)
if Log_Option == 0:
    viss = Aiss * np.sqrt(D * dnu) / (freq/1000 * tau)
#
if Log_Option == 1:
    viss = np.log(Aiss * np.sqrt(D * dnu) / (freq/1000 * tau))

ddnu = abs(Aiss*np.sqrt(dnu*D*X)/(2*dnu*freqs*tau))
dtau = abs(-Aiss*np.sqrt(dnu*D*X)/(freqs*tau**2))
dD = Aiss*np.sqrt(abs(dnu*D*X))/(2*D*freqs*tau)
#dAiss = np.sqrt(dnu*D*X)/((freqs)*(tau))
if Log_Option == 0:
    visserr = np.sqrt((abs(ddnu)*dnuerr)**2 + (abs(dtau)*tauerr)**2 + (abs(dD)*Derr)**2) # + (abs(dAiss)*Aisserr)**2)
if Log_Option == 1:
    visserr = np.sqrt( (abs( (dnuerr)/(dnu*2) ))**2 + (abs( (tauerr)/(tau) ))**2 + (abs( (Derr)/(D*2) ))**2 )# + (abs( (ferr)/(f) ))**2 + (abs( (dAiss)/(Aiss) ))**2 )


##############################################################################
# Plotting Results #

Font = 35
Size = 80*np.pi #Determines the size of the datapoints used
font = {'size'   : 32}
matplotlib.rc('font', **font)

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
plt.show()

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
plt.show()

# MJD #
fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(1, 1, 1)
fig.subplots_adjust(hspace=0.5, wspace=0.5)
plt.scatter(mjd, viss, facecolor='black', marker='o',  s=Size, alpha=0.75, label='Scintillation Velocity')
plt.errorbar(mjd, viss, yerr=visserr, fmt=' ', ecolor='black', elinewidth = 3, capsize = 0, alpha=0.4)
# plt.plot(mjd, model, color='purple', label='Effective Velocity (Fitted Model)', linewidth = 4, alpha=0.7)
plt.title('Effective Velocity ' + str(psrname), fontsize=Font, ha='center')
plt.xlabel('MJD', fontsize=Font, ha='center')
if Log_Option == 0:
    plt.ylabel(r'$V_{iss}$ ($kms^{-1}$)', fontsize=Font, ha='center')
if Log_Option == 1:
    plt.ylabel(r'Natural Log $V_{iss}$ ($kms^{-1}$)', fontsize=Font, ha='center')
ax.legend(fontsize='xx-small')
plt.grid(True, which="both", ls="-", color='0.65')
plt.savefig(str(plotdir) + str(psrname) + "_" + "Viss_yrs.png")
plt.show()









