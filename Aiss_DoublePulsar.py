#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 17:21:08 2023

@author: jacobaskew
"""
###############################################################################
# Importing neccessary things #
from __future__ import division 
from scintools.scint_sim import ACF
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from scipy import interpolate
import pickle
import os
import glob
import pandas as pd
###############################################################################
Size = 80*np.pi  # Determines the size of the datapoints used
font = {'size': 20}
matplotlib.rc('font', **font)
###############################################################################


def interpolated_intercept(x, y1, y2):
    """Find the intercept of two curves, given by the same x data"""

    def intercept(point1, point2, point3, point4):
        """find the intersection between two lines
        the first line is defined by the line between point1 and point2
        the first line is defined by the line between point3 and point4
        each point is an (x,y) tuple.

        So, for example, you can find the intersection between
        intercept((0,0), (1,1), (0,1), (1,0)) = (0.5, 0.5)

        Returns: the intercept, in (x,y) format
        """    

        def line(p1, p2):
            A = (p1[1] - p2[1])
            B = (p2[0] - p1[0])
            C = (p1[0]*p2[1] - p2[0]*p1[1])
            return A, B, -C

        def intersection(L1, L2):
            D  = L1[0] * L2[1] - L1[1] * L2[0]
            Dx = L1[2] * L2[1] - L1[1] * L2[2]
            Dy = L1[0] * L2[2] - L1[2] * L2[0]

            x = Dx / D
            y = Dy / D
            return x,y

        L1 = line([point1[0],point1[1]], [point2[0],point2[1]])
        L2 = line([point3[0],point3[1]], [point4[0],point4[1]])

        R = intersection(L1, L2)

        return R

    idx = np.argwhere(np.diff(np.sign(y1 - y2)) != 0)
    xc, yc = intercept((x[idx], y1[idx]),((x[idx+1], y1[idx+1])), ((x[idx], y2[idx])), ((x[idx+1], y2[idx+1])))
    return xc,yc


# def main():
#     x  = np.linspace(1, 4, 20)
#     y1 = np.sin(x)
#     y2 = 0.05*x

#     plt.plot(x, y1, marker='o', mec='none', ms=4, lw=1, label='y1')
#     plt.plot(x, y2, marker='o', mec='none', ms=4, lw=1, label='y2')

#     idx = np.argwhere(np.diff(np.sign(y1 - y2)) != 0)

#     plt.plot(x[idx], y1[idx], 'ms', ms=7, label='Nearest data-point method')

#     # new method!
#     xc, yc = interpolated_intercept(x,y1,y2)
#     plt.plot(xc, yc, 'co', ms=5, label='Nearest data-point, with linear interpolation')


#     plt.legend(frameon=False, fontsize=10, numpoints=1, loc='lower left')

#     plt.savefig('curve crossing.png', dpi=200)
#     plt.show()


# if __name__ == '__main__': 
#     main()

alpha_prime = np.linspace(2.2, 6, 1000)
alpha_primeerr = 0.05

alpha_prime = 3.17
alpha_primeerr = 0.05

Beta = (2 * alpha_prime)/(alpha_prime - 2)
Beta_err = np.sqrt(((-4)/((alpha_prime - 2)**2))**2*(alpha_primeerr)**2)
alpha = Beta - 2
alpha_err = Beta_err


###############################################################################
# alpha'=3.17+-0.05, alpha=3.42+-0.15, from a bilby model Daniel ran
# alphas = np.asarray([5/3, 2, 8/3])
alphas = np.asarray([3.27, 3.42, 3.57])
# psis = np.asarray([0, 30, 60, 90, 120, 150, 180])
psis = np.asarray([0])
# ars = np.asarray([1.0, 1.5, 2, 2.5, 3, 5, 10])
ars = np.asarray([1.0])
# phasegrads = np.asarray([0, 0.25, 0.5, 0.75, 1])
phasegrads = np.asarray([0, 0.3])
# thetas = np.asarray([0, 30, 60, 90, 120, 150, 180])
thetas = np.asarray([0, 30, 60, 90, 120, 150, 180])
Taumax = 10
Dnuemax = 50
NF = 501
NT = 251
inpt_psi = 0
overwrite_acf = False
for i in range(0, len(alphas)):
    inpt_alpha = alphas[i]
    for ii in range(0, len(psis)):
        inpt_psi = psis[ii]
        for iii in range(0, len(thetas)):
            inpt_theta = thetas[iii]
            for iv in range(0, len(ars)):
                inpt_ar = ars[iv]
                for v in range(0, len(phasegrads)):
                    inpt_phasegrad = phasegrads[v]

                    if inpt_alpha == 2:
                        Theory_C1_measurement = 1  # From Lambert and Rickett 1999, we should derive our own as well
                    else:
                        Theory_C1_measurement = 0.654  # From Lambert and Rickett 1999, we should derive our own as well
                    
                    # This is grabbing the ACF from the simulated data using default inputs except
                    #    We have increased nf and nt and dnumax and taumax as well as ...
                    #    We are changing the psi, phasegrad, theta and ar as above.
                    wd = '/Users/jacobaskew/Desktop/C1_calcs/'
                    wd_Pickles = wd+'Pickles/'
                    wd_ACFs = wd+'ACFs/'
                    wd_Dnu = wd+'Dnu/'
                    wd_Tau = wd+'Tau/'
                    wd_C_1 = wd+'C_1/'
                    wd_Results = wd+'Results/'
                    name = 'psi'+str(inpt_psi)+'_AR'+str(inpt_ar)+'_alpha' + \
                            str(round(inpt_alpha, 2))+'_phasegrad'+str(inpt_phasegrad) + \
                            '_theta'+str(inpt_theta)+'_taumax'+str(Taumax)+'_dnumax' + \
                            str(Dnuemax)+'_nf'+str(NF)+'_nt'+str(NT)
                    wd_name = wd+str(name)+'/'
                    if os.path.exists(wd_name):
                        test = 1
                    else:
                        os.mkdir(wd_name)
                    acfilenm = wd_Pickles+name+'_acf.obj'
                    acfilenm2 = wd_Pickles+name
                    acfilenm3 = wd_name+name+'_acf.obj'
                    acfilenm4 = wd_name+name
                    if os.path.exists(acfilenm) and not overwrite_acf:
                        acf = pickle.load(open(acfilenm,'rb'))
                        acf.plot_acf(filename=acfilenm2)
                        acf.plot_acf(filename=acfilenm4)
                    else:
                        acf = ACF(V=1, psi=inpt_psi, phasegrad=inpt_phasegrad, theta=inpt_theta, ar=inpt_ar,
                              alpha=inpt_alpha, taumax=Taumax, dnumax=Dnuemax, nf=NF, nt=NT, amp=1,
                              wn=0, spatial_factor=2, resolution_factor=1, core_factor=2, cropped=False,
                              auto_sampling=False, plot=True, display=True, filename=acfilenm2)
                        acf.plot_acf(filename=acfilenm4, cropped=True)
                        file_acf = open(acfilenm, 'wb')
                        pickle.dump(acf, file_acf)
                        file_acf2 = open(acfilenm3, 'wb')
                        pickle.dump(acf, file_acf2)
                    
                    ###############################################################################
                    if inpt_phasegrad != 0:
                        nf = np.shape(acf.gammitv)[0] // 2
                    else:
                        nf = 0
                    # Here we are grabbing the timescale data which will later be Fourier transformed
                    # Fig.2 Lambert 1999 and eq.A2 Ricket 2014
                    pos_gammitv = acf.gammitv[nf, :]  # Load in the gamma before we apply np.conj
                    neg_gammitv = np.conj(np.flip(pos_gammitv))[:-1]
                    timescale_data = np.concatenate((neg_gammitv, pos_gammitv))
                    nt = len(timescale_data)
                    cw = np.hanning(np.floor(1*nt))
                    chan_window = np.insert(cw, int(np.ceil(len(cw)/2)),
                                            np.ones([nt-len(cw)]))
                    timescale_data = np.multiply(chan_window, timescale_data)
                    
                    # Determine the x-axis of the time lags (unitless) # AGAIN
                    freqshift = np.fft.fftshift(acf.fn)  # shift the frequency axis data
                    d = acf.fn[1] - acf.fn[0]
                    time_axis = np.fft.fftfreq(n=freqshift.shape[-1], d=d) # / (d*n) * freqshift.shape[-1]
                    time_axis = np.fft.fftshift(time_axis) * 2 * np.pi # shift the frequency axis data back
                    #
                    # Take the fft of gamma to transform the frequency data into time data
                    acf_fft_data = np.fft.fftshift(timescale_data)  # shift
                    acf_fft_data = np.fft.ifft(acf_fft_data)  # fft
                    acf_fft_data_real = np.fft.fftshift(acf_fft_data).real  # shift and extract real component
                    fft_norm = (acf_fft_data_real / np.max(acf_fft_data_real))  # normalise the data
                    # Define some datasets for plotting and fitting
                    tau_argmax = np.argmax(fft_norm)
                    tau_ydata = fft_norm[tau_argmax:]  # ydata timescale
                    tau_xdata = time_axis[tau_argmax:]  # xdata timescale
                    
                    tau_xdata -= tau_xdata[0]
                                        
                    data = acf.gammitv2
                    nmf = np.shape(data)[1] - 1  # mid point of frequency axis
                    nmt = np.shape(data)[1] // 2  # mid point of time axis
                    nu_xdata = acf.fn[nmf:]
                    if inpt_phasegrad != 0:
                        nu_ydata = data[nmt, :]
                        nu_ydata /= np.max(nu_ydata)
                    else:
                        nu_ydata = data[0, :]
                    if inpt_alpha != 5/3:
                        nu_ydata /= np.max(nu_ydata)
                        nu_xdata -= nu_xdata[np.argmax(nu_ydata)]
                        # xdata = acf.fn[nmf:]
                        # ydata = acf.tn[nmt:]
                        # zdata = acf.gammitv2
                        # ind1, ind2 = \
                        #     np.argwhere(acf.gammitv2 > 1)[
                        #         np.argmax(acf.gammitv2[np.argwhere(
                        #         acf.gammitv2 > 1)])-1]
                        # xdata_new = xdata - xdata[ind1]
                        # ydata_new = ydata - ydata[ind2]
                        # nu_xdata = xdata_new
                        # nu_ydata = data[0, :]
                    # Here we determine the value of C_1 by measuring the 1/e and 0.5 moment for timescale and frequnecy data respectively
                    nu_X = nu_xdata
                    nu_Y1 = nu_ydata
                    nu_Y2 = np.ones(np.shape(nu_ydata)) * 0.5
                    tau_X = tau_xdata
                    tau_Y1 = tau_ydata
                    tau_Y2 = np.ones(np.shape(tau_ydata)) * 1/np.e
                    dnu_measured, yc = interpolated_intercept(nu_X, nu_Y1, nu_Y2)
                    dnu_measured = float(dnu_measured)
                    tau_measured, yc = interpolated_intercept(tau_X, tau_Y1, tau_Y2)
                    tau_measured = float(tau_measured)
                    C1_measurement = dnu_measured * tau_measured

                    fig = plt.figure(figsize=(20, 10))
                    ax = fig.add_subplot(1, 1, 1)
                    plt.plot(nu_xdata, nu_ydata, c='C0', linewidth=4,
                             label=r'$ACF1D_{\nu}$, $\nu_d$=' + str(round(dnu_measured, 5)))
                    xl = (np.min(nu_xdata), np.max(nu_xdata))
                    plt.xlim(xl[0], xl[1])
                    plt.hlines(0.5, xl[0], xl[1], colors='k', linestyle='dotted', label='0.5')
                    plt.xlabel(r"$\nu$ (Normalised Frequency Lags)")
                    plt.ylabel(r"$|\Gamma (\sigma=0; \nu)|^2$")
                    plt.title("Figure 4 from Lambert & Rickett 1999")
                    ax.legend()
                    plt.xlim(0, 5)
                    plt.ylim(0, 1)
                    plt.savefig(wd_Dnu+name+"_nu_d.png")
                    plt.savefig(wd_name+"nu_d.png")
                    plt.show()
                    plt.close()
                    
                    fig = plt.figure(figsize=(20, 10))
                    ax = fig.add_subplot(1, 1, 1)
                    plt.plot(tau_xdata, tau_ydata, c='C1', linewidth=4,
                             label=r'$ACF1D_{\tau}$, $\tau_d$=' + str(round(tau_measured, 5)))
                    xl = (0, np.max(tau_xdata))
                    plt.xlim(xl[0], xl[1])
                    plt.hlines(1/np.e, xl[0], xl[1], colors='k', linestyle='dotted', label='1/e')
                    plt.xlabel(r"$\tau$ (Normalised Time Lags)")
                    plt.ylabel(r"$Q_D$norm($\tau$)")
                    plt.title("Figure 2 from Lambert & Rickett 1999")
                    ax.legend()
                    plt.xlim(0, 2)
                    plt.ylim(0, 1)
                    plt.savefig(wd_Tau+name+"_tau_d.png")
                    plt.savefig(wd_name+"tau_d.png")
                    plt.show()
                    plt.close()
                    
                    if Theory_C1_measurement == 0.654:
                        dnu_diff = dnu_measured - 0.957  # Kolmogorov 0.957
                        tau_diff = tau_measured - 0.683  # Kolmogorov 0.683
                        C1_diff = C1_measurement - (0.957 * 0.683)
                    else:    
                        dnu_diff = dnu_measured - 1  # Kolmogorov 0.957
                        tau_diff = tau_measured - 1  # Kolmogorov 0.683
                        C1_diff = C1_measurement - (1 * 1)
                    # Therefore the value of the C1 constant is . . .
                    print(r"Using the above ACF we determine that nu_d =", str(dnu_measured))
                    print(r"This is off the expected value by", str(round(dnu_diff, 3)))
                    print(r"Using the above ACF we determine that tau_d =", str(tau_measured))
                    print(r"This is off the expected value by", str(round(tau_diff, 3)))
                    print(r"Using the above ACF we determine that C_1 =", str(C1_measurement))
                    print(r"This is off the expected value by", str(C1_diff))
                    ###############################################################################
                    # C_u = 1.16  # From Cordes and Rickett 1998, we should derive our own as well
                    C_u = 0.741  # From Lambert and Rickett 1999, we should derive our own as well
                    s = 0.71 # From our results this may change
                    c_ms = 299792458  # speed of light in ms^-1
                    pc_km = 3.08567758128e19
                    GHz_Hz = 1e9
                    km_m = 1e3
                    MHz_Hz = 1e6
                    
                    W_C = np.sqrt(C_u/C1_measurement)
                                        
                    unit_conversion = (np.sqrt(pc_km*MHz_Hz)/(GHz_Hz)) / km_m
                    
                    A_ISS_kms = np.sqrt((c_ms)/(4 * np.pi * C1_measurement)) * 1 * 1 * unit_conversion
                    
                    # Collecting a result.csv file
                    ResultsFileLoc = wd_Results+name+"_Results.csv"
                    ResultsFileLoc2 = wd_name+"Results.csv"
                    
                    ResultsFile = np.array(["psi", inpt_psi, "theta",
                                            inpt_theta, "ar", inpt_ar,
                                            "phasegrad", inpt_phasegrad,
                                            "alpha", inpt_alpha, "tau_d",
                                            tau_measured, "nu_d", dnu_measured,
                                            "C_1", C1_measurement, "W_C", W_C,
                                            "A_ISS", A_ISS_kms])
                    
                    np.savetxt(ResultsFileLoc, ResultsFile, delimiter=',', fmt='%s')
                    np.savetxt(ResultsFileLoc2, ResultsFile, delimiter=',', fmt='%s')
                    ###############################################################################
                    W_D = 1
                    W_C = 1
                    
                    unit_conversion = (np.sqrt(pc_km*MHz_Hz)/(GHz_Hz)) / km_m
                    
                    A_ISS_kms = np.sqrt((c_ms)/(4 * np.pi * C1_measurement)) * W_D * W_C * unit_conversion
                    if C1_measurement < 0.25:
                        C1_range = np.linspace(C1_measurement*0.9, 2, 1000)   
                    elif C1_measurement > 2:
                        C1_range = np.linspace(0.25, C1_measurement*1.1, 1000)   
                    else:
                        C1_range = np.linspace(0.25, 2, 1000)
                    W_C_range = np.sqrt(C_u/C1_range)
                    W_C_range = np.ones(np.shape(W_C_range))
                    A_ISS_range = (np.sqrt((c_ms)/(4 * np.pi * C1_range)) * W_D * W_C_range * unit_conversion) / 1e4
                    
                    Theory_W_C = np.sqrt(C_u/Theory_C1_measurement)
                    Theory_W_C = 1
                    Theory_A_ISS_kms = np.sqrt((c_ms)/(4 * np.pi * Theory_C1_measurement)) * W_D * Theory_W_C * unit_conversion
                    
                    fig = plt.figure(figsize=(20, 10))
                    ax = fig.add_subplot(1, 1, 1)
                    plt.plot(C1_range, A_ISS_range, c='C0')
                    xl = (np.min(C1_range), np.max(C1_range))
                    yl = plt.ylim()
                    plt.xlim(xl[0], xl[1])
                    plt.ylim(yl[0], yl[1])
                    plt.vlines(C1_measurement, yl[0], yl[1], colors='C1', linestyle='dashed', label='Measurement $C_1$=' + str(round(C1_measurement, 3)))
                    plt.hlines(A_ISS_kms/1e4, xl[0], xl[1], colors='C1', linestyle='dashed', label=r'Measurement $A_{ISS}$=' + str(round(A_ISS_kms/1e4, 2)) + ' x $10^{4}$ $km\,s^{-1}$')
                    plt.vlines(Theory_C1_measurement, yl[0], yl[1], colors='C3', linestyle='dashed', label='Kolmogorov Thin Screen $C_1$=' + str(round(Theory_C1_measurement, 3)))
                    plt.hlines(Theory_A_ISS_kms/1e4, xl[0], xl[1], colors='C3', linestyle='dashed', label=r'Kolmogorov Thin Screen $A_{ISS}$=' + str(round(Theory_A_ISS_kms/1e4, 2)) + ' x $10^{4}$ $km\,s^{-1}$')
                    plt.xlabel(r"$C_1$")
                    plt.ylabel(r"$A_{ISS}$ ($km\,s^{-1}$)($10^{4}$)")
                    ax.legend()
                    plt.savefig(wd_C_1+name+"_C_1.png")
                    plt.savefig(wd_name+"C_1.png")
                    plt.show()
                    plt.close()
datadir = '/Users/jacobaskew/Desktop/C1_calcs/Results/'
if os.path.exists(datadir + 'All.csv'):
    os.remove(datadir + 'All.csv')
csvs = glob.glob(datadir+'*.csv')
tau_d = []
nu_d = []
C_1 = []
W_C = []
A_ISS = []
psi = []
theta = []
alpha = []
ar = []
phasegrad = []
for i in range(0, len(csvs)):
    csvs_data = np.asarray(pd.read_csv(csvs[i]))
    psi.append(float(csvs_data[0]))
    theta.append(float(csvs_data[2]))
    ar.append(float(csvs_data[4]))
    phasegrad.append(float(csvs_data[6]))
    alpha.append(float(csvs_data[8]))
    tau_d.append(float(csvs_data[10]))
    nu_d.append(float(csvs_data[12]))
    C_1.append(float(csvs_data[14]))
    W_C.append(float(csvs_data[16]))
    A_ISS.append(float(csvs_data[18]))
dataframe = pd.DataFrame(np.transpose([psi, theta, ar, phasegrad, alpha, tau_d, nu_d, C_1, W_C, A_ISS]), columns=['psi', 'theta', 'ar', 'phasegrad', 'alpha', 'tau_d', 'nu_d', 'C_1', 'W_C', 'A_ISS'])

from natsort import natsort_keygen

dataframe = dataframe.sort_values(by=["psi", "theta"], key=natsort_keygen())
pd.DataFrame(dataframe).to_csv(datadir + 'All.csv', index=False)

###############################################################################
# alpha=3.5, from a bilby model I ran
# Depending on alpha
# Theory_C1_measurement = 0.957  # From Cordes and Rickett 1998, we should derive our own as well
# Taumax = 10
# Dnuemax = 50
# NF = 151
# NT = 151
# inpt_alpha = 5/3
# inpt_ar = 1
# inpt_psi = 0
# inpt_theta = 0
# inpt_phasegrad = 0
# if inpt_alpha == 2:
#     Theory_C1_measurement = 1  # From Lambert and Rickett 1999, we should derive our own as well
# else:
#     Theory_C1_measurement = 0.654  # From Lambert and Rickett 1999, we should derive our own as well

# # This is grabbing the ACF from the simulated data using default inputs except
# #    We have increased nf and nt and dnumax and taumax as well as ...
# #    We are changing the psi, phasegrad, theta and ar as above.
# wd = '/Users/jacobaskew/Desktop/C1_calcs/'
# wd1 = wd+'psi'+str(inpt_psi)+'_AR'+str(inpt_ar)+'_alpha' + \
#         str(round(inpt_alpha, 2))+'_phasegrad'+str(inpt_phasegrad) + \
#         '_theta'+str(inpt_theta)+'_taumax'+str(Taumax)+'_dnumax' + \
#         str(Dnuemax)+'_nf'+str(NF)+'_nt'+str(NT)
# acfilenm = wd1+'_acf.obj'
# if os.path.exists(acfilenm):
#     acf = pickle.load(open(acfilenm,'rb'))
#     acf.plot_acf(filename=wd1)
# else:
#     acf = ACF(V=1, psi=inpt_psi, phasegrad=inpt_phasegrad, theta=inpt_theta, ar=inpt_ar,
#           alpha=inpt_alpha, taumax=Taumax, dnumax=Dnuemax, nf=NF, nt=NT, amp=1,
#           wn=0, spatial_factor=2, resolution_factor=1, core_factor=2,
#           auto_sampling=False, plot=True, display=True, filename=wd1)
#     file_acf = open(acfilenm, 'wb')
#     pickle.dump(acf, file_acf)

# ###############################################################################
# if inpt_phasegrad != 0:
#     nf = np.shape(acf.gammitv)[0] // 2
# else:
#     nf = 0
# # Here we are grabbing the timescale data which will later be Fourier transformed
# # Fig.2 Lambert 1999 and eq.A2 Ricket 2014
# pos_gammitv = acf.gammitv[nf, :]  # Load in the gamma before we apply np.conj
# neg_gammitv = np.conj(np.flip(pos_gammitv))[:-1]
# timescale_data = np.concatenate((neg_gammitv, pos_gammitv))
# nt = len(timescale_data)
# cw = np.hanning(np.floor(1*nt))
# chan_window = np.insert(cw, int(np.ceil(len(cw)/2)),
#                         np.ones([nt-len(cw)]))
# timescale_data = np.multiply(chan_window, timescale_data)

# # Determine the x-axis of the time lags (unitless) # AGAIN
# freqshift = np.fft.fftshift(acf.fn)  # shift the frequency axis data
# d = acf.fn[1] - acf.fn[0]
# time_axis = np.fft.fftfreq(n=freqshift.shape[-1], d=d) # / (d*n) * freqshift.shape[-1]
# time_axis = np.fft.fftshift(time_axis) * 2 * np.pi # shift the frequency axis data back
# #
# # Take the fft of gamma to transform the frequency data into time data
# acf_fft_data = np.fft.fftshift(timescale_data)  # shift
# acf_fft_data = np.fft.ifft(acf_fft_data)  # fft
# acf_fft_data_real = np.fft.fftshift(acf_fft_data).real  # shift and extract real component
# fft_norm = (acf_fft_data_real / np.max(acf_fft_data_real))  # normalise the data
# # Define some datasets for plotting and fitting
# tau_argmax = np.argmax(fft_norm)
# tau_ydata = fft_norm[tau_argmax:]  # ydata timescale
# tau_xdata = time_axis[tau_argmax:]  # xdata timescale

# tau_xdata = tau_xdata - tau_xdata[0]

# # Here we are grabbing the frequency slice of the real gamma Fig.4 Lambert 1999 and eq.A2 Ricket 2014
# nr, nc = np.shape(acf.gammitv2)
# gam2 = np.zeros((nr, nc*2-1))
# gam2[:, 0:nc-1] = np.fliplr(acf.gammitv2[:, 1:])
# gam2[:, nc-1:] = acf.gammitv2
# gam2 = gam2.squeeze()
# # Build full ACF
# gam3 = np.zeros((nr*2-1, nc*2-1))
# gam3[0:nr-1, :] = np.flipud(gam2[1:, :])
# gam3[nr-1:, :] = gam2
# gam3 = np.transpose(gam3)
# # Determine the mid point in the dataset
# midind = np.shape(acf.gammitv2)[1]
# nuind = np.shape(gam3)[0] // 2
# tauind = np.shape(gam3)[1] // 2
# # Only consider the positive lags of the ACF I field
# frequency_data = gam3[nuind:, tauind]
# nu_ydata = frequency_data  # ydata frequency
# nu_xdata = acf.fn[nuind:]  # xdata frequency

# if inpt_phasegrad != 0:
#     nt2 = np.shape(acf.gammitv2)[0] // 2
#     nu_ydata = acf.gammitv2[nt2, :]
#     nu_xdata = acf.fn[nt2:]
# else:
#     nt2 = np.shape(acf.gammitv2)[0] - 1
#     nu_ydata = acf.gammitv2[0, :]
#     nu_xdata = acf.fn[nt2:]    

# # Here we determine the value of C_1 by measuring the 1/e and 0.5 moment for timescale and frequnecy data respectively
# freq_interpolate = interpolate.interp1d(nu_xdata, nu_ydata)
# time_interpolate = interpolate.interp1d(tau_xdata, tau_ydata)
# dnu_finex = np.linspace(np.min(nu_xdata), np.max(nu_xdata), 10000)
# dnu_finey = freq_interpolate(dnu_finex)
# tau_finex = np.linspace(np.min(tau_xdata), np.max(tau_xdata), 10000)
# tau_finey = time_interpolate(tau_finex)
# dnu_measured = dnu_finex[np.argmin(abs(0.5 - dnu_finey))]
# tau_measured = abs(tau_finex[np.argmin(abs(1/np.e - tau_finey))])
# C1_measurement = dnu_measured * tau_measured

# fig = plt.figure(figsize=(20, 10))
# ax = fig.add_subplot(1, 1, 1)
# plt.plot(nu_xdata, nu_ydata, c='C0', linewidth=4,
#          label=r'$ACF1D_{\nu}$, $\nu_d$=' + str(round(dnu_measured, 5)))
# xl = (np.min(nu_xdata), np.max(nu_xdata))
# plt.xlim(xl[0], xl[1])
# plt.hlines(0.5, xl[0], xl[1], colors='k', linestyle='dotted', label='0.5')
# plt.xlabel(r"$\nu$ (Normalised Frequency Lags)")
# plt.ylabel(r"$|\Gamma (\sigma=0; \nu)|^2$")
# plt.title("Figure 4 from Lambert & Rickett 1999")
# ax.legend()
# plt.xlim(0, 5)
# plt.ylim(0, 1)
# plt.savefig(wd1+"_nu_d.png")
# plt.show()
# plt.close()

# fig = plt.figure(figsize=(20, 10))
# ax = fig.add_subplot(1, 1, 1)
# plt.plot(tau_xdata, tau_ydata, c='C1', linewidth=4,
#          label=r'$ACF1D_{\tau}$, $\tau_d$=' + str(round(tau_measured, 5)))
# xl = (0, np.max(tau_xdata))
# plt.xlim(xl[0], xl[1])
# plt.hlines(1/np.e, xl[0], xl[1], colors='k', linestyle='dotted', label='1/e')
# plt.xlabel(r"$\tau$ (Normalised Time Lags)")
# plt.ylabel(r"$Q_D$norm($\tau$)")
# plt.title("Figure 2 from Lambert & Rickett 1999")
# ax.legend()
# plt.xlim(0, 2)
# plt.ylim(0, 1)
# plt.savefig(wd1+"_tau_d.png")
# plt.show()
# plt.close()

# if Theory_C1_measurement == 0.654:
#     dnu_diff = dnu_measured - 0.957  # Kolmogorov 0.957
#     tau_diff = tau_measured - 0.683  # Kolmogorov 0.683
#     C1_diff = C1_measurement - (0.957 * 0.683)
# else:    
#     dnu_diff = dnu_measured - 1  # Kolmogorov 0.957
#     tau_diff = tau_measured - 1  # Kolmogorov 0.683
#     C1_diff = C1_measurement - (1 * 1)
# # Therefore the value of the C1 constant is . . .
# print(r"Using the above ACF we determine that nu_d =", str(dnu_measured))
# print(r"This is off the expected value by", str(round(dnu_diff, 3)))
# print(r"Using the above ACF we determine that tau_d =", str(tau_measured))
# print(r"This is off the expected value by", str(round(tau_diff, 3)))
# print(r"Using the above ACF we determine that C_1 =", str(C1_measurement))
# print(r"This is off the expected value by", str(C1_diff))
# ###############################################################################
# # C_u = 1.16  # From Cordes and Rickett 1998, we should derive our own as well
# C_u = 0.741  # From Lambert and Rickett 1999, we should derive our own as well
# s = 0.71 # From our results this may change
# c_ms = 299792458  # speed of light in ms^-1
# pc_km = 3.08567758128e19
# GHz_Hz = 1e9
# km_m = 1e3
# MHz_Hz = 1e6

# W_C = np.sqrt(C_u/C1_measurement)

# W_D = np.sqrt((2*(1-s))/(s))

# unit_conversion = (np.sqrt(pc_km*MHz_Hz)/(GHz_Hz)) / km_m

# A_ISS_kms = np.sqrt((c_ms)/(4 * np.pi * C1_measurement)) * 1 * 1 * unit_conversion

# # Collecting a result.csv file
# ResultsFileLoc = wd1+"_Results.csv"

# ResultsFile = np.array(["tau_d", tau_measured, "nu_d", dnu_measured, "C_1",
#                         C1_measurement, "W_C", W_C, "W_D", W_D, "A_ISS",
#                         A_ISS_kms])

# np.savetxt(ResultsFileLoc, ResultsFile, delimiter=',', fmt='%s')
# ###############################################################################
# W_D = 1
# W_C = 1

# unit_conversion = (np.sqrt(pc_km*MHz_Hz)/(GHz_Hz)) / km_m

# A_ISS_kms = np.sqrt((c_ms)/(4 * np.pi * C1_measurement)) * W_D * W_C * unit_conversion
# if C1_measurement < 0.25:
#     C1_range = np.linspace(C1_measurement*0.9, 2, 1000)   
# elif C1_measurement > 2:
#     C1_range = np.linspace(0.25, C1_measurement*1.1, 1000)   
# else:
#     C1_range = np.linspace(0.25, 2, 1000)
# W_C_range = np.sqrt(C_u/C1_range)
# W_C_range = np.ones(np.shape(W_C_range))
# A_ISS_range = (np.sqrt((c_ms)/(4 * np.pi * C1_range)) * W_D * W_C_range * unit_conversion) / 1e4

# Theory_W_C = np.sqrt(C_u/Theory_C1_measurement)
# Theory_W_C = 1
# Theory_A_ISS_kms = np.sqrt((c_ms)/(4 * np.pi * Theory_C1_measurement)) * W_D * Theory_W_C * unit_conversion

# fig = plt.figure(figsize=(20, 10))
# ax = fig.add_subplot(1, 1, 1)
# plt.plot(C1_range, A_ISS_range, c='C0')
# xl = (np.min(C1_range), np.max(C1_range))
# yl = plt.ylim()
# plt.xlim(xl[0], xl[1])
# plt.ylim(yl[0], yl[1])
# plt.vlines(C1_measurement, yl[0], yl[1], colors='C1', linestyle='dashed', label='Measurement $C_1$=' + str(round(C1_measurement, 3)))
# plt.hlines(A_ISS_kms/1e4, xl[0], xl[1], colors='C1', linestyle='dashed', label=r'Measurement $A_{ISS}$=' + str(round(A_ISS_kms/1e4, 2)) + ' x $10^{4}$ $km\,s^{-1}$')
# plt.vlines(Theory_C1_measurement, yl[0], yl[1], colors='C3', linestyle='dashed', label='Kolmogorov Thin Screen $C_1$=' + str(round(Theory_C1_measurement, 3)))
# plt.hlines(Theory_A_ISS_kms/1e4, xl[0], xl[1], colors='C3', linestyle='dashed', label=r'Kolmogorov Thin Screen $A_{ISS}$=' + str(round(Theory_A_ISS_kms/1e4, 2)) + ' x $10^{4}$ $km\,s^{-1}$')
# plt.xlabel(r"$C_1$")
# plt.ylabel(r"$A_{ISS}$ ($km\,s^{-1}$)($10^{4}$)")
# ax.legend()
# plt.savefig(wd1+"_C_1.png")
# plt.show()
# plt.close()
# ###############################################################################
# # Determining the Ar of our data
# R_xdata = np.linspace(0, 1, 1000)
# A_R_ydata = np.sqrt(abs(-R_xdata - 1)) / np.sqrt(abs(R_xdata - 1))
# # A_R_ydata = np.sqrt((R_xdata + 1) / (1 - R_xdata))

# R_modeldata = np.random.normal(loc=0.3, scale=0.14, size=1000)
# A_R_modeldata = np.sqrt(abs(-R_modeldata - 1)) / np.sqrt(abs(R_modeldata - 1))
# A_R_modeldata_std = np.std(A_R_modeldata)
# A_R_modeldata_median = np.median(A_R_modeldata)

# fig = plt.figure(figsize=(20, 10))
# ax = fig.add_subplot(1, 1, 1)
# plt.plot(R_xdata, A_R_ydata, c='C0')
# xl = (np.min(R_xdata), np.max(R_xdata))
# plt.xlim(xl[0], xl[1])
# plt.xlabel(r"$R$ (unitless)")
# plt.ylabel(r"$A_R$ (unitless)")
# plt.title(r"'Real' Values for $A_R$ given $R$")
# plt.show()
# plt.close()

# fig = plt.figure(figsize=(20, 10))
# ax = fig.add_subplot(1, 1, 1)
# plt.hist(A_R_modeldata, bins=25, color='C0')
# yl = plt.ylim()
# plt.ylim(yl[0], yl[1])
# plt.vlines(A_R_modeldata_median, yl[0], yl[1], alpha=0.5, colors='C3', label=r'$A_R$ = ' + str(round(A_R_modeldata_median, 2)) + ' $\pm$ ' + str(round(A_R_modeldata_std, 2)))
# plt.vlines(A_R_modeldata_median - A_R_modeldata_std, yl[0], yl[1], linestyle='dashed', alpha=0.5, colors='C3')
# plt.vlines(A_R_modeldata_median + A_R_modeldata_std, yl[0], yl[1], linestyle='dashed', alpha=0.5, colors='C3')
# ax.legend()
# plt.xlabel(r"$A_R$ (unitless)")
# plt.ylabel("Frequency")
# plt.title(r"$A_R$ given $R$=0.3$\pm$0.14 from Rickett et al. 2014")
# plt.show()
# plt.close()
# ###############################################################################