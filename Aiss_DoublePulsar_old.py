#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 15:22:35 2023

@author: jacobaskew
"""

# from scintools.scint_utils import read_par, pars_to_params

# desktopdir = '/Users/jacobaskew/Desktop/'
# datadir = desktopdir + 'DoublePulsar_Project/datasets/FullDataStorage/'
# par_dir = '/Users/jacobaskew/Desktop/Swinburne_Daniel/ParFiles/'
# psrname = 'J0737-3039A'
# wd0 = "/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/New/"
# outdir = wd0 + "Modelling"
# pars = read_par(str(par_dir) + str(psrname) + '.par')
# params = pars_to_params(pars)

# viss = np.loadtxt(datadir + 'Full_VissData.txt', dtype='float')
# visserr = np.loadtxt(datadir + 'Full_VisserrData.txt', dtype='float')
# mjd = np.loadtxt(datadir + 'Full_MJDData.txt', dtype='float')
# freqMHz = np.loadtxt(datadir + 'Full_FreqData.txt', dtype='float')
# freqGHz = freqMHz / 1e3
# dnu = np.loadtxt(datadir + 'Full_DnuData.txt', dtype='float')
# dnuerr = np.loadtxt(datadir + 'Full_DnuerrData.txt', dtype='float')
# tau = np.loadtxt(datadir + 'Full_TauData.txt', dtype='float')
# tauerr = np.loadtxt(datadir + 'Full_TauerrData.txt', dtype='float')
# phase = np.loadtxt(datadir + 'Full_PhaseData.txt', dtype='float')
# U = np.loadtxt(datadir + 'Full_UData.txt', dtype='float')
# ve_ra = np.loadtxt(datadir + 'Full_ve_raData.txt', dtype='float')
# ve_dec = np.loadtxt(datadir + 'Full_ve_decData.txt', dtype='float')

# kwargs = {"U": U, "ve_ra": ve_ra, "ve_dec": ve_dec, "params": params}

###############################################################################
# A simple simulation ACF
# nu_c = 1000
# nu_d = 0.08897998108076888
# mb2 = 0.773 * (nu_c / nu_d) ** (5/6)
# s = 0.7
# Dpsr = 735
# z = Dpsr * (1-s) * 3.08567758128e16
# c = 299792458
# f = 1e9
# wavelength = c/f
# k = (2*np.pi)/wavelength
# rf = np.sqrt(z/k)

# sim = Simulation(mb2=150, rf=0.5, ar=2.5, psi=94, freq=815.7343375, dt=8,
#                  ns=1024, nf=16384, dlam=0.033203125)
# dyn = Dynspec(dyn=sim, process=False)
# dyn.plot_dyn()
# dyn.calc_acf()
# dyn.plot_acf(crop=True)

###############################################################################
# s = 0.7
# kappa = 1
# C = 0.957
# Cu = 1.16
# C2 = 1
# c = 299792458  # m/s
# # c = 3e8  # m/s
# D = 0.735
# D_err = 0.06

# A_iss_Kol = (np.sqrt((c)/(4*np.pi*C)) * (np.sqrt(3.086e19*1e6)/(1e9))) / 1e3
# A_iss_2 = (np.sqrt((c)/(4*np.pi*C2)) * (np.sqrt(3.086e19*1e6)/(1e9))) / 1e3
# A_iss_u = (np.sqrt((c)/(4*np.pi*Cu)) * (np.sqrt(3.086e19*1e6)/(1e9))) / 1e3
# A_iss_CR2 = 2.72e4

# coeff_Kol = A_iss_Kol * np.sqrt((2*(1-s))/(s)) * np.sqrt(1.16/C)
# coeff_2 = A_iss_2 * np.sqrt((2*(1-s))/(s)) * np.sqrt(1.16/C2)
# coeff_u = A_iss_u * np.sqrt((2*(1-s))/(s)) * np.sqrt(1.16/Cu)
# coeff_CR2 = A_iss_CR2 * np.sqrt((2*(1-s))/(s)) * np.sqrt(1.16/C2)

# viss_Kol = coeff_Kol * (np.sqrt(D*dnu))/(freqGHz*tau)
# viss_Kol_err = viss_Kol * np.sqrt((D_err/(2*D))**2+(dnuerr/(2*dnu))**2 +
#                                   (-tauerr/tau)**2)
# viss_2 = coeff_2 * (np.sqrt(D*dnu))/(freqGHz*tau)
# viss_2_err = viss_2 * np.sqrt((D_err/(2*D))**2+(dnuerr/(2*dnu))**2 +
#                               (-tauerr/tau)**2)
# viss_u = coeff_u * (np.sqrt(D*dnu))/(freqGHz*tau)
# viss_u_err = viss_u * np.sqrt((D_err/(2*D))**2+(dnuerr/(2*dnu))**2 +
#                               (-tauerr/tau)**2)
# viss_CR2 = coeff_CR2 * (np.sqrt(D*dnu))/(freqGHz*tau)
# viss_CR2_err = viss_CR2 * np.sqrt((D_err/(2*D))**2+(dnuerr/(2*dnu))**2 +
#                                   (-tauerr/tau)**2)
###############################################################################
# A plot of possible axial ratios

# Input frequency data
# Sampling rate (samples per second)
# sampling_rate = 801  # Replace with your actual sampling rate
# # Perform FFT
# fft_data = np.fft.fftshift(timescale_data)
# fft_data = np.fft.fft(fft_data)
# fft_data = np.fft.fftshift(fft_data).real[midind:]
# fft_norm = fft_data / np.max(fft_data)
# # Determine the corresponding frequency axis
# frequency_axis = np.fft.fftfreq(len(frequency_data)) * sampling_rate
# # Determine the corresponding time axis
# time_axis = (np.arange(len(frequency_data)) / sampling_rate)[midind:]

# Plot the transformed data
# plt.plot(time_axis, fft_norm)
# plt.xlabel('Time (s)')
# plt.ylabel('Amplitude')
# plt.show()

# time_axis = np.fft.fftfreq(freqshift.shape[-1]) * (1 / np.max(acf.fn)) * freqshift.shape[-1]
# time_axis = np.fft.fftshift(time_axis) * np.pi # shift the frequency axis data back
# mindex_dnu = np.argmin(abs(0.5 - nu_ydata))
# mindex_tau = np.argmin(abs(1/np.e - tau_ydata))

# tau_finex = np.linspace(tau_xdata[mindex_tau-1], tau_xdata[mindex_tau+2], int(1e6))
# dnu_finex = np.linspace(nu_xdata[mindex_dnu-1], nu_xdata[mindex_dnu+2], int(1e6))
# tau_finey = np.linspace(tau_ydata[mindex_tau-1], tau_ydata[mindex_tau+2], int(1e6))
# dnu_finey = np.linspace(nu_ydata[mindex_dnu-1], nu_ydata[mindex_dnu+2], int(1e6))
# This section was added to improve the measurement that already worked a bit
# Determine the x-axis of the time lags (unitless)
# freqshift = np.fft.fftshift(acf.fn)  # shift the frequency axis data
# d = acf.fn[1] - acf.fn[0]
# time_axis = np.fft.fftfreq(n=freqshift.shape[-1], d=d) # / (d*n) * freqshift.shape[-1]
# time_axis = np.fft.fftshift(time_axis) * 2 * np.pi # shift the frequency axis data back
# #


# def exponential_func(xdata, a, b, c):
#     return a * np.exp(-b * xdata) + c


# x_curve = time_axis[np.argwhere(time_axis >= 0)].flatten()
# params, _ = curve_fit(f=exponential_func, xdata=x_curve, ydata=pos_gammitv)

# a_fit, b_fit, c_fit = params
# y_curve = exponential_func(x_curve, a_fit, b_fit, c_fit)

# x_pos = np.linspace(0, 100000, 100000)
# y_pos = exponential_func(x_pos, a_fit, b_fit, c_fit)
# y_data = np.concatenate((np.flip(y_pos), y_pos))
# x_data = np.concatenate((-np.flip(x_pos), x_pos))
# nt = len(timescale_data)
# nt2 = nt // 2
# lydata = len(y_data)
# yd2 = lydata // 2

# timescale_data = np.concatenate((timescale_data, y_data[yd2+nt:]))
# timescale_data = np.concatenate((y_data[:yd2-nt], timescale_data))

#
# tau_ydata = tau_ydata / tau_ydata[0]


# Code with help from Atharva

# # acf_fft_data = np.fft.fft(timescale_data)  # fft
# acf_fft_data = np.fft.fft(pos_gammitv)  # fft
# acf_fft_data = np.fft.fftshift(acf_fft_data).real  # shift
# # fft_norm = (acf_fft_data / np.max(acf_fft_data))  # normalise the data
# fft_norm = np.flip((acf_fft_data / np.max(acf_fft_data)))  # normalise the data
# d = acf.fn[1] - acf.fn[0]
# # time_axis = np.fft.fftfreq(n=acf.fn.shape[-1], d=d) # / (d*n) * freqshift.shape[-1]
# time_axis = np.fft.fftfreq(n=acf.fn[len(acf.fn)//2:].shape[-1], d=d) # / (d*n) * freqshift.shape[-1]
# time_axis = np.fft.fftshift(time_axis) * 2 * np.pi # shift the frequency axis data back

# tau_argmax = np.argmax(fft_norm)
# tau_ydata = fft_norm[tau_argmax:]  # ydata timescale
# # tau_xdata = time_axis[tau_argmax-1:][:-1]  # xdata timescale
# tau_xdata = time_axis[tau_argmax:]  # xdata timescale


# gammitv = acf.gammitv
# Build first half
# nr, nc = np.shape(gammitv)
# gam2 = np.zeros((nr, nc*2-1))
# gam2[:, 0:nc-1] = np.fliplr(gammitv[:, 1:])
# gam2[:, nc-1:] = gammitv
# gam2 = gam2.squeeze()

# # Build full ACF
# gam3 = np.zeros((nr*2-1, nc*2-1))
# gam3[0:nr-1, :] = np.flipud(gam2[1:, :])
# gam3[nr-1:, :] = gam2
# gam3 = np.transpose(gam3)



# Step 0: we need to make sure we don't ignore complex values
# Step 1: take a slice of the ACF and get pos and neg values
# Step 2: use that for timescale data!


# plt.contourf(data)
# plt.title("gammitv")
# plt.show()
# plt.close()

# data = acf.acf
# data = acf.acf
# data = acf.acf_efield

# plt.contourf(acf.acf_efield)
# plt.title("E-Field")
# plt.show()
# plt.close()


# plt.contourf(acf.gammes)
# plt.title("Gammes")
# plt.show()
# plt.close()


# plt.contourf(acf.gammes2)
# plt.title("Gammes2")
# plt.show()
# plt.close()


# plt.contourf(frequency_data)
# plt.title("gammitv 2")
# plt.show()
# plt.close()

# plt.contourf(timescale_data)
# plt.title("ACF")
# plt.show()
# plt.close()

# mid_tau = np.shape(frequency_data)[1] // 2


# tau_ydata = acf_fft_data[mid_nu, mid_tau:] / np.max(acf_fft_data[mid_nu, mid_tau:])
# tau = acf.tn[mid_nu:]

# NOTE: Our theory Aiss is 0.4% different when compared to Rickett and Cordes 1998
###############################################################################
# Creating a bunch of ACFs for a range of phase gradients
# psi_range = np.asarray([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120,
#                         130, 140, 150, 160, 170, 180])
# Ar_range = np.asarray([1.1, 1.3, 1.5, 1.7, 1.9, 2, 3, 4, 5, 10])
# for i in range(0, len(psi_range)):
#     for ii in range(0, len(Ar_range)):        
#         acf = ACF(V=1, psi=psi_range[i], phasegrad=0, theta=0, ar=Ar_range[ii],
#                   alpha=5/3, taumax=4, dnumax=4, nf=51, nt=51, amp=1, wn=0,
#                   spatial_factor=2, resolution_factor=1, core_factor=2,
#                   auto_sampling=True, plot=True, display=True)

###############################################################################
## CALCULATING C1 ##

# c_ms = 299792458  # speed of light in ms^-1
# freq_Hz = 1e9  # 1 GHz in Hz, observational center frequency
# s = 0.72  # relative distance to the scattering screen
# L_m = 3.086e16 * (735)  # Distance to the source, Double Pulsar, in meters
# Z0 = L_m * (1-s)  # distance to the screen from the observer
# f_MHz = freq_Hz / 1e6  # Reference frequency in MHz
# wavelength = c_ms/freq_Hz # Reference wavelength in m
# k_m = (2 * np.pi * freq_Hz) / c_ms  # wavenumber in m^-1
# rf = np.sqrt(Z0/k_m)  # fresnel scale
# dnu_c = 1.1e5  # Estimated scintillation bandwidth in Hz ## CHECK ##
# dnu_c_MHz = 0.11  # Estimated scintillation bandwidth in MHz
# s0 = np.sqrt(dnu_c/freq_Hz) * rf  # field coherence length  ## CHECK ##
# s0 = 3.8e6  # field coherence length in meters  ## CHECK ##
# Nu_d = 0.957  # Taking from table, normalised decorrelation bandwidth  ## CHECK ##
# Zp = L_m - Z0  # distance to the pulsar from the screen
# Z_scatt = Z0 * Zp / L_m  # the scattered path length
# DeltaNu_d = (2 * np.pi * Nu_d * freq_Hz**2 * s0**2) / (Z_scatt * c_ms)  # diffractive scintillation bandwidth
# DeltaNu_d_MHz = DeltaNu_d / 1e6  # diffractive scintillation bandwidth in MHz
# alpha = 5/3  # Kolmogorov alpha where Beta = 11/3
# alpha2 = 2  # square law structure function alpha where Beta = 4
# z_m = L_m - (L_m*0.7)  # distance to screen from observer assuming s=0.7, in meters
# t_s = np.linspace(0, 3e-5, 1001)  # time steps in units of seconds
# tau_s = (c_ms * s0**2 * k_m**2 * t_s) / (L_m)  # The time delay due to pulse broadening, in seconds
# n = np.arange(1, 1002)  # n an integer for infinite sums ...
# ###############################################################################
# # Making some plots ...


# fig = plt.figure(figsize=(20, 10))
# ax = fig.add_subplot(1, 1, 1)
# sumnum48 = []
# for i in range(0, len(tau_s)):
#     sumnum48.append(np.sum((-1)**(n-1) * tau_s[i]**2 *
#                            np.exp(-(n * np.pi)**2/(3) * tau_s[i])))
# sumnum48 = np.asarray(sumnum48)
# Q_SD_tau_s = (4 / 3) * np.pi**3 * (c_ms * s0**2 * k_m**2)/(L_m) * sumnum48  # Equation 48
# Q_SD_tau_s_norm = Q_SD_tau_s/np.max(Q_SD_tau_s)
# plt.plot(tau_s, Q_SD_tau_s_norm, label='Kolmogorov')
# xl = plt.xlim()
# plt.hlines(1/np.e, xl[0], xl[1], colors='k', linestyle='dotted', label='1/e')
# plt.xlim(xl[0], xl[1])
# plt.xlabel(r"$\tau_s$ (s)")
# plt.ylabel(r"$Q_{SD}^{norm}\,(\tau_s)$")
# plt.title("An attempted copy of Figure 1 from Lambert & Rickett 1999")
# ax.legend()
# plt.savefig("/Users/jacobaskew/Desktop/Figure1Tau_d.png")
# plt.show()
# plt.close()

# tau_s_measurement = tau_s[np.argmin(abs((Q_SD_tau_s_norm) - (1/np.e)))]
# C1_measurement = 2 * np.pi * tau_s_measurement * DeltaNu_d
# print("The estimated scattering timescale from equation 48 is " +
#       str(round(tau_s_measurement, 2)) + "s")
# print("The estimated diffractive scintillation bandwidth from equation 67 is " +
#       str(round(DeltaNu_d_MHz, 2)) + "MHz")
# print("Therefore the estimated value for C1 using equation 68 is " +
#       str(round(C1_measurement, 5)))

# t_s = np.linspace(0, 3e-5, 143)  # time steps in units of seconds
# tau_s = (c_ms * s0**2 * k_m**2 * t_s) / (L_m)  # The time delay due to pulse broadening, in seconds
# # tau_s = np.linspace(0, 1, 143)
# n = np.arange(0, 143)  # n an integer for infinite sums ...

# fig = plt.figure(figsize=(20, 10))
# ax = fig.add_subplot(1, 1, 1)
# sumnum54 = []
# sumnumalpha2 = []
# A = (-1/2)**n
# B = scipy.special.gamma((2 * (n + 1)) / (alpha))
# D = 2**((2 * (n + 1))/(alpha) - 1)
# for i in range(0, len(tau_s)):
#     C = (tau_s[i]**n)/(scipy.special.factorial(n)**2)
#     sumnum54.append(np.sum(A * B * C * D))
#     sumnumalpha2.append(np.sum((-1/2)**(n) * scipy.special.gamma((2 * (n + 1))/alpha2 - 1) * ((tau_s[i]**n)/(scipy.special.factorial(n)**2)) * 2**((((2 * (n + 1)))/(alpha2)) - 1)))
# sumnum54 = np.asarray(sumnum54)
# sumnumalpha2 = np.asarray(sumnum54)
# Q_PD_tau = 2 * np.pi * ((c_ms * s0**2 * k_m**2)/(alpha * z_m)) * sumnum54  # Equation 54
# Q_PD_tau2 = 2 * np.pi * ((c_ms * s0**2 * k_m**2)/(alpha2 * z_m)) * sumnumalpha2  # Equation 54
# Q_PD_tau_norm = Q_PD_tau / np.max(abs(Q_PD_tau))
# Q_PD_tau_norm2 = Q_PD_tau2 / np.max(abs(Q_PD_tau))
# plt.plot(tau_s, Q_PD_tau_norm, linewidth=4, label='Kolmogorov',alpha=0.4)
# plt.plot(tau_s, Q_PD_tau_norm2, linewidth=4, linestyle='dashed', label='Square Law', alpha=0.4)
# xl = plt.xlim()
# plt.hlines(1/np.e, xl[0], xl[1], colors='k', linestyle='dotted', label='1/e')
# plt.xlim(xl[0], xl[1])
# plt.xlabel(r"$\tau_s$ (s)")
# plt.ylabel(r"$Q_{PD}^{norm}\,(\tau_s)$")
# plt.title("An attempted copy of Figure 2 from Lambert & Rickett 1999")
# ax.legend()
# plt.savefig("/Users/jacobaskew/Desktop/Figure2Tau_d.png")
# plt.show()
# plt.close()

# tau_s_measurement = tau_s[np.argmin(abs((Q_PD_tau_norm) - (1/np.e)))]
# C1_measurement = 2 * np.pi * tau_s_measurement * DeltaNu_d
# print("The estimated scattering timescale from equation 54 is " +
#       str(round(tau_s_measurement, 2)) + "s")
# print("The estimated diffractive scintillation bandwidth from equation 67 is " +
#       str(round(DeltaNu_d_MHz, 2)) + "MHz")
# print("Therefore the estimated value for C1 using equation 68 is " +
#       str(round(C1_measurement, 5)))
# #

# fig = plt.figure(figsize=(20, 10))
# ax = fig.add_subplot(1, 1, 1)
# plt.errorbar(phase, viss*np.sqrt(1.16/C), yerr=visserr, fmt='o',
#              label='CR98 Kolmogorov', alpha=0.5, c='C0')
# plt.errorbar(phase, viss_CR2, yerr=viss_CR2_err, fmt='o',
#              label=r'CR98 $\alpha$=2', alpha=0.5, c='C1')
# plt.errorbar(phase, viss_Kol, yerr=viss_Kol_err, fmt='o', label='Kolmogorov',
#              alpha=0.5, c='C2')
# plt.errorbar(phase, viss_2, yerr=viss_2_err, fmt='o', label=r'$\alpha$=2',
#              alpha=0.5, c='C3')
# # plt.errorbar(phase, viss_u, yerr=viss_u_err, fmt='o', label='Uniform',
# #              alpha=0.5, c='C4')
# plt.xlabel("Orbital Phase (degrees)")
# plt.ylabel(r"Scintillation Velocity ($km\,s^{-1}$)")
# ax.legend()
# plt.show()
# plt.close()

# fig = plt.figure(figsize=(20, 10))
# ax = fig.add_subplot(1, 1, 1)
# plt.hist(viss*np.sqrt(1.16/C), bins=50, label='CR98 Kolmogorov', alpha=0.5,
#          density=True, color='C0')
# plt.hist(viss_CR2, bins=50, label=r'CR98 $\alpha$=2', alpha=0.5,
#          density=True, color='C1')
# plt.hist(viss_Kol, bins=50, label='Kolmogorov', alpha=0.5, density=True,
#          color='C2')
# plt.hist(viss_2, bins=50, label=r'$\alpha$=2', alpha=0.5, density=True,
#          color='C3')
# # plt.hist(viss_u, bins=50, label='Uniform', alpha=0.5, density=True,
# #          color='C4')
# plt.ylabel("Density")
# plt.xlabel(r"Scintillation Velocity ($km\,s^{-1}$)")
# ax.legend()
# plt.show()
# plt.close()
