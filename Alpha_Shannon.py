#!/usr/bin/env python
"""
A measurement of the spectral index.
 
"""
###############################################################################
# Importing neccessary things #
from scintools.scint_utils import read_par, pars_to_params
import bilby
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from AlphaLikelihood import AlphaLikelihood
# from lmfit import Minimizer, Parameters, minimize, fit_report
###############################################################################

desktopdir = '/Users/jacobaskew/Desktop/'
datadir = desktopdir + 'DoublePulsar_Project/datasets/FullDataStorage/'
par_dir = '/Users/jacobaskew/Desktop/Swinburne_Daniel/ParFiles/'
psrname = 'J0737-3039A'
wd0 = "/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/New/"
outdir = wd0 + "Modelling"
pars = read_par(str(par_dir) + str(psrname) + '.par')
params = pars_to_params(pars)

viss = np.loadtxt(datadir + 'Full_VissData.txt', dtype='float')
visserr = np.loadtxt(datadir + 'Full_VisserrData.txt', dtype='float')
mjd = np.loadtxt(datadir + 'Full_MJDData.txt', dtype='float')
freqMHz = np.loadtxt(datadir + 'Full_FreqData.txt', dtype='float')
freqGHz = freqMHz / 1e3
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

# A few simple setup steps
label = "linear_regression"
outdir = "/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/New/Spectral_Index/Bilby/"
bilby.utils.check_directory_exists_and_if_not_mkdir(outdir)

# xdata = freqMHz[freqMHz > 800]
# ydata = dnu[freqMHz > 800]
# sigma = dnuerr[freqMHz > 800]

xdata = freqMHz[np.argwhere((mjd > 59760) * (mjd < 59762))][50:70].flatten()
ydata = dnu[np.argwhere((mjd > 59760) * (mjd < 59762))][50:70].flatten()
sigma = dnuerr[np.argwhere((mjd > 59760) * (mjd < 59762))][50:70].flatten() * 10

ydata = ((xdata/1000) ** 4.4) * 0.1 + np.random.normal()*sigma

# S1dir = '/Users/jacobaskew/Desktop/tmp/'
# S16dir = '/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/New/DataFiles/2023-05-16/'
# freqS1 = np.loadtxt(S1dir+'freq.txt', dtype=float)
# freqS16 = np.loadtxt(S16dir+'freq.txt', dtype=float)
# dnuS1 = np.loadtxt(S1dir+'dnu.txt', dtype=float)
# dnuS16 = np.loadtxt(S16dir+'dnu.txt', dtype=float)
# dnuerrS1 = np.loadtxt(S1dir+'dnuerr.txt', dtype=float)
# dnuerrS16 = np.loadtxt(S16dir+'dnuerr.txt', dtype=float)

# xdata = freqS1
# ydata = dnuS1
# sigma = dnuerrS1

c_init = np.median(ydata[np.argwhere((xdata > 970) * (xdata < 1001))])
# c_init = np.median(ydata[np.argwhere((xdata > 2100) * (xdata < 2300))])

# First, we define our "signal model", in this case a simple linear function
# def model(xdata, alpha, EFAC, EQUAD):
def model(xdata, alpha):
    return (xdata/1000)**alpha * c_init


# Now lets instantiate a version of our GaussianLikelihood, giving it
# the xdata, ydata and signal model
likelihood = AlphaLikelihood(x=xdata, y=ydata, func=model, sigma=sigma)

# From hereon, the syntax is exactly equivalent to other bilby examples
# We make a prior on alpha and white noise parameters
priors = dict()
priors["alpha"] = bilby.core.prior.Uniform(0, 6, "alpha")
# priors["EFAC"] = bilby.core.prior.Uniform(-2, 2, "EFAC")
# priors["EQUAD"] = bilby.core.prior.Uniform(-2, 2, "EQUAD")

# And run sampler
result = bilby.run_sampler(
    likelihood=likelihood,
    priors=priors,
    sampler="dynesty",
    nlive=250,
    outdir=outdir,
    label=label,
)

# Finally plot a corner plot: all outputs are stored in outdir
font = {'size': 16}
matplotlib.rc('font', **font)
result.plot_corner()
plt.show()
plt.close()

NUM = np.argmax(result.posterior['log_likelihood'])
alpha = result.posterior['alpha'][NUM]
alpha_std = np.std(result.posterior['alpha'].values)
# EFAC = result.posterior['EFAC'][NUM]
# EFAC_std = np.std(result.posterior['EFAC'].values)
# EQUAD = result.posterior['EQUAD'][NUM]
# EQUAD_std = np.std(result.posterior['EQUAD'].values)

# alpha_std = np.sqrt((alpha_std1 * 10**EFAC)**2 + (10**EQUAD)**2)

xsort = np.argsort(xdata)

Size = 80*np.pi  # Determines the size of the datapoints used
font = {'size': 28}
matplotlib.rc('font', **font)

# fig = plt.figure(figsize=(20, 10))
# fig.subplots_adjust(hspace=0.5, wspace=0.5)
# ax = fig.add_subplot(1, 1, 1)
# plt.hist(result.posterior['alpha'], bins=20, color='C0', alpha=0.9)
# yl = plt.ylim()
# plt.vlines(alpha, yl[0], yl[1], colors='C3', label=r"$\alpha^\prime$="+str(round(alpha, 3))+"$\pm$"+str(round(alpha_std, 3)))
# plt.vlines(alpha - alpha_std, yl[0], yl[1], linestyles='dashed', colors='C3')
# plt.vlines(alpha + alpha_std, yl[0], yl[1], linestyles='dashed', colors='C3')
# ax.legend(fontsize='small')
# ax.set_xlabel(r"Spectral Index, $\alpha^\prime$ (MHz)")
# plt.ylim(yl)
# fig.savefig("{}/{}_alpha_dist.png".format(outdir, label))
# plt.show()
# plt.close()

# We quickly plot the data to check it looks sensible
fig = plt.figure(figsize=(20, 10))
fig.subplots_adjust(hspace=0.5, wspace=0.5)
ax = fig.add_subplot(1, 1, 1)
# cm = plt.cm.get_cmap('viridis')
# z = mjd[xsort]
ax.scatter(xdata, ydata, c='C0', label="data", alpha=0.2)
ax.errorbar(xdata, ydata, yerr=sigma, fmt=' ', color='k', linewidth=1, capsize=1, alpha=0.2)
xl = plt.xlim()
xrange = np.linspace(xl[0], xl[1], 1000)
# ax.plot(xrange, model(xrange, alpha=alpha, EFAC=EFAC, EQUAD=EQUAD), "C3", label=r"$\alpha^\prime$="+str(round(alpha, 3))+"$\pm$"+str(round(alpha_std, 3)))
ax.plot(xrange, model(xrange, alpha=alpha), "C3", label=r"$\alpha^\prime$="+str(round(alpha, 3))+"$\pm$"+str(round(alpha_std, 3)))
ax.set_xlabel(r"Observational Frequency, $\nu$ (MHz)")
ax.set_ylabel(r"Scintillation Bandwidth, $\Delta\nu_d$ (MHz)")
# ax.set_title("Frequency dependence and day of year")
ax.legend(fontsize='small')
plt.xlim(xl)
fig.savefig("{}/{}_data_model.png".format(outdir, label))
plt.show()
plt.close()

# alpha_prime = alpha
# Beta = (2 * alpha_prime)/(alpha_prime - 2)
# Beta_err = np.sqrt(((-4)/((alpha_prime - 2)**2))**2*(alpha_std)**2)
# alpha = Beta - 2
# alpha_err = Beta_err

# model = (xdata/2200)**alpha * c_init
# chisqr = np.sum((ydata - model)**2/(model))
# Ndata = len(xdata)
# Nparam = 1
# Nfree = Ndata - Nparam
# red_chisqr = chisqr / Nfree

# print("========== We found the relationship between our data and observational frequency to be ==========")
# print("========== alpha_prime = "+str(round(alpha, 3))+" +/- "+str(round(alpha_std, 3))+" ==========")
# print("========== alpha       = "+str(round(alpha, 3))+" +/- "+str(round(alpha_err, 3))+" ==========")
# print("========== Beta        = "+str(round(Beta, 3))+" +/- "+str(round(Beta_err, 3))+" ==========")
# print("========== Chisqr      = "+str(round(chisqr, 3))+" ====================")
# print("========== RedChisqr   = "+str(round(red_chisqr, 3))+" ====================")
###############################################################################
