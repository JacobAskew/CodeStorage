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
from Alpha_Likelihood import AlphaLikelihood2

###############################################################################

desktopdir = '/Users/jacobaskew/Desktop/'
datadir = desktopdir + 'DoublePulsar_Project/datasets/FullDataStorage/'
par_dir = '/Users/jacobaskew/Desktop/Swinburne_Daniel/ParFiles/'
psrname = 'J0737-3039A'
wd0 = "/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/New/"
outdir = wd0 + "Spectral_Index/All/"
pars = read_par(str(par_dir) + str(psrname) + '.par')
params = pars_to_params(pars)

# viss = np.loadtxt(datadir + 'Full_VissData.txt', dtype='float')
# visserr = np.loadtxt(datadir + 'Full_VisserrData.txt', dtype='float')
mjd = np.loadtxt(datadir + 'Full_MJDData.txt', dtype='float')
freqMHz = np.loadtxt(datadir + 'Full_FreqData.txt', dtype='float')
# freqGHz = freqMHz / 1e3
dnu = np.loadtxt(datadir + 'Full_DnuData.txt', dtype='float')
dnuerr = np.loadtxt(datadir + 'Full_DnuerrData.txt', dtype='float')
# tau = np.loadtxt(datadir + 'Full_TauData.txt', dtype='float')
# tauerr = np.loadtxt(datadir + 'Full_TauerrData.txt', dtype='float')
phase = np.loadtxt(datadir + 'Full_PhaseData.txt', dtype='float')
# U = np.loadtxt(datadir + 'Full_UData.txt', dtype='float')
# ve_ra = np.loadtxt(datadir + 'Full_ve_raData.txt', dtype='float')
# ve_dec = np.loadtxt(datadir + 'Full_ve_decData.txt', dtype='float')

# kwargs = {"U": U, "ve_ra": ve_ra, "ve_dec": ve_dec, "params": params}
###############################################################################
# L-Band data

###############################################################################
# S-Band data
datadirS = "/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/New/DataFiles/2023-05-16/"

freqMHzS = np.loadtxt(datadirS + 'freq.txt', dtype='float')
dnuS = np.loadtxt(datadirS + 'dnu.txt', dtype='float')
dnuerrS = np.loadtxt(datadirS + 'dnuerr.txt', dtype='float')
###############################################################################
# A few simple setup steps
label = "AlphaPrime"
bilby.utils.check_directory_exists_and_if_not_mkdir(outdir)

xdata = freqMHz[freqMHz > 800]
ydata = dnu[freqMHz > 800]
sigma = dnuerr[freqMHz > 800]

xdata = np.concatenate((xdata, freqMHzS))
ydata = np.concatenate((ydata, dnuS))
sigma = np.concatenate((sigma, dnuerrS))

dnu_init_min = np.min(ydata)
dnu_init_max = np.max(ydata)
###############################################################################


def model(xdata, alpha, dnu_init, EFAC, EQUAD):
    return (xdata/1000)**alpha * dnu_init


###############################################################################
likelihood = AlphaLikelihood2(xdata, ydata, model, sigma)

priors = dict()
priors["alpha"] = bilby.core.prior.Uniform(0, 6, "alpha")
priors["dnu_init"] = bilby.core.prior.Uniform(dnu_init_min, dnu_init_max, "dnu_init")
priors["EFAC"] = bilby.core.prior.Uniform(-10, 10, "EFAC")
priors["EQUAD"] = bilby.core.prior.Uniform(-10, 10, "EQUAD")

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
m_result = result.posterior['alpha'][NUM]
m_std = np.std(result.posterior['alpha'].values)
dnu_init_result = result.posterior['dnu_init'][NUM]
dnu_init_std = np.std(result.posterior['dnu_init'].values)
EFAC_result = result.posterior['EFAC'][NUM]
EFAC_std = np.std(result.posterior['EFAC'].values)
EQUAD_result = result.posterior['EQUAD'][NUM]
EQUAD_std = np.std(result.posterior['EQUAD'].values)

m_std = np.sqrt((m_std * 10**EFAC_result)**2+(10**EQUAD_result)**2)

xsort = np.argsort(xdata)

Size = 80*np.pi  # Determines the size of the datapoints used
font = {'size': 28}
matplotlib.rc('font', **font)

fig = plt.figure(figsize=(20, 10))
fig.subplots_adjust(hspace=0.5, wspace=0.5)
ax = fig.add_subplot(1, 1, 1)
plt.hist(result.posterior['alpha'], bins=20, color='C0', alpha=0.9)
yl = plt.ylim()
plt.vlines(m_result, yl[0], yl[1], colors='C3', label=r"$\alpha^\prime$="+str(round(m_result, 3))+"$\pm$"+str(round(m_std, 3)))
plt.vlines(m_result - m_std, yl[0], yl[1], linestyles='dashed', colors='C3')
plt.vlines(m_result + m_std, yl[0], yl[1], linestyles='dashed', colors='C3')
ax.legend(fontsize='small')
ax.set_xlabel(r"Spectral Index, $\alpha^\prime$ (MHz)")
plt.ylim(yl)
fig.savefig("{}/{}_alpha_dist.png".format(outdir, label))
plt.show()
plt.close()

# We quickly plot the data to check it looks sensible
fig = plt.figure(figsize=(20, 10))
fig.subplots_adjust(hspace=0.5, wspace=0.5)
ax = fig.add_subplot(1, 1, 1)
ax.scatter(xdata, ydata, c='C0', label="data", alpha=0.2)
ax.errorbar(xdata, ydata, yerr=sigma, fmt=' ', color='k', linewidth=1, capsize=1, alpha=0.2)
xl = plt.xlim()
xrange = np.linspace(xl[0], xl[1], 1000)
yresults = model(xrange, alpha=m_result, dnu_init=dnu_init_result, EFAC=EFAC_result, EQUAD=EQUAD_result)
yresults_min = model(xrange, alpha=m_result-m_std, dnu_init=dnu_init_result, EFAC=EFAC_result, EQUAD=EQUAD_result)
yresults_max = model(xrange, alpha=m_result+m_std, dnu_init=dnu_init_result, EFAC=EFAC_result, EQUAD=EQUAD_result)
ax.plot(xrange, yresults, "C3", label=r"$\alpha^\prime$="+str(round(m_result, 3))+"$\pm$"+str(round(m_std, 3)))
plt.fill_between(xrange, yresults_min, yresults_max, color='C3', alpha=0.2)
ax.set_xlabel(r"Observational Frequency, $\nu$ (MHz)")
ax.set_ylabel(r"Scintillation Bandwidth, $\Delta\nu_d$ (MHz)")
ax.legend(fontsize='small')
plt.xlim(xl)
fig.savefig("{}/{}_data_model.png".format(outdir, label))
ax.set_xscale('log')
ax.set_yscale('log')
plt.show()
plt.close()

alpha_prime = m_result
Beta = (2 * alpha_prime)/(alpha_prime - 2)
Beta_err = np.sqrt(((-4)/((alpha_prime - 2)**2))**2*(m_std)**2)
alpha = Beta - 2
alpha_err = Beta_err

model2 = (xdata/1000)**m_result * dnu_init_result
chisqr = np.sum((ydata - model2)**2/(model2))
Ndata = len(xdata)
Nparam = 1
Nfree = Ndata - Nparam
red_chisqr = chisqr / Nfree

print("========== We found the relationship between our data and observational frequency to be ==========")
print("========== alpha_prime = "+str(round(m_result, 3))+" +/- "+str(round(m_std, 3))+" ==========")
print("========== alpha       = "+str(round(alpha, 3))+" +/- "+str(round(alpha_err, 3))+" ==========")
print("========== Beta        = "+str(round(Beta, 3))+" +/- "+str(round(Beta_err, 3))+" ==========")
print("========== Chisqr      = "+str(round(chisqr, 3))+" ====================")
print("========== RedChisqr   = "+str(round(red_chisqr, 3))+" ====================")
###############################################################################
