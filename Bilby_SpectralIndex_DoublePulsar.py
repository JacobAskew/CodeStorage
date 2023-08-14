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
from lmfit import Minimizer, Parameters, minimize, fit_report
from Alpha_Likelihood import AlphaLikelihood2

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
# L-Band data

###############################################################################
# S-Band data
datadirS = "/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/New/DataFiles/2023-05-16/"

freqMHzS = np.loadtxt(datadirS + 'freq.txt', dtype='float')
dnuS = np.loadtxt(datadirS + 'dnu.txt', dtype='float')
dnuerrS = np.loadtxt(datadirS + 'dnuerr.txt', dtype='float')
###############################################################################
# # This is how I find the spectral index using bilby ...

# label = "linear_regression"
# outdir = "/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/New/Spectral_Index/Bilby/"
# bilby.utils.check_directory_exists_and_if_not_mkdir(outdir)

# ###############################################################################


# def powerlaw(xdata, a, nu_c, alpha):
#     return a * (xdata/nu_c) ** (alpha)


# ###############################################################################


# def powerlaw_lmfit(params, xdata):
#     a = params['a']
#     nu_c = params['nu_c']
#     alpha = params['alpha']
#     return a * (xdata/nu_c) ** (alpha)

# ###############################################################################


# def straight_line(xdata, params, m, b):
#     return m * xdata + b


# ###############################################################################


# def straight_line_lmfit(params, xdata):
#     m = params['m']
#     # b = params['b']
#     return m * xdata + 5


# ###############################################################################


# def fitter(model, params, args, mcmc=False, pos=None, nwalkers=100,
#            steps=1000, burn=0.2, progress=True, workers=1,
#            nan_policy='raise', max_nfev=None, thin=10, is_weighted=True):

#     # Do fit
#     if mcmc:
#         func = Minimizer(model, params, fcn_args=args)
#         mcmc_results = func.emcee(nwalkers=nwalkers, steps=steps,
#                                   burn=int(burn * steps), pos=pos,
#                                   is_weighted=is_weighted, progress=progress,
#                                   thin=thin, workers=workers)
#         results = mcmc_results
#     else:
#         func = Minimizer(model, params, fcn_args=args, nan_policy=nan_policy,
#                          max_nfev=max_nfev)
#         results = func.minimize()

#     return results


# ###############################################################################
# sigma = 0
# data = np.log(dnu)
# time = np.log(freqMHz)
# data2 = dnu
# time2 = freqMHz
# time2_sort = np.argsort(time2)
# freq_run = np.log(freqMHz)
# # dnu_run = np.log(dnu[freq_run>800])
# freq_run2 = freqMHz[freqMHz>800]
# dnu_run2 = dnu[freqMHz>800]
# injection_parameters = dict(m=3.5, b=-26.45)
# injection_parameters2 = dict(a=0.1, nu_c=1000, alpha=3.5)

# # We quickly plot the data to check it looks sensible
# fig, ax = plt.subplots()
# ax.plot(time, data, "o", label="data")
# ax.plot(time, straight_line(time, **injection_parameters), "--r", label="signal")
# ax.set_xlabel("time")
# ax.set_ylabel("y")
# ax.legend()
# fig.savefig("{}/{}log_data.png".format(outdir, label))
# plt.show()
# plt.close()

# fig, ax = plt.subplots()
# ax.plot(time2, data2, "o", label="data")
# ax.plot(time2[time2_sort], powerlaw(time2, **injection_parameters2)[time2_sort], "--r", label="signal")
# ax.set_xlabel("time")
# ax.set_ylabel("y")
# ax.legend()
# fig.savefig("{}/{}_data.png".format(outdir, label))
# plt.show()
# plt.close()


# # Now lets instantiate a version of our GaussianLikelihood, giving it
# # the time, data and signal model
# likelihood = bilby.likelihood.GaussianLikelihood(freq_run2, dnu_run2, powerlaw, sigma)

# # From hereon, the syntax is exactly equivalent to other bilby examples
# # We make a prior
# priors = dict()
# priors["a"] = bilby.core.prior.Uniform(0, 1, "a")
# priors["nu_c"] = bilby.core.prior.analytical.DeltaFunction(1000, "nu_c")
# priors["alpha"] = bilby.core.prior.Uniform(2, 6, "alpha")
# # priors["m"] = bilby.core.prior.Uniform(2, 6, "m")
# # priors["b"] = bilby.core.prior.Uniform(-100, 100, "b")

# # And run sampler
# result = bilby.run_sampler(likelihood=likelihood, priors=priors,
#                            sampler="dynesty", nlive=50, verbose=True,
#                            injection_parameters=injection_parameters,
#                            outdir=outdir, label=label, resume=False)

# # Finally plot a corner plot: all outputs are stored in outdir
# font = {'size': 16}
# matplotlib.rc('font', **font)
# result.plot_corner()
# plt.show()
# plt.close()
# ###############################################################################
# # This is how I would use lmfit to get the spectral index ...

# xdata = np.arange(0, 5)

# ydata = xdata * 3 + 5
# yerr = ydata * 0.1

# chisqr = np.inf
# posarray=[]
# nitr = 3000
# for itr in range(0, nitr):
#     ipos=[]
#     params = Parameters()
#     params.add('m', value=np.random.uniform(low=2, high=6), vary=True, min=2,
#                 max=6)
#     func = minimize(straight_line_lmfit, params, args=(xdata,))

#     ipos.append(params['m'].value)
    
#     posarray.append(ipos)    
    
#     if func.chisqr < chisqr:
#         chisqr = func.chisqr
#         func_new = func

# print(fit_report(func_new))

# fig, ax = plt.subplots()
# plt.errorbar(xdata, ydata, yerr=yerr, fmt='o', ecolor='C0', color='C0')
# plt.plot(xdata, xdata * 3 + 5, label='3')
# plt.plot(xdata, xdata * func.params['m'].value + 5,
#          label=str(round(func.params['m'].value, 2))+' $\pm$ ' +
#          str(round(func.params['m'].stderr, 2)))
# plt.legend()
# plt.show()
# plt.close()

# xdata = freqMHz

# ydata = dnu
# yerr = dnuerr

# chisqr = np.inf
# posarray=[]
# nitr = 100
# for itr in range(0, nitr):
#     ipos=[]
#     params = Parameters()
#     params.add('m', value=np.random.uniform(low=2, high=6), vary=True, min=2,
#                 max=6)
#     func = minimize(straight_line_lmfit, params, args=(xdata,))

#     ipos.append(params['m'].value)
    
#     posarray.append(ipos)    
    
#     if func.chisqr < chisqr:
#         chisqr = func.chisqr
#         func_new = func

# print(fit_report(func_new))

# fig, ax = plt.subplots()
# plt.errorbar(xdata, ydata, yerr=yerr, fmt='o', ecolor='C0', color='C0')
# plt.plot(xdata, xdata * 3 + 5, label='3')
# plt.plot(xdata, xdata * func.params['m'].value + 5,
#          label=str(round(func.params['m'].value, 2))+' $\pm$ ' +
#          str(round(func.params['m'].stderr, 2)))
# plt.legend()
# plt.show()
# plt.close()

###############################################################################

# A few simple setup steps
label = "linear_regression"
outdir = "/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/New/Spectral_Index/Bilby/"
bilby.utils.check_directory_exists_and_if_not_mkdir(outdir)

xdata = freqMHz[freqMHz > 800]
ydata = dnu[freqMHz > 800]
sigma = dnuerr[freqMHz > 800]

# S1dir = '/Users/jacobaskew/Desktop/tmp/'
# # S16dir = '/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/New/DataFiles/2023-05-16/'
# freqS1 = np.loadtxt(S1dir+'freq.txt', dtype=float)
# # freqS16 = np.loadtxt(S16dir+'freq.txt', dtype=float)
# dnuS1 = np.loadtxt(S1dir+'dnu.txt', dtype=float)
# # dnuS16 = np.loadtxt(S16dir+'dnu.txt', dtype=float)
# dnuerrS1 = np.loadtxt(S1dir+'dnuerr.txt', dtype=float)
# # dnuerrS16 = np.loadtxt(S16dir+'dnuerr.txt', dtype=float)

# xdata = freqS1
# ydata = dnuS1
# sigma = dnuerrS1

c_init = np.median(ydata[np.argwhere((xdata > 970) * (xdata < 1001))])
# c_init = np.median(ydata[np.argwhere((xdata > 2100) * (xdata < 2300))])

# First, we define our "signal model", in this case a simple linear function
def model(xdata, alpha, EFAC, EQUAD):
    return (xdata/1000)**alpha * c_init


# # Now we define the injection parameters which we make simulated data with
# injection_parameters = dict(m=0.5, c=0.2)

# # For this example, we'll use standard Gaussian noise

# # These lines of code generate the fake data. Note the ** just unpacks the
# # contents of the injection_parameters when calling the model function.
# sampling_frequency = 10
# time_duration = 10
# time = np.arange(0, time_duration, 1 / sampling_frequency)
# N = len(time)
# sigma = np.random.normal(1, 0.01, N)
# data = model(time, **injection_parameters) + np.random.normal(0, sigma, N)

# # We quickly plot the data to check it looks sensible
# fig, ax = plt.subplots()
# ax.plot(time, data, "o", label="data")
# ax.plot(time, model(time, **injection_parameters), "--r", label="signal")
# ax.set_xlabel("time")
# ax.set_ylabel("y")
# ax.legend()
# fig.savefig("{}/{}_data.png".format(outdir, label))


# Now lets instantiate a version of our GaussianLikelihood, giving it
# the xdata, ydata and signal model
likelihood = AlphaLikelihood2(xdata, ydata, model, sigma)

# From hereon, the syntax is exactly equivalent to other bilby examples
# We make a prior
priors = dict()
priors["alpha"] = bilby.core.prior.Uniform(0, 6, "alpha")
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
EFAC_result = result.posterior['EFAC'][NUM]
EFAC_std = np.std(result.posterior['EFAC'].values)
EQUAD_result = result.posterior['EQUAD'][NUM]
EQUAD_std = np.std(result.posterior['EQUAD'].values)

m_std = np.sqrt((m_std * 10**EFAC_result)**2+(10**EQUAD_result)**2)

# xdata = freqMHz
# ydata = dnu
# sigma = dnuerr
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
# cm = plt.cm.get_cmap('viridis')
# z = mjd[xsort]
ax.scatter(xdata, ydata, c='C0', label="data", alpha=0.2)
ax.errorbar(xdata, ydata, yerr=sigma, fmt=' ', color='k', linewidth=1, capsize=1, alpha=0.2)
xl = plt.xlim()
xrange = np.linspace(xl[0], xl[1], 1000)
yresults = model(xrange, alpha=m_result, EFAC=EFAC_result, EQUAD=EQUAD_result)
yresults_min = model(xrange, alpha=m_result-m_std, EFAC=EFAC_result, EQUAD=EQUAD_result)
yresults_max = model(xrange, alpha=m_result+m_std, EFAC=EFAC_result, EQUAD=EQUAD_result)
ax.plot(xrange, yresults, "C3", label=r"$\alpha^\prime$="+str(round(m_result, 3))+"$\pm$"+str(round(m_std, 3)))
plt.fill_between(xrange, yresults_min, yresults_max, color='C3', alpha=0.2)
ax.set_xlabel(r"Observational Frequency, $\nu$ (MHz)")
ax.set_ylabel(r"Scintillation Bandwidth, $\Delta\nu_d$ (MHz)")
# ax.set_title("Frequency dependence and day of year")
ax.legend(fontsize='small')
plt.xlim(xl)
fig.savefig("{}/{}_data_model.png".format(outdir, label))
plt.show()
plt.close()

alpha_prime = m_result
Beta = (2 * alpha_prime)/(alpha_prime - 2)
Beta_err = np.sqrt(((-4)/((alpha_prime - 2)**2))**2*(m_std)**2)
alpha = Beta - 2
alpha_err = Beta_err

model2 = (xdata/1000)**m_result * c_init
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
# A few simple setup steps
# label = "linear_regression"
# outdir = "/Users/jacobaskew/Desktop/DoublePulsar_Project/0737-3039A/New/Spectral_Index/Bilby/"
# bilby.utils.check_directory_exists_and_if_not_mkdir(outdir)


# # First, we define our "signal model", in this case a simple linear function
# # def model(xdata, m, c):
# #     return xdata * m + c
# ###############################################################################


# def model(xdata, m, c):
#     return c * (xdata/1000)**m


# ###############################################################################
# # Now we define the injection parameters which we make simulated data with
# injection_parameters = dict(m=3, c=0.1)
# # Fake Data
# xdata = np.arange(0, 10)
# ydata = 0.1 * (xdata/1000)**3
# sigma = ydata * 0.2
# # Real Data
# # xdata = freqMHz
# # ydata = dnu

# # We quickly plot the data to check it looks sensible
# fig, ax = plt.subplots()
# ax.plot(xdata, ydata, "o", label="data")
# ax.plot(xdata, model(xdata, **injection_parameters), "o", label="signal")
# ax.set_xlabel("freq")
# ax.set_ylabel("dnu")
# ax.legend()
# fig.savefig("{}/{}_data.png".format(outdir, label))

# # Now lets instantiate a version of our GaussianLikelihood, giving it
# # the time, data and signal model
# likelihood = bilby.likelihood.GaussianLikelihood(xdata, ydata, model, sigma=sigma)

# # From hereon, the syntax is exactly equivalent to other bilby examples
# # We make a prior
# priors = dict()
# priors["m"] = bilby.core.prior.Uniform(-100, 100, "m")
# priors["c"] = bilby.core.prior.Uniform(-100, 100, "c")

# # And run sampler
# result = bilby.run_sampler(
#     likelihood=likelihood,
#     priors=priors,
#     sampler="dynesty",
#     nlive=50,
#     injection_parameters=None,
#     outdir=outdir,
#     label=label,
#     resume=False)

# # Finally plot a corner plot: all outputs are stored in outdir
# font = {'size': 16}
# matplotlib.rc('font', **font)
# result.plot_corner()
# plt.show()
# plt.close()

###############################################################################
