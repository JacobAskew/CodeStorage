#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 14:04:17 2023

@author: dreardon
"""

import numpy as np
import matplotlib.pyplot as plt
import bilby
from bilby.core.likelihood import Analytical1DLikelihood


class GaussianLikelihood(Analytical1DLikelihood):
    def __init__(self, x, y, func, sigma=None, freq=None, **kwargs):
        """
        A general Gaussian likelihood for known or unknown noise - the model
        parameters are inferred from the arguments of function

        Parameters
        ==========
        x, y: array_like
            The data to analyse
        func:
            The python function to fit to the data. Note, this must take the
            dependent variable as its first argument. The other arguments
            will require a prior and will be sampled over (unless a fixed
            value is given).
        sigma: None, float, array_like
            If None, the standard deviation of the noise is unknown and will be
            estimated (note: this requires a prior to be given for sigma). If
            not None, this defines the standard-deviation of the data points.
            This can either be a single float, or an array with length equal
            to that for `x` and `y`.
        """

        super(GaussianLikelihood, self).__init__(x=x, y=y, func=func, **kwargs)
        self.sigma = sigma
        self.freq = freq

        # Check if sigma was provided, if not it is a parameter
        if self.sigma is None:
            self.parameters['sigma'] = None


    def log_likelihood(self):
        Q = self.model_parameters["Q"]
        F = self.model_parameters["F"]
        alpha = self.model_parameters["alpha"]
        freq = self.freq
        # Modifying the noise levels within the pdfs
        Sigma = np.sqrt((self.sigma * F)**2 + (Q *
                        (freq/1000)**alpha)**2)

        log_l = np.sum(- (self.residual / Sigma)**2 / 2 -
                       np.log(2 * np.pi * Sigma**2) / 2)
        return log_l

    def __repr__(self):
        return self.__class__.__name__ + '(x={}, y={}, func={}, sigma={})' \
            .format(self.x, self.y, self.func.__name__, self.sigma)

    @property
    def sigma(self):
        """
        This checks if sigma has been set in parameters. If so, that value
        will be used. Otherwise, the attribute sigma is used. The logic is
        that if sigma is not in parameters the attribute is used which was
        given at init (i.e. the known sigma as either a float or array).
        """
        return self.parameters.get('sigma', self._sigma)

    @sigma.setter
    def sigma(self, sigma):
        if sigma is None:
            self._sigma = sigma
        elif isinstance(sigma, float) or isinstance(sigma, int):
            self._sigma = sigma
        elif len(sigma) == self.n:
            self._sigma = sigma
        else:
            raise ValueError('Sigma must be either float or array-like x.')




def spectrum_jumps(freqs, alpha, F, Q,
                   dnu1, dnu2, dnu3, dnu4, dnu5, dnu6, dnu7, dnu8, dnu9):

    model = np.ones_like(freqs)

    dnu_epochs = [dnu1, dnu2, dnu3, dnu4, dnu5, dnu6, dnu7, dnu8, dnu9]

    epochs = np.unique(np.round(mjd, -1))

    for i in range(len(epochs)):
        ep = epochs[i]
        inds = np.argwhere(np.round(mjd, -1) == ep).squeeze()

        model[inds] = dnu_epochs[i] * (freqs[inds] / 1000) ** alpha

    return model


plt.figure(figsize=(9, 6))

dnu = np.loadtxt('/Users/jacobaskew/Desktop/Data4Daniel/HiRes_UHF_Data/_dnu.txt')
df = np.loadtxt('/Users/jacobaskew/Desktop/Data4Daniel/HiRes_UHF_Data/_df.txt')
dnuerr = np.loadtxt('/Users/jacobaskew/Desktop/Data4Daniel/HiRes_UHF_Data/_dnuerr.txt')
freq = np.loadtxt('/Users/jacobaskew/Desktop/Data4Daniel/HiRes_UHF_Data/_freqMHz.txt')
mjd = np.loadtxt('/Users/jacobaskew/Desktop/Data4Daniel/HiRes_UHF_Data/_mjd.txt')

plt.errorbar(freq, dnu, yerr=dnuerr, fmt='rx')
plt.yscale('log')
plt.xscale('log')


l_dnu = np.loadtxt('/Users/jacobaskew/Desktop/Data4Daniel/HiRes_Lband_Data/_dnu.txt')
l_df = np.loadtxt('/Users/jacobaskew/Desktop/Data4Daniel/HiRes_Lband_Data/_df.txt')
l_dnuerr = np.loadtxt('/Users/jacobaskew/Desktop/Data4Daniel/HiRes_Lband_Data/_dnuerr.txt')
l_freq = np.loadtxt('/Users/jacobaskew/Desktop/Data4Daniel/HiRes_Lband_Data/_freqMHz.txt')
l_mjd = np.loadtxt('/Users/jacobaskew/Desktop/Data4Daniel/HiRes_Lband_Data/_mjd.txt')

plt.errorbar(l_freq, l_dnu, yerr=l_dnuerr, fmt='gx')
plt.yscale('log')
plt.xscale('log')

s_dnu = np.loadtxt('/Users/jacobaskew/Desktop/Data4Daniel/HiRes_Sband_Data/_dnu.txt')
s_df = np.loadtxt('/Users/jacobaskew/Desktop/Data4Daniel/HiRes_Sband_Data/_df.txt')
s_dnuerr = np.loadtxt('/Users/jacobaskew/Desktop/Data4Daniel/HiRes_Sband_Data/_dnuerr.txt')
s_freq = np.loadtxt('/Users/jacobaskew/Desktop/Data4Daniel/HiRes_Sband_Data/_freqMHz.txt')
s_mjd = np.loadtxt('/Users/jacobaskew/Desktop/Data4Daniel/HiRes_Sband_Data/_mjd.txt')

plt.errorbar(s_freq, s_dnu, yerr=s_dnuerr, fmt='bx')
plt.yscale('log')
plt.xscale('log')

plt.ylabel('Scint bandwidth (MHz)')
plt.xlabel('Frequency (MHz)')

plt.savefig('data.png')
plt.close()


mjd = np.concatenate((mjd, l_mjd, s_mjd))

sortind = np.argsort(mjd)
mjd = mjd[sortind]
dnu = np.concatenate((dnu, l_dnu, s_dnu))[sortind]
df = np.concatenate((df, l_df, s_df))[sortind]
dnuerr = np.concatenate((dnuerr, l_dnuerr, s_dnuerr))[sortind]
freq = np.concatenate((freq, l_freq, s_freq))[sortind]

inds = np.argwhere((dnu > df) * (freq > 800)).squeeze()
mjd = mjd[inds]
dnu = dnu[inds]
df = df[inds]
dnuerr = dnuerr[inds]
freq = freq[inds]

epoch = np.round(mjd, -1)


priors = dict()
priors['dnu1'] = bilby.core.prior.Uniform(0.01, 0.3, 'dnu1')  # MHz
priors['dnu2'] = bilby.core.prior.Uniform(0.01, 0.3, 'dnu2')  # MHz
priors['dnu3'] = bilby.core.prior.Uniform(0.01, 0.3, 'dnu3')  # MHz
priors['dnu4'] = bilby.core.prior.Uniform(0.01, 0.3, 'dnu4')  # MHz
priors['dnu5'] = bilby.core.prior.Uniform(0.01, 0.3, 'dnu5')  # MHz
priors['dnu6'] = bilby.core.prior.Uniform(0.01, 0.3, 'dnu6')  # MHz
priors['dnu7'] = bilby.core.prior.Uniform(0.01, 0.3, 'dnu7')  # MHz
priors['dnu8'] = bilby.core.prior.Uniform(0.01, 0.3, 'dnu8')  # MHz
priors['dnu9'] = bilby.core.prior.Uniform(0.01, 0.3, 'dnu9')  # MHz
priors['alpha'] = bilby.core.prior.Uniform(2, 5, 'alpha')  # Dimensionless index
priors['F'] = bilby.core.prior.Uniform(0, 10, 'F')  # Dimensionless scale factor
priors['Q'] = bilby.core.prior.Uniform(0, 1, 'Q')  # MHz

likelihood = GaussianLikelihood(freq, dnu, spectrum_jumps, sigma=dnuerr, freq=freq)

outdir = '/Users/jacobaskew/Desktop/Data4Daniel/output/'

results = bilby.core.sampler.run_sampler(
    likelihood, priors=priors, sampler='dynesty', label='dynesty',
    nlive=100, verbose=True, resume=True, outdir=outdir,
    check_point_delta_t=120)

results.plot_corner()

def modify_errors(dnuerr, freq, F, Q, alpha):
    return  np.sqrt((dnuerr * F)**2 + (Q * (freq/1000)**alpha)**2)




F = np.median(results.posterior['F'])
Q = np.median(results.posterior['Q'])
alpha = np.median(results.posterior['alpha'])
alpha_err = np.std(results.posterior['alpha'])


dnuerr = modify_errors(dnuerr, freq, F, Q, alpha)

dnu1 = np.median(results.posterior['dnu1'])
dnu2 = np.median(results.posterior['dnu2'])
dnu3 = np.median(results.posterior['dnu3'])
dnu4 = np.median(results.posterior['dnu4'])
dnu5 = np.median(results.posterior['dnu5'])
dnu6 = np.median(results.posterior['dnu6'])
dnu7 = np.median(results.posterior['dnu7'])
dnu8 = np.median(results.posterior['dnu8'])
dnu9 = np.median(results.posterior['dnu9'])

dnu_epochs = [dnu1, dnu2, dnu3, dnu4, dnu5, dnu6, dnu7, dnu8, dnu9]
colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8']

plt.figure(figsize=(9, 6))
epochs = np.unique(np.round(mjd, -1))
for i in range(len(epochs)):
    ep = epochs[i]
    inds = np.argwhere(epoch == ep).squeeze()
    model = dnu_epochs[i] * (freq[inds] / 1000) ** alpha
    plt.errorbar(freq[inds], dnu[inds], yerr=dnuerr[inds], fmt='x', color=colors[i])
    plt.plot(freq[inds], model, color=colors[i])


plt.yscale('log')
plt.xscale('log')
plt.ylabel('Scint bandwidth (MHz)')
plt.xlabel('Frequency (MHz)')
plt.title('Data with modified error bars')
plt.savefig('result.png')
plt.close()


plt.figure(figsize=(9, 6))
epochs = np.unique(np.round(mjd, -1))
for i in range(len(epochs)):
    ep = epochs[i]
    inds = np.argwhere(epoch == ep).squeeze()
    model = dnu_epochs[i] * (freq[inds] / 1000) ** alpha
    plt.errorbar(freq[inds], dnu[inds] - model, yerr=dnuerr[inds], fmt='x', color=colors[i])

plt.ylabel('Residual (MHz)')
plt.xlabel('Frequency (MHz)')
plt.title('Residuals')
plt.savefig('res.png')
plt.close()

plt.figure(figsize=(9, 6))
epochs = np.unique(np.round(mjd, -1))
for i in range(len(epochs)):
    ep = epochs[i]
    inds = np.argwhere(epoch == ep).squeeze()
    model = dnu_epochs[i] * (freq[inds] / 1000) ** alpha
    plt.scatter(freq[inds], (dnu[inds] - model) / dnuerr[inds], marker='x', color=colors[i])

plt.ylabel('Normalised Residual')
plt.xlabel('Frequency (MHz)')
plt.title('Normalised residuals')
plt.savefig('normres.png')
plt.close()

dnu1_err = np.std(results.posterior['dnu1'])
dnu2_err = np.std(results.posterior['dnu2'])
dnu3_err = np.std(results.posterior['dnu3'])
dnu4_err = np.std(results.posterior['dnu4'])
dnu5_err = np.std(results.posterior['dnu5'])
dnu6_err = np.std(results.posterior['dnu6'])
dnu7_err = np.std(results.posterior['dnu7'])
dnu8_err = np.std(results.posterior['dnu8'])
dnu9_err = np.std(results.posterior['dnu9'])

dnu_err_epochs = [dnu1_err, dnu2_err, dnu3_err, dnu4_err, dnu5_err,
                  dnu6_err, dnu7_err, dnu8_err, dnu9_err]

plt.figure(figsize=(9, 6))
epochs = np.unique(np.round(mjd, -1))
plt.errorbar(epochs, dnu_epochs, yerr=dnu_err_epochs, fmt='x')
plt.ylabel('Scint bandwidth at 1 GHz')
plt.xlabel('MJD')
plt.title('Scint bandwidth width time')
plt.savefig('dnu_time.png')
plt.close()


print("########")
print("Alpha = {} +/- {}".format(alpha, alpha_err))
print("########\n")








