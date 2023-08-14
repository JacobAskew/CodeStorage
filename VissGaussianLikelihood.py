#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 15:23:28 2023

@author: jacobaskew
"""

import numpy as np
import types
import inspect
import sys
import matplotlib.pyplot as plt
# from scintools.scint_utils import get_true_anomaly, get_earth_velocity, \
#     read_par, pars_to_params


def infer_parameters_from_function(func):
    """ Infers the arguments of a function
    (except the first arg which is assumed to be the dep. variable).

    Throws out `*args` and `**kwargs` type arguments

    Can deal with type hinting!

    Parameters
    ==========
    func: function or method
       The function or method for which the parameters should be inferred.

    Returns
    =======
    list: A list of strings with the parameters

    Raises
    ======
    ValueError
       If the object passed to the function is neither a function nor a method.

    Notes
    =====
    In order to handle methods the `type` of the function is checked, and
    if a method has been passed the first *two* arguments are removed rather
    than just the first one.
    This allows the reference to the instance (conventionally named `self`)
    to be removed.
    """
    if isinstance(func, types.MethodType):
        return infer_args_from_function_except_n_args(func=func, n=2)
    elif isinstance(func, types.FunctionType):
        return _infer_args_from_function_except_for_first_arg(func=func)
    else:
        raise ValueError("This doesn't look like a function.")


def infer_args_from_function_except_n_args(func, n=1):
    """ Inspects a function to find its arguments, and ignoring the
    first n of these, returns a list of arguments from the function's
    signature.

    Parameters
    ==========
    func : function or method
       The function from which the arguments should be inferred.
    n : int
       The number of arguments which should be ignored,
       staring at the beginning.

    Returns
    =======
    parameters: list
       A list of parameters of the function, omitting the first `n`.

    Extended Summary
    ================
    This function is intended to allow the handling of named arguments
    in both functions and methods; this is important, since the first
    argument of an instance method will be the instance.

    See Also
    ========
    infer_args_from_method: Provides the arguments for a method
    infer_args_from_function: Provides the arguments for a function
    infer_args_from_function_except_first_arg: Provides all but first
    argument of a function or method.

    Examples
    ========

    .. code-block:: python

        >>> def hello(a, b, c, d):
        >>>     pass
        >>>
        >>> infer_args_from_function_except_n_args(hello, 2)
        ['c', 'd']

    """
    parameters = inspect.getfullargspec(func).args
    del parameters[:n]
    return parameters


def _infer_args_from_function_except_for_first_arg(func):
    return infer_args_from_function_except_n_args(func=func, n=1)


class Likelihood(object):

    def __init__(self, parameters=None):
        """Empty likelihood class to be subclassed by other likelihoods

        Parameters
        ==========
        parameters: dict
            A dictionary of the parameter names and associated values
        """
        self.parameters = parameters
        self._meta_data = None
        self._marginalized_parameters = []

    def __repr__(self):
        return self.__class__.__name__ + \
            '(parameters={})'.format(self.parameters)

    def log_likelihood(self):
        """

        Returns
        =======
        float
        """
        return np.nan

    def noise_log_likelihood(self):
        """

        Returns
        =======
        float
        """
        return np.nan

    def log_likelihood_ratio(self):
        """Difference between log likelihood and noise log likelihood

        Returns
        =======
        float
        """
        return self.log_likelihood() - self.noise_log_likelihood()

    @property
    def meta_data(self):
        return getattr(self, '_meta_data', None)

    @meta_data.setter
    def meta_data(self, meta_data):
        if isinstance(meta_data, dict):
            self._meta_data = meta_data
        else:
            raise ValueError("The meta_data must be an instance of dict")

    @property
    def marginalized_parameters(self):
        return self._marginalized_parameters


class Analytical1DLikelihood(Likelihood):
    """
    A general class for 1D analytical functions. The model
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
    """

    def __init__(self, x, y, func, freq, tau, dnu, tauerr, dnuerr, **kwargs):
        parameters = infer_parameters_from_function(func)
        super(Analytical1DLikelihood, self).__init__(dict())
        self.x = x
        self.y = y
        self._func = func
        self.freq = freq
        self.dnu = dnu
        self.dnuerr = dnuerr
        self.tau = tau
        self.tauerr = tauerr
        self._function_keys = [key for key in parameters if key not in kwargs]
        self.kwargs = kwargs

    def __repr__(self):
        return self.__class__.__name__ + \
            '(x={}, y={}, func={})'.format(self.x, self.y, self.func.__name__)

    @property
    def func(self):
        """ Make func read-only """
        return self._func

    @property
    def model_parameters(self):
        """ This sets up the function only parameters
        (i.e. not sigma for the GaussianLikelihood) """
        return {key: self.parameters[key] for key in self.function_keys}

    @property
    def function_keys(self):
        """ Makes function_keys read_only """
        return self._function_keys

    @property
    def n(self):
        """ The number of data points """
        return len(self.x)

    @property
    def x(self):
        """ The independent variable. Setter assures that single numbers
        will be converted to arrays internally """
        return self._x

    @x.setter
    def x(self, x):
        if isinstance(x, int) or isinstance(x, float):
            x = np.array([x])
        self._x = x

    @property
    def y(self):
        """ The dependent variable. Setter assures that single numbers
        will be converted to arrays internally """
        return self._y

    @y.setter
    def y(self, y):
        if isinstance(y, int) or isinstance(y, float):
            y = np.array([y])
        self._y = y

    @property
    def residual(self):
        """ Residual of the function against the data. """
        return self.y - self.func(self.x, **self.model_parameters,
                                  **self.kwargs)


class VissGaussianLikelihood(Analytical1DLikelihood):

    def __init__(self, x, y, func, freq, tau, dnu, tauerr, dnuerr, sigma=None,
                 **kwargs):
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

        super(VissGaussianLikelihood, self).__init__(
            x=x, y=y, func=func, freq=freq, tau=tau, dnu=dnu, tauerr=tauerr,
            dnuerr=dnuerr, **kwargs)
        self.sigma = sigma

        # Check if sigma was provided, if not it is a parameter
        if self.sigma is None:
            self.parameters['sigma'] = None

    def log_likelihood(self):
        # DNUESKEW, TAUESKEW, DNUEFAC, TAUEFAC, alpha, A, D, kappa, \
        #     kappa2, s, s2 = map(
        #         float, [params["DNUESKEW"], params["TAUESKEW"],
        #                 params["DNUEFAC"], params["TAUEFAC"], params["alpha"],
        #                 params["A"], params["d"], params["k"], params["k2"],
        #                 params["s"], params["s2"]])
        # D_err = 0.06
        # mjd_stop = int(self.model_parameters["mjd_stop"])
        # windowing = True
        # if not windowing:
        #     mjd = self.x
        #     # argsort1 = mjd < mjd_stop
        #     # argsort2 = mjd > mjd_stop
        #     freqGHz1 = self.freq[argsort1] / 1e3
        #     freqGHz2 = self.freq[argsort2] / 1e3
        #     dnu1 = self.dnu[argsort1]
        #     dnu2 = self.dnu[argsort2]
        #     tau1 = self.tau[argsort1]
        #     tau2 = self.tau[argsort2]
        #     dnuerr1 = self.dnuerr[argsort1]
        #     dnuerr2 = self.dnuerr[argsort2]
        #     tauerr1 = self.tauerr[argsort1]
        #     tauerr2 = self.tauerr[argsort2]
        #     DSKEW1 = (10**DNUESKEW) * (freqGHz1/1)**alpha
        #     DSKEW2 = (10**DNUESKEW) * (freqGHz2/1)**alpha
        #     TSKEW1 = (10**TAUESKEW) * (freqGHz1/1)**(alpha/2)
        #     TSKEW2 = (10**TAUESKEW) * (freqGHz2/1)**(alpha/2)
        #     # TAU EFAC ESKEW, DNU EFAC ESKEW
        #     dnuerr1 *= DFAC
        #     dnuerr1 = np.sqrt(dnuerr1**2 + DSKEW1**2)
        #     dnuerr2 *= DFAC
        #     dnuerr2 = np.sqrt(dnuerr2**2 + DSKEW2**2)
        #     tauerr1 *= TFAC
        #     tauerr1 = np.sqrt(tauerr1**2 + TSKEW1**2)
        #     tauerr2 *= TFAC
        #     tauerr2 = np.sqrt(tauerr2**2 + TSKEW2**2)
        #     # TAU EFAC EQUAD, DNU EFAC EQUAD
        #     # dnuerr = np.sqrt((self.dnuerr * 10**DNUEFAC)**2 +
        #     #                  (10**DNUESKEW)**2)
        #     # tauerr = np.sqrt((self.tauerr * 10**TAUEFAC)**2 +
        #     #                  (10**TAUESKEW)**2)
        #     # Measuring viss and visserr
        #     Aiss1 = kappa * A * np.sqrt((2*(1-s))/(s))
        #     # Aiss2 = kappa2 * A * np.sqrt((2*(1-s2))/(s2))
        #     viss1 = Aiss1 * (np.sqrt(D*dnu1))/(freqGHz1*tau1)
        #     # viss2 = Aiss2 * (np.sqrt(D*dnu2))/(freqGHz2*tau2)
        #     Sigma1 = viss1 * np.sqrt((D_err/(2*D))**2+(dnuerr1/(2*dnu1))**2 +
        #                                 (-tauerr1/tau1)**2)
        #     # Sigma2 = viss2 * np.sqrt((D_err/(2*D))**2+(dnuerr2/(2*dnu2))**2 +
        #     #                             (-tauerr2/tau2)**2)
        #     # Sigma = np.concatenate((Sigma1, Sigma2))
        #     # viss = np.concatenate((viss1, viss2))
        #     # # Always keep this
        #     # Residual1 = viss1 - self.func(mjd, **self.model_parameters,
        #     #                               **self.kwargs)[argsort1]
        #     # Residual2 = viss2 - self.func(mjd, **self.model_parameters,
        #     #                               **self.kwargs)[argsort2]
        #     # Residual = np.concatenate((Residual1, Residual2))
        # else:
        params = self.model_parameters
        DNUESKEW, TAUESKEW, DNUEFAC, TAUEFAC, alpha, A, D, kappa, s = map(
            float, [params["DNUESKEW"], params["TAUESKEW"], params["DNUEFAC"],
                    params["TAUEFAC"], params["alpha"], params["A"],
                    params["d"], params["k"], params["s"]])
        freqGHz = self.freq/ 1e3
        dnu = self.dnu
        tau = self.tau
        dnuerr = self.dnuerr
        tauerr = self.tauerr
        DFAC = 10**DNUEFAC
        TFAC = 10**TAUEFAC
        DSKEW = (10**DNUESKEW) * (freqGHz/1)**alpha
        TSKEW = (10**TAUESKEW) * (freqGHz/1)**(alpha/2)
        dnuerr = np.sqrt((dnuerr * DFAC)**2 + DSKEW**2)
        tauerr = np.sqrt((tauerr * TFAC)**2 + TSKEW**2)
        # Measuring viss and visserr
        Aiss = kappa * A * np.sqrt((2*(1-s))/(s))
        viss = Aiss * (np.sqrt(D*dnu))/(freqGHz*tau)
        Sigma = viss * np.sqrt((dnuerr/(2*dnu))**2+(-tauerr/tau)**2)

        Residual = viss - self.func(self.x, **self.model_parameters,
                                    **self.kwargs)
        
        # Alternative method 173294
        # model_range = self.func(self.x, **self.model_parameters, **self.kwargs)
        # step = 0
        # model_mean = []
        # for i in range(0, int(len(self.x)/11)):
        #     model_slice = model_range[step:11+step]
        #     step += 11
        #     # print(np.shape(model_slice))
        #     model_mean.append(np.mean(model_slice))
        #     # print(np.shape(np.mean(model_slice)))
        # # print(np.shape(model_mean))
        # Model_Mean = np.asarray(model_mean)
        # Residual = viss - Model_Mean
        # VISS EFAC EQUAD
        # viss_err = np.sqrt((viss_err * 10**TAUEFAC)**2+(10**TAUESKEW)**2)
        # VISS EFAC ESKEW
        # viss_err = np.sqrt((viss_err * 10**TAUEFAC)**2 +
        #                     ((10**TAUESKEW)*(freqGHz/1)**(alpha))**2)

        # params = self.model_parameters.update(self.kwargs)
        # params = {**self.model_parameters, **self.kwargs}
        # params['derr'] = 0.060
        # print(type(params))
        # Alternative method
        # viss, viss_err = scint_velocity_alternate(params, dnu, tau, freq*1e3,
        #                                           dnuerr, tauerr)
        #
        # Aiss_C_1 = 3.33493134133462e4 for thin screen alpha=5/3 no anisotropy # Jacob
        # Aiss_C_1 = 2.78e4 for thin screen alpha=5/3 no anisotropy # Cordes
        # Alternate errobars
        # dnuerr = self.dnuerr
        # tauerr = self.tauerr
        # Aiss = kappa * 3.33493134133462e4 * np.sqrt((2*(1-s))/(s))
        # viss = Aiss * (np.sqrt(D*dnu))/(freqGHz*tau)
        # viss_err = viss * np.sqrt((D_err/(2*D))**2 +(dnuerr/(2*dnu))**2 +
        #                           (-tauerr/tau)**2)
        # viss_err = np.sqrt((viss_err * 10**TAUEFAC)**2+((10**TAUESKEW) *
        #                                                 (freqGHz/1)**(alpha)
        #                                                 )**2)
        # Alternate method
        #
        # coeff_err = (dnu / s) * ((1 - s) * d_err**2 / (2 * d) +
        #                          (d * s_err**2 / (2 * s**2 * (1 - s))))
        # # viss = coeff * np.sqrt(dnu * d) / (freq * tau)
        # viss_err = (1 / (freq * tau)) * \
        #     np.sqrt(coeff**2 * ((dnuerr**2 / (4 * dnu)) +
        #                         (dnu * tauerr**2 / tau**2)) + coeff_err)
        #
        # Alternate method
        #
        # viss = []
        # viss_err = []
        # for i in range(0, len(dnu)):
        #     d_normal = np.random.normal(loc=float(d), scale=float(derr),
        #                                 size=1000)
        #     dnu_normal = np.random.normal(loc=dnu[i], scale=dnuerr[i],
        #                                   size=1000)
        #     dnu_normal = dnu_normal[dnu_normal > 0]
        #     dnu_normal_sort = np.argsort(dnu_normal)
        #     tau_normal = np.random.normal(loc=tau[i], scale=tauerr[i],
        #                                   size=1000)
        #     tau_normal = tau_normal[dnu_normal_sort]
        #     d_normal = d_normal[dnu_normal_sort]
        #     coeff = 2.78e4 * np.sqrt((2*(1-s))/(s))
        #     viss_normal = \
        #         coeff * (np.sqrt(d_normal*dnu_normal))/(freq[i]*tau_normal)
        #     viss.append(np.median(viss_normal))
        #     viss_err.append(np.std(viss_normal))
        # viss = np.array(viss)
        # viss_err = np.array(viss_err)
        #
        # Alernate method of determining residuals
        # par_dir = '/Users/jacobaskew/Desktop/Swinburne_Daniel/ParFiles/'
        # psrname = 'J0737-3039A'
        # pars = read_par(str(par_dir) + str(psrname) + '.par')
        # params = pars_to_params(pars)
        # mjd = self.x
        # model_mean = []
        # for i in range(0, len(mjd)):
        #     mjd_diff = 5/1440
        #     mjd_range = np.linspace(mjd[i] - mjd_diff, mjd[i] + mjd_diff, 11)
            # U = get_true_anomaly(mjd_range, pars)
            # ve_ra, ve_dec = get_earth_velocity(mjd_range, pars['RAJ'],
            #                                     pars['DECJ'])
        #     kwargs = {"U": U, "ve_ra": ve_ra, "ve_dec": ve_dec,
        #               "params": params}
        #     model_range = self.func(mjd_range, **self.model_parameters,
        #                             **kwargs)
        #     Model = np.asarray(model_range)
        #     model_mean.append(np.mean(Model))
        # Model_Mean = np.asarray(model_mean)
        # Residual = viss - Model_Mean
        # print(Model_Mean)
        # print(self.func(self.x, **self.model_parameters, **self.kwargs))
        # print(type(Model_Mean))
        # print(len(Model_Mean))
        # print()
        # print(type(Residual))
        # print(len(Residual))
        # print()
        #
        #
        # plt.plot(Residual)
        # plt.plot(Residual2)
        # print(Residual)
        # print(Residual2)
        # print(type(self.func(self.x, **self.model_parameters,
        # **self.kwargs)))
        # print(len(self.func(self.x, **self.model_parameters, **self.kwargs)))
        # print()
        # print(type(Residual))
        # print(len(Residual))
        # print()
        # sys.exit("error message")
        log_l = np.sum(- (Residual / Sigma)**2 / 2 - np.log(2 * np.pi * Sigma**2) / 2)
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
