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
from scintools.scint_utils import get_true_anomaly, get_earth_velocity, \
    read_par, pars_to_params
import scipy.special as scispec
import mpmath as mp

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

    def __init__(self, x, y, func, **kwargs):
        parameters = infer_parameters_from_function(func)
        super(Analytical1DLikelihood, self).__init__(dict())
        self._func = func
        self.x = x
        self.y = y
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


class AlphaLikelihood(Analytical1DLikelihood):

    def __init__(self, x, y, func, sigma=None, **kwargs):
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

        super(AlphaLikelihood, self).__init__(x=x, y=y, func=func, **kwargs)
        self.sigma = sigma

        # Check if sigma was provided, if not it is a parameter
        if self.sigma is None:
            self.parameters['sigma'] = None

    def log_likelihood(self):
        alpha = float(self.model_parameters["alpha"])
        # EFAC = float(self.model_parameters["EFAC"])
        # EQUAD = float(self.model_parameters["EQUAD"])
        f = self.x / 1000
        nu = self.y
        # Sigma = self.sigma
        sigma = self.sigma
        # sigma = np.sqrt((Sigma*10**EFAC)**2 + (10**EQUAD)**2)
        #
        f_sqr_a = f ** (2 * alpha)
        f_a = f ** alpha
        sigma_sqr = sigma ** 2
        A = np.sum(f_sqr_a / sigma_sqr)
        B = np.sum(f_a * nu / sigma_sqr)
        nu_sqr = nu ** 2
        # exp_term = (B ** 2) / (2 * A)
        # sqrt_term = 1 / (2 * np.sqrt(A))
        #
        # log_l = np.prod(np.exp(-(nu_sqr)/(2 * sigma_sqr)) *
        #                 (1/(np.sqrt(2 * np.pi * sigma_sqr))) *
        #                 exp_term * (sqrt_term) *
        #                 (1 + np.math.erf(-(B)/(np.sqrt(2 * A)))))
        # erf_term = -(B**2)/(2*A) - np.log(B/(2 * np.sqrt(A)) * np.sqrt(np.pi))
        
        erf_val = np.math.erf(-(B)/(np.sqrt(2 * A)))
        
        if erf_val > -1:
            erf_term = np.log(1 + erf_val)
            print("erf_val > -1")
        else:
            # print("erf_val < -1")
            erf_term = -(B**2)/(2*A) - np.log(B/(2 * np.sqrt(A)) * np.sqrt(np.pi))
        
        #erf_term = np.log(1 + np.math.erf(-(B)/(np.sqrt(2 * A))))
        
        
        log_l = np.sum(-(nu_sqr)/(2 * sigma_sqr) - 0.5 * np.log(2 * np.pi *
                                                                sigma_sqr) -
                       (B**2) / (2 * A) + erf_term)
        # print("one", -(nu_sqr)/(2 * sigma_sqr))
        # print("two", 0.5 * np.log(2 * np.pi * sigma_sqr))
        # print("three", (B**2) / (2 * A))
        # print("four", np.log(1 + np.math.erf(-(B)/(np.sqrt(2 * A)))))
        # print("test", -(B)/(np.sqrt(2 * A)))
        # print("test", np.math.erf(-(B)/(np.sqrt(2 * A))))
        
        # print("log_l", log_l)
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
