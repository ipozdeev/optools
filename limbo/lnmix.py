# mixture of n log-normals
import pandas as pd
import numpy as np
from scipy.special import erf
from scipy.stats import norm
from scipy.optimize import fsolve

class lognormal_mixture():
    """ Mixture of log-normals

    Parameters
    ----------
    mu: numpy.ndarray
        of means of normal variables that underlie log-normals
    sigma: numpy.ndarray
        of st. dev's of normal variables that underlie log-normals
    wght: numpy.ndarray
        of weights of each component
    """
    def __init__(self, mu, sigma, wght):
        """
        """
        assert all([len(p) == len(wght) for p in [mu, sigma]])

        self.mu = mu
        self.sigma = sigma
        self.wght = wght

    def moments(self):
        """ Compute mean and variance of X ~ lnmix
        """
        iE_x = np.exp(self.mu+self.sigma*self.sigma/2)
        iE_x2 = (np.exp(self.sigma**2)-1)*np.exp(2*self.mu+self.sigma**2) + \
            iE_x**2
        E_x = iE_x.dot(self.wght)
        Var_x = iE_x2.dot(self.wght) - E_x**2

        return E_x, Var_x

    def moments_of_log(self):
        """ Compute mean and variance of ln(X) when X ~ lnmix
        """
        iE_x2 = self.sigma**2 + self.mu**2
        E_x = self.mu.dot(self.wght)
        Var_x = iE_x2.dot(self.wght) - E_x**2

        return E_x, Var_x

    def pdf(self, x):
        """ Compute PDF of mixture of log-normals at points in x

        Parameters
        ----------
        x: numpy.ndarray
            of points

        Return
        ------
        p: numpy.ndarray
            of probability densities

        """
        flag = False
        if not (type(x) is np.array):
            flag = True
            x = np.array([x,])

        # dimensions
        M = len(self.wght)
        N = len(x)

        # broadcast
        mu = np.array([self.mu,]*N).transpose()
        sigma = np.array([self.sigma,]*N).transpose()
        x = np.array([x,]*M)

        # densities
        arg = (np.log(x) - mu)/sigma
        d = 1/(x*sigma*np.sqrt(2*np.pi))*np.exp(-arg*arg/2)

        # weighted average
        d = self.wght.dot(d)

        return(d[0] if flag else d)

    def cdf(self, x):
        """ Compute CDF of mixture of log-normals
        """
        flag = False
        if not (type(x) is np.ndarray):
            flag = True
            x = np.array([x,])

        # dimensions
        M = len(self.wght)
        N = len(x)

        # broadcast
        mu = np.array([self.mu,]*N).transpose()
        sigma = np.array([self.sigma,]*N).transpose()
        x = np.array([x,]*M)

        p = 0.5*(1+erf((np.log(x)-mu)/(np.sqrt(2)*sigma)))
        p = self.wght.dot(p)

        return(p[0] if flag else p)

    def quantile(self, p):
        """ Compute quantiles of mixture of log-normals

        Parameters
        ----------
        p : float or (M,) numpy.ndarray
            probability at which quantile to be calculated

        Returns
        -------
        q : float or (M,) numpy.ndarray
            of quantiles at probabilities in `p`
        """
        flag = False
        if not (type(p) is np.ndarray):
            flag = True
            p = np.array([p,])

        # find root of CDF(x)=p
        obj_fun = lambda x: self.cdf(x) - p

        # starting value: exp{quantiles of single normal} TODO: think about it
        x0 = np.exp(norm.ppf(p,
            loc=self.mu.dot(self.wght),
            scale=self.sigma.dot(self.wght)))

        # fprime is pdf
        fprime = lambda x: np.diag(self.pdf(x))

        # solve
        q = fsolve(func=obj_fun, x0=x0, fprime=fprime)

        return(q[0] if flag else q)
