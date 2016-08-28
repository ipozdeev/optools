#
import numpy as np
from scipy.special import erf
from scipy.stats import norm
from scipy.optimize import fsolve

def lossWrapper(par, wght, K, rf, cTrue, fTrue, is_iv, W):
    """
    """
    # decompose par into mu and sigma
    mu = par[1,:]
    sigma = par[2,:]

    # fitted values
    cHat = priceUnderMixture(K, rf, mu, sigma, wght)
    fHat = np.dot(wght, np.exp(mu + 0.5*sigma*sigma))

    # if implied_vol
    if is_iv:
        cHat = bsIV()


def lossFun(cTrue, cHat, fTrue, fHat, is_iv = True, W = None):
    """
    Loss function.
    Parameters
    ----------
    cTrue: numpy.array
        of true call prices/ivs
    """
    if W is None:
        W = np.eye(len(cHat))

    cDev = cTrue - cHat
    fDev = np.log(fTrue/fHat) if is_iv else (fTrue - fHat)

    loss = 1e04*(np.dot(cDev, np.dot(W, cDev)) + fDev*fDev)

    return loss

def bsIV(cHat, f, K, rf, tau):
    """
    """
    # fprime: derivative of bsPrice, or vega

    # fprime: f*e^{-rf*tau} is the same as S*e^{-y*tau}
    # lower part is dplus
    fPrime = lambda x: np.diag(f*np.exp(-rf*tau)*np.sqrt(tau)*norm.pdf(
        (np.log(f/K) + x*x/2*tau)/(x*np.sqrt(tau))))

    # fPrime(sigma)

    # saddle point (Wystup (2006), p. 19)
    saddle = np.sqrt(2/tau * np.abs(np.log(f/K)))

    # make sure it is positive, else set it next to 0
    saddle *= 0.9
    saddle[saddle <= 0] = 0.1

    # solve with fsolve
    res = fsolve(func = lambda x: bsIVobjective(cHat, f, K, rf, tau, x),
        x0 = saddle, fprime = fPrime)

    return res

def bsIVobjective(cHat, f, K, rf, tau, sigma):
    """
    """
    c = bsPrice(f, K, rf, tau, sigma) - cHat

    return c

    # return c.dot(np.eye(len(sigma)).dot(c))

def priceUnderMixture(K, rf, mu, sigma, wght):
    """
    Computes the price of a call option under assumption that the underlying follows a mixture of lognormal distributions with parameters specified in 3mu* and *sigma* and weights specified in *w*.
    Parameters
    ----------
    K: numpy.array
        of strike prices
    rf: float
        risk-free rate, in fractions of 1, per period
    mu: numpy.array
        of means of log-normal distributions in the mixture
    sigma: numpy.array
        of st. deviations of log-normal distributions in the mixture
    wght: numpy.array
        of component weights
    Returns
    -------
    c: numpy.array
        of call option prices
    p: numpy.array
        of put option prices
    """
    # tile with K a matrix with rows for distributional components
    K = np.array([K,]*len(K))

    # tile with mu and sigma matrices with columns for strikes
    mu = np.array([mu,]*len(K)).transpose()
    # %timeit sigma = np.tile(sigma[np.newaxis].T, (1, len(K)))
    sigma = np.array([sigma,]*len(K)).transpose()

    # calculate forward price based on distributional assumptions
    f = np.exp(mu + 0.5*sigma*sigma)

    # finally, call prices
    c = bsPrice(f, K, rf, tau = 1, sigma = sigma)

    return wght.dot(c)

def bsPrice(f, K, rf, tau, sigma):
    """
    Black-Scholes formula in terms of forward price *f*. Definitions are from Wystup (2006).
    """

    # d+ and d-
    dplus = (np.log(f/K) + sigma*sigma/2*tau)/(sigma*np.sqrt(tau))
    dminus = dplus - sigma*np.sqrt(tau)

    res = np.exp(-rf*tau)*(f*fastNormCdf(dplus) - K*fastNormCdf(dminus))

    # return
    return res

def fastNormCdf(x):
    """
    Using error function from scipy.special (o(1) faster)
    """
    return(1/2*(1 + erf(x/np.sqrt(2))))
