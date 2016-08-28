#
import numpy as np
from scipy.special import erf

def priceUnderMixture(X, K, rf, tau, mu, sigma, w):
    """
    with mu and sigmas in row dimension, strikes in column dimension
    Parameters
    ----------
    X: float
        price of the underlying
    K: numpy.array
        of strike prices
    rf: float
        risk-free rate, in fractions of 1, annualized
    y: float TODO: do we need this???
        dividend yield (foreign risk-free rate), in fractions of 1, annualized
    tau: float
        time to expiry, in years
    mu: numpy.array
        of means of log-normal distributions in the mixture
    sigma: numpy.array
        of st. deviations of log-normal distributions in the mixture
    w: numpy.array
        of component weights
    Returns
    -------
    c: numpy.array
        of call option prices
    p: numpy.array
        of put option prices
    """
    # f =
    # c, p = bsPrice(X, K, rf, y, tau, sigma)
    #
    # return c.dot(w), p.dot(w)
    pass

def blPrice(f, K, rf, tau, sigma):
    """
    Black-Scholes formula interms of forward price. Definitions are from Wystup (2006).
    """

    # d+ and d-
    dplus = (np.log(f/K) + sigma*sigma/2*tau)/(sigma*np.sqrt(tau))
    dminus = dplus - sigma*np.sqrt(tau)

    # phi = 1 for calls, -1 for puts
    res = list()
    for phi in [1, -1]:
        res.append(phi*np.exp(-rf*tau)*(
            f*fastNormCdf(phi*dplus) -
            K*fastNormCdf(phi*dminus)))

    # return
    return res[0], res[1]

def fastNormCdf(x):
    """
    Using error function from scipy.special (o(1) faster)
    """
    return(1/2*(1 + erf(x/np.sqrt(2))))
