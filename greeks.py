import numpy as np
from scipy.stats import norm


def vega(forward, strike, r_base, tau, sigma):
    """Compute Black-Scholes vega as in Wystup (2006)

    For each strike in `K` and associated `sigma` computes sensitivity of
    option to changes in volatility.

    Parameters
    ----------
    forward: float
        forward price of the underlying
    strike: numpy.ndarray
        of strike prices
    r_base: float
        risk-free rate in the base currency, continuously comp.,
        in frac. of 1 p.a.
    tau: float
        time to maturity, in years
    sigma: numpy.ndarray
        implied vola, in (frac of 1) p.a.

    Returns
    -------
    vega: numpy.ndarray
        vegas
    """
    dplus = (np.log(forward / strike) + sigma ** 2 / 2 * tau) / \
            (sigma * np.sqrt(tau))
    vega = forward * np.exp(-r_base * tau) * np.sqrt(tau) * norm.pdf(dplus)

    return vega
