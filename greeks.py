import numpy as np
from scipy.stats import norm


def vega(forward, strike, div_yield, tau, sigma):
    """Compute Black-Scholes vega as in Wystup (2006)

    For each strike in `K` and associated `sigma` computes sensitivity of
    option to changes in volatility.

    Parameters
    ----------
    forward: float
        forward price of the underlying
    strike: numpy.ndarray
        of strike prices
    div_yield: float
        dividend yield (foreign interest rate)
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
    vega = forward * np.exp(-div_yield * tau) * np.sqrt(tau) * norm.pdf(dplus)

    return vega
