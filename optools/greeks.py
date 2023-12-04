import numpy as np
from scipy.stats import norm

from optools.blackscholes import d1, d2
from optools.helpers import fast_norm_cdf


def delta_pips_forward(forward, strike, vola, tau, is_call):
    omega = is_call*2 - 1
    res = omega * fast_norm_cdf(omega * d1(forward, strike, vola, tau))
    return res


def delta_premiumadj_forward(forward, strike, vola, tau, is_call):
    omega = is_call*2 - 1
    res = omega * strike / forward * \
        fast_norm_cdf(omega * d2(forward, strike, vola, tau))
    return res


def delta_pips_spot(forward, strike, vola, r_base, tau, is_call):
    omega = is_call*2 - 1
    res = omega * np.exp(-r_base * tau) * \
        fast_norm_cdf(omega * d1(forward, strike, vola, tau))
    return res


def delta_premiumadj_spot(forward, spot, strike, vola, r_counter, tau,
                          is_call):
    omega = is_call*2 - 1
    res = omega * np.exp(-r_counter * tau) * strike / spot * \
        fast_norm_cdf(omega * d2(forward, strike, vola, tau))
    return res


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
