import numpy as np
from scipy.stats import norm
from scipy.optimize import fsolve

from optools.helpers import fast_norm_cdf
from optools.blackscholes import d1, d2


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


def strike_from_delta(delta, tau, vol, is_call, spot=None, forward=None,
                      rf=None, div_yield=None, is_forward: bool = False,
                      is_premiumadj: bool = False) -> np.ndarray:
    """Calculate strike price given delta.

    Everything relevant is annualized. Details in Clark (2011).

    Parameters
    ----------
    delta: float or numpy.ndarray or str
        of option deltas, in (frac of 1), or one of ('atmf', 'atms', 'dns')
    spot: float
        underlying price
    forward : float
    rf: float
        risk-free rate, in (frac of 1) p.a.
    div_yield: float
        dividend yield, in (frac of 1) p.a.
    tau: float
        time to maturity, in years
    vol: float or numpy.ndarray
        implied vol
    is_call: bool
        whether options are call options
    is_forward : bool
        if delta is forward delta (dV/df)
    is_premiumadj : bool
        if delta is pips or percentage

    Return
    ------
    k: float or numpy.ndarray
        of strike prices
    """
    # +1 for calls, -1 for puts
    omega = is_call*2 - 1.0

    # function to calculate delta given strike and the rest
    if is_forward:
        if is_premiumadj:
            def delta_fun(strike):
                res_ = omega * strike / forward * \
                    fast_norm_cdf(omega * d2(forward, strike, vol, tau))
                return res_
        else:
            def delta_fun(strike):
                res_ = omega * \
                    fast_norm_cdf(omega * d1(forward, strike, vol, tau))
                return res_
    else:
        if is_premiumadj:
            def delta_fun(strike):
                res_ = omega * np.exp(-rf * tau) * strike / spot * \
                    fast_norm_cdf(omega * d2(forward, strike, vol, tau))
                return res_
        else:
            def delta_fun(strike):
                res_ = omega * np.exp(-div_yield * tau) * \
                    fast_norm_cdf(omega * d1(forward, strike, vol, tau))
                return res_

    def obj_fun(strike):
        return delta_fun(strike) - delta

    # solve with fsolve, use f_prime for gradient
    x0 = forward if forward is not None else spot
    if hasattr(delta, "__iter__"):
        x0 = np.array([x0, ] * len(delta))

    res = fsolve(func=obj_fun, x0=x0)

    return res