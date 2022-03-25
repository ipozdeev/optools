import numpy as np
from scipy.optimize import fsolve

from optools.blackscholes import d2, d1
from optools.helpers import fast_norm_cdf


def strike_from_delta(delta, vola, tau, is_call, spot=None, forward=None,
                      r_counter=None, r_base=None, is_forward: bool = False,
                      is_premiumadj: bool = False) -> np.ndarray:
    """Calculate strike price given delta.

    Everything relevant is annualized. Details in Clark (2011).

    Parameters
    ----------
    delta: float or numpy.ndarray or str
        of option deltas, in (frac of 1), or one of ('atmf', 'atms', 'dns')
    vola: float or numpy.ndarray
        implied vol
    tau: float
        time to maturity, in years
    is_call: bool or np.ndarray
        whether options are call options
    spot: float
        underlying price
    forward : float
    r_counter: float
        risk-free rate, in frac. of 1 p.a.
    r_base: float
        dividend yield, in frac. of 1 p.a.
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
                    fast_norm_cdf(omega * d2(forward, strike, vola, tau))
                return res_
        else:
            def delta_fun(strike):
                res_ = omega * \
                    fast_norm_cdf(omega * d1(forward, strike, vola, tau))
                return res_
    else:
        if is_premiumadj:
            def delta_fun(strike):
                res_ = omega * np.exp(-r_counter * tau) * strike / spot * \
                       fast_norm_cdf(omega * d2(forward, strike, vola, tau))
                return res_
        else:
            def delta_fun(strike):
                res_ = omega * np.exp(-r_base * tau) * \
                       fast_norm_cdf(omega * d1(forward, strike, vola, tau))
                return res_

    def obj_fun(strike):
        return delta_fun(strike) - delta

    # solve with fsolve, use f_prime for gradient
    x0 = forward if forward is not None else spot

    if hasattr(delta, "__iter__"):
        x0 = np.array([x0, ] * len(delta))

    res = fsolve(func=obj_fun, x0=x0)

    return res


def strike_from_atm(atm_def, is_premiumadj=None, spot=None, forward=None,
                    vola=None, tau=None) -> float:
    """Calculate strike of an FX at-the-money contract.

    There can be many possible scenarios, captured by `atm_def` and
    `is_premiumadj`:
        - atm forward
        - atm spot
        - strike equal to that of a delta-neutral straddle
            - delta being premium-adjusted or not

    Based on the foreign exchange optin pricing book by Clark (2011).

    Parameters
    ----------
    atm_def : str
        definition of at-the-money
    is_premiumadj : bool
        must be specified if `atm_def` is True
    spot : float
    forward : float
    vola : float
        in frac of 1 p.a.
    tau : float
        maturity, in years

    """
    if atm_def == "atmf":
        return forward
    elif atm_def == "spot":
        return spot
    elif atm_def == "dns":
        if is_premiumadj is None:
            raise ValueError("`is_premiumadj` must be set if "
                             "`atm_def == 'dns'")
        if is_premiumadj:
            return forward * np.exp(-0.5 * vola ** 2 * tau)
        else:
            return forward * np.exp(+0.5 * vola ** 2 * tau)
    else:
        raise ValueError("unknown `atm_def`!")
