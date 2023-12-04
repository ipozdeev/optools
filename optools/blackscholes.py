import numpy as np
from optools.helpers import fast_norm_cdf

from typing import Union


def d1(forward, strike, vola, tau) -> Union[float, np.ndarray]:
    """Black-Scholes d1 (vectorized).

    Parameters
    ----------
    forward : float or np.ndarray
    strike : float or np.ndarray
    vola : float or np.ndarray
        implied volatility, in frac of 1 p.a.
    tau : float
        maturity, in years

    """
    res = (np.log(forward / strike) + 0.5 * vola ** 2 * tau) / \
          (vola * np.sqrt(tau))

    return res


def d2(forward, strike, vola, tau) -> Union[float, np.ndarray]:
    """Black-Scholes d2 (vectorized).

    Parameters
    ----------
    forward : float or np.ndarray
    strike : float or np.ndarray
    vola : float or np.ndarray
        implied volatility, in frac of 1 p.a.
    tau : float
        maturity, in years

    """
    res = (np.log(forward / strike) - 0.5 * vola ** 2 * tau) / \
          (vola * np.sqrt(tau))

    return res


def option_price(strike, vola, forward, r_counter, tau, is_call) \
        -> Union[float, np.ndarray]:
    """Compute the Black-Scholes option price.

    Vectorized for `strike` and `vola` to return output of the same
    dimension as (strike * vola): e.g. if strike.shape == (1, 4) and
    vola.shape == (2, 1), res.dim == (2, 4)

    Parameters
    ----------
    strike : float or numpy.ndarray
        strikes prices
    vola : float or numpy.ndarray
        volatility, in (frac of 1) p.a.
    forward : float
        forward price of the underlying
    r_counter : float
        risk-free rate in the counter currency (continuously comp), in frac.
        of 1 p.a.
    tau : float
        maturity, in years
    is_call : bool or np.ndarray
        True (False) to return prices of call (put) options

    Returns
    -------
    res : float or numpy.ndarray
        price, in domestic currency
    """
    # +1 for call, -1 for put
    omega = is_call * 2 - 1.0

    res = omega * np.exp(-r_counter * tau) * (
        forward * fast_norm_cdf(omega * d1(forward, strike, vola, tau)) -
        strike * fast_norm_cdf(omega * d2(forward, strike, vola, tau))
    )

    return res