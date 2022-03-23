import numpy as np
from optools.helpers import fast_norm_cdf

from typing import Union


def d1(forward, strike, vol, tau) -> isinstance(float, np.ndarray):
    """Black-Scholes d1 (vectorized).

    Parameters
    ----------
    forward : float or np.ndarray
    strike : float or np.ndarray
    vol : float or np.ndarray
        implied volatility, in frac of 1 p.a.
    tau : float
        maturity, in years

    """
    res = (np.log(forward / strike) + 0.5 * vol ** 2 * tau) / \
          (vol * np.sqrt(tau))

    return res


def d2(forward, strike, vol, tau) -> isinstance(float, np.ndarray):
    """Black-Scholes d2 (vectorized).

    Parameters
    ----------
    forward : float or np.ndarray
    strike : float or np.ndarray
    vol : float or np.ndarray
        implied volatility, in frac of 1 p.a.
    tau : float
        maturity, in years

    """
    res = (np.log(forward / strike) - 0.5 * vol ** 2 * tau) / \
          (vol * np.sqrt(tau))

    return res


def option_price(strike, rf, tau, vola, div_yield=None, spot=None, forward=None,
                 is_call=True) -> Union[float, np.ndarray]:
    """Compute the Black-Scholes option price.

    Vectorized for `strike` and `vola`. Definitions are as in Wystup (2006).

    Parameters
    ----------
    strike : float or numpy.ndarray
        strikes prices
    rf : float
        risk-free rate, in (frac of 1) p.a.
    tau : float
        maturity, in years
    vola : float or numpy.ndarray
        volatility, in (frac of 1) p.a.
    div_yield : float
        dividend yield
    spot : float
        spot price of the underlying
    forward : float
        forward price of the underlying
    is_call : bool or np.ndarray
        True (False) to return prices of call (put) options

    Returns
    -------
    res : float or numpy.ndarray
        price, in domestic currency
    """
    # +1 for call, -1 for put
    omega = is_call * 2 - 1.0

    if forward is None:
        try:
            forward = spot * np.exp((rf - div_yield)*tau)
        except TypeError:
            raise TypeError("Make sure to provide rf, div_yield and spot!")

    res = omega * np.exp(-rf * tau) * \
        (forward * fast_norm_cdf(omega * d1(forward, strike, vola, tau)) -
         strike * fast_norm_cdf(omega * d2(forward, strike, vola, tau)))

    return res