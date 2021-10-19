import pandas as pd
import numpy as np
from optools.helpers import fast_norm_cdf
from scipy.stats import norm
from scipy.optimize import fsolve
from scipy import integrate


def d1(forward, strike, vol, tau) -> isinstance(float, np.ndarray):
    """Black-Scholes d1

    Parameters
    ----------
    forward : float or np.ndarray
    strike : float or np.ndarray
    vol : float or np.ndarray
    tau : float

    """
    res = (np.log(forward / strike) + 0.5 * vol ** 2 * tau) / \
          (vol * np.sqrt(tau))

    return res


def d2(forward, strike, vol, tau) -> isinstance(float, np.ndarray):
    """Black-Scholes d2

    Parameters
    ----------
    forward : float or np.ndarray
    strike : float or np.ndarray
    vol : float or np.ndarray
    tau : float

    """
    res = (np.log(forward / strike) - 0.5 * vol ** 2 * tau) / \
          (vol * np.sqrt(tau))

    return res


def bs_price(strike, rf, tau, vol, div_yield=None, spot=None, forward=None,
             is_call=True):
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
    vol : float or numpy.ndarray
        volatility, in (frac of 1) p.a.
    div_yield : float
        dividend yield
    spot : float
        spot price of the underlying
    forward : float
        forward price of the underlying
    is_call : bool
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
        (forward * fast_norm_cdf(omega * d1(forward, strike, vol, tau)) -
         strike * fast_norm_cdf(omega * d2(forward, strike, vol, tau)))

    return res


def call_to_put(call_p, strike, forward, rf, tau):
    """Calculate the put price from the put-call parity relation.

    Vectorized for call_p, strike

    Parameters
    ----------
    call_p : float or numpy.array
        call price
    strike : float or numpy.array
        strike price
    forward : float
        forward price of the underlying
    rf : float
        risk-free rate, in (frac of 1) p.a.
    tau : float
        maturity, in years

    Returns
    -------
    p : float or numpy.array
        put price
    """
    p = call_p - forward * np.exp(-rf * tau) + strike * np.exp(-rf * tau)

    return p


def bs_iv(call_p, forward, strike, rf, tau, **kwargs):
    """Compute Black-Scholes implied volatility.

    Inversion of Black-Scholes formula to obtain implied volatility. Saddle
    point is calculated and used as the initial guess.

    Parameters
    ----------
    call_p: numpy.ndarray
        (M,) array of fitted option prices (IVs)
    forward: float
        forward price of underlying
    strike: numpy.ndarray
        (M,) array of strike prices
    rf: float
        risk-free rate, (in frac of 1) per period tau
    tau: float
        time to maturity, in years
    **kwargs: dict
        other arguments to fsolve

    Return
    ------
    res: numpy.ndarray
        (M,) array of implied volatilities
    """
    # lower part is dplus
    def f_prime(x):
        """Derivative of bs_price, or vega."""
        # forward*e^{-rf*tau} is the same as S*e^{-y*tau}
        x_1 = forward * np.exp(-rf * tau) * np.sqrt(tau)
        x_2 = norm.pdf(
            (np.log(forward / strike) + x * x / 2 * tau) / (x * np.sqrt(tau)))

        val = np.diag(x_1 * x_2)

        return val

    # saddle point (Wystup (2006), p. 19)
    saddle = np.sqrt(2 / tau * np.abs(np.log(forward / strike)))

    # make sure it is positive, else set it next to 0
    saddle *= 0.9
    saddle[saddle <= 0] = 0.1

    # objective
    def f_obj(x):
        """Objective function: call minus target call."""
        val = bs_price(forward, strike, rf, tau, x) - call_p

        return val

    # solve with fsolve, use f_prime for gradient
    res = fsolve(func=f_obj, x0=saddle, fprime=f_prime, **kwargs)

    return res


def bs_vega(forward, strike, y, tau, sigma):
    """Compute Black-Scholes vega as in Wystup (2006)

    For each strike in `K` and associated `sigma` computes sensitivity of
    option to changes in volatility.

    Parameters
    ----------
    forward: float
        forward price of the underlying
    strike: numpy.ndarray
        of strike prices
    y: float
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
    vega = forward * np.exp(-y * tau) * np.sqrt(tau) * norm.pdf(dplus)

    return vega


def vanillas_from_combinations(r, b, atm_vol, delta=None):
    """Calculate implied vola of calls from that of put/call combinations.

    See Wystup, p.24.

    Parameters
    ----------
    atm_vol : float
        implied vola of the at-the-money option
    r : float
        implied vola of the risk reversal
    b : float
        implied vola of the butterfly contract
    delta : float
        delta, in ((frac of 1)), e.g. 0.25 or 0.10


    Returns
    -------
    res : list or pandas.Series
        of implied volas; pandasSeries indexed by delta if `delta` was provided

    """
    # implied volas
    two_ivs = np.array([
        atm_vol + b + 0.5 * r,
        atm_vol + b - 0.5 * r
    ])

    # if delta was not supplied, return list
    if delta is None:
        return two_ivs

    # deltas
    two_deltas = [delta, 1 - delta]

    # create a Series
    res = pd.Series(index=two_deltas, data=two_ivs).rename("iv")
    res.index.name = "delta"

    return res


def strike_from_delta(delta, spot, forward, rf, div_yield, tau, vol, is_call,
                      is_forward: bool = False,
                      is_premiumadjusted: bool = False) -> float:
    """Calculate strike prices given delta.

    Everything relevant is annualized. Details in Clark (2011).

    Calculates the strike price of an option given its delta and specific
    identifying parameters.

    Parameters
    ----------
    delta: float or numpy.ndarray or str
        of option deltas, in (frac of 1), or one of ('atmf', 'atms', 'dns')
    spot: float
        underlying price
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

    Return
    ------
    k: float or numpy.ndarray
        of strike prices
    """
    # +1 for calls, -1 for puts
    omega = is_call*2 - 1.0

    # atm case
    if delta == "atmf":
        return forward
    elif delta == "atms":
        return spot
    elif delta == "dns":
        if is_premiumadjusted:
            return forward * np.exp(-0.5 * vol ** 2 * tau)
        else:
            return forward * np.exp(0.5 * vol ** 2 * tau)

    # function to calculate delta given strike and the rest
    if is_forward:
        if is_premiumadjusted:
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
        if is_premiumadjusted:
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
    res = fsolve(func=obj_fun, x0=spot)[0]

    return res


def mfivariance(call_p, strike, forward_p, rf, tau):
    """Calculate the mfiv as the integral over call prices.

    For details, see Jiang and Tian (2005).

    Parameters
    ----------
    call_p : numpy.ndarray
        of call option prices
    strike : numpy.ndarray
        of strike prices
    forward_p : float
        forward price
    rf : float
        risk-free rate, in (frac of 1) p.a.
    tau : float
        maturity, in years

    Returns
    -------
    res : float
        mfiv, in (frac of 1) p.a.

    """
    # integrate
    integrand = (call_p * np.exp(rf * tau) -\
        np.maximum(np.zeros(shape=(len(call_p), )), forward_p-strike)) /\
        (strike * strike)

    res = integrate.simps(integrand, strike) * 2

    # annualize
    res /= tau

    return res


def simple_var_swap_rate(call_p, strike, forward_p, rf, tau):
    """Calculate simple variance swap rate as in Martin (2017).

    Parameters
    ----------
    call_p : numpy.ndarray
    strike : numpy.ndarray
    forward_p : float
    rf : float
    tau : float
        maturity, in years

    Returns
    -------
    res : float
        swap rate, annualized

    """
    # split into otm calls and puts
    otm_call_idx = strike >= forward_p

    # otm calls
    otm_call_p = call_p[otm_call_idx]
    otm_call_strike = strike[otm_call_idx]

    # convert itm calls to puts
    otm_put_strike = strike[~otm_call_idx]
    otm_put_p = call_to_put(call_p[~otm_call_idx], otm_put_strike,
                            forward_p, rf, tau)

    # integrate
    res = \
        integrate.simps(otm_put_p, otm_put_strike) + \
        integrate.simps(otm_call_p, otm_call_strike)

    res *= 2 * np.exp(rf * tau) / forward_p**2 / tau

    return res


def mfiskewness(call_p, strike, spot, forward, rf, tau):
    """Calculate the MFIskewness.

    For details, see Bakshi et al. (2003).

    Parameters
    ----------
    call_p : numpy.ndarray
        of call prices
    strike : numpy.ndarray
        of strike prices
    spot : float
        spot price of the underlying
    forward : float
        spot price of the underlying
    rf : float
        risk-free rate, in (frac of 1) p.a.
    tau : float
        maturity, in years

    Returns
    -------
    res : float
        model-free implied skewness

    """
    # put-call parity: call for for strike > spot and put for strike <= spot
    strike_put = strike[strike <= spot]
    otm_puts = call_to_put(call_p=call_p[strike <= spot], strike=strike_put,
                           forward=forward, rf=rf, tau=tau)

    strike_call = strike[strike > spot]
    otm_calls = call_p[strike > spot]

    # cubic contract
    c_cube = (6*np.log(strike_call/spot) - 3*np.log(strike_call/spot)**2) / \
        strike_call**2 * otm_calls
    p_cube = (6*np.log(spot/strike_put) + 3*np.log(spot/strike_put)**2) / \
        strike_put**2 * otm_puts
    cube = integrate.simps(c_cube, strike_call) - \
        integrate.simps(p_cube, strike_put)

    # quadratic contract
    mfiv = mfivariance(call_p, strike, forward, rf, tau)

    # quartic contract
    c_quart = (12*np.log(strike_call/spot)**2
               - 4*np.log(strike_call/spot)**3) / strike_call**2 * otm_calls
    p_quart = (12*np.log(spot/strike_put)**2
              + 4*np.log(spot/strike_put)**3) / strike_put**2 * otm_puts
    quart = integrate.simps(c_quart, strike_call) + \
        integrate.simps(p_quart, strike_put)

    # mu
    mu = np.exp(rf*tau) - 1 - np.exp(rf*tau) / 2 * mfiv - \
         np.exp(rf*tau)/6 * cube - np.exp(rf*tau)/24 * quart

    # all together
    res = (np.exp(rf*tau)*cube - 3*mu*np.exp(rf*tau)*mfiv + 2*mu**3) /\
        (np.exp(rf*tau)*mfiv - mu**2) ** (3/2)

    return res


def fill_by_no_arb(spot, forward, rf, div_yield, tau, raise_errors=False):
    """Fill one missing value using the no-arbitrage relation.

    Parameters
    ----------
    spot : float
    forward : float
    rf : float
        in (frac of 1) p.a.
    div_yield : float
        in (frac of 1) p.a.
    tau : float
        maturity, in years
    raise_errors : bool
        True to raise errors when more than two argument are missing

    Returns
    -------
    args : dict

    """
    # collect all arguments
    args = {"spot": spot, "forward": forward, "rf": rf, "div_yield": div_yield}

    # find one nan
    where_nan = {k: v for k, v in args.items() if np.isnan(v)}

    if len(where_nan) > 1:
        if raise_errors:
            raise ValueError("Only one argument can be missing!")
        else:
            return args
    if len(where_nan) < 1:
        return args

    # tau needs to be set
    if (tau is None) | np.isnan(tau):
        raise ValueError("Maturity not provided!")

    k, v = list(where_nan.items())[0]

    # no-arb relationships
    if k == "spot":
        args["spot"] = forward / np.exp((rf - div_yield) * tau)
    elif k == "forward":
        args["forward"] = spot * np.exp((rf - div_yield) * tau)
    elif k == "rf":
        args["rf"] = np.log(forward / spot) / tau + div_yield
    elif k == "div_yield":
        args["div_yield"] = rf - np.log(forward / spot) / tau

    return args


def foreign_domestic_symmetry(option_price_ab, strike_ab, spot_ab) -> tuple:
    """Implement the foreign-domestic symmetry relation of currency options.

    Given the price of a XXX-denominated call option w/ strike of K units of
    XXX per YYY, computes the price of a YYY-denominated put option w/
    strike of 1/K units of YYY per XXX.

    Parameters
    ----------
    option_price_ab : float or numpy.ndarray or pandas.Series
        of prices (in currency b) of options to transact 1 unit of currency a
    strike_ab : float or numpy.ndarray or pandas.Series
    spot_ab : float

    """
    option_price_ba = option_price_ab / strike_ab / spot_ab

    strike_ba = 1 / strike_ab

    return option_price_ba, strike_ba


if __name__ == "__main__":
    strike_from_delta(0.25, 1.3465, 1.3395, 0.0294, 0.0346, 1, 0.192, True,
                      is_forward=False, is_premiumadjusted=False)
