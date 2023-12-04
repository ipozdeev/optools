import numpy as np
from scipy.optimize import fsolve
from scipy import integrate
from scipy.stats import norm

from optools.blackscholes import option_price
from optools.noarbitrage import call_put_parity


def implied_vol_bs(call_price, forward, strike, rf, tau, **kwargs):
    """Compute Black-Scholes implied volatility.

    Inversion of Black-Scholes formula to obtain implied volatility. Saddle
    point is calculated and used as the initial guess.

    Parameters
    ----------
    call_price: numpy.ndarray
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
        val = option_price(forward, strike, rf, tau, x) - call_price

        return val

    # solve with fsolve, use f_prime for gradient
    res = fsolve(func=f_obj, x0=saddle, fprime=f_prime, **kwargs)

    return res


def mfivariance(call_p, strike, forward, rf, tau):
    """Calculate the mfiv as the integral over call prices.

    For details, see Jiang and Tian (2005).

    Parameters
    ----------
    call_p : numpy.ndarray
        of call option prices
    strike : numpy.ndarray
        of strike prices
    forward : float
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
    integrand = (call_p * np.exp(rf * tau) - \
        np.maximum(0, forward - strike)) / \
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
    otm_put_p = call_put_parity(call_p[~otm_call_idx], otm_put_strike,
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
    otm_puts = call_put_parity(call_price=call_p[strike <= spot], strike=strike_put,
                               forward=forward, r_counter=rf, tau=tau)

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


if __name__ == "__main__":
    pass
