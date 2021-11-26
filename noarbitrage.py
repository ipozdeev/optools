import numpy as np


def call_put_parity(strike, forward, rf, tau, call_price=None, put_price=None):
    """Calculate the put price from the put-call parity relation.

    Vectorized for call_p, strike

    Parameters
    ----------
    strike : float or numpy.array
        strike price
    forward : float
        forward price of the underlying
    rf : float
        risk-free rate, in (frac of 1) p.a.
    tau : float
        maturity, in years
    call_price : float or numpy.array
    put_price : float or numpy.array

    Returns
    -------
    res : float or numpy.array
        put price
    """
    assert (call_price is not None) | (put_price is not None)

    if put_price is None:
        res = call_price - forward * np.exp(-rf * tau) + \
              strike * np.exp(-rf * tau)
    else:
        res = put_price + forward * np.exp(-rf * tau) - \
              strike * np.exp(-rf * tau)

    return res


def covered_interest_parity(tau, spot=np.nan, forward=np.nan, rf=np.nan,
                            div_yield=np.nan, raise_errors=False) -> dict:
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
    where_none = {k: v for k, v in args.items() if np.isnan(v)}

    if len(where_none) > 1:
        if raise_errors:
            raise ValueError("Only one argument can be missing!")
        else:
            return args
    if len(where_none) < 1:
        return args

    k, v = list(where_none.items())[0]

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

