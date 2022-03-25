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
        risk-free rate, aka rate in the counter currency, in frac. of 1 p.a.
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


def covered_interest_parity(tau, spot=np.nan, forward=np.nan, r_counter=np.nan,
                            r_base=np.nan, raise_errors=False) -> dict:
    """Fill one missing value using the no-arbitrage relation.

    Parameters
    ----------
    spot : float
    forward : float
    r_counter : float
        risk-free rate in the counter currency, cont.comp., in frac. of 1 p.a.
    r_base : float
        risk-free rate in the base currency, cont.comp., in frac. of 1 p.a.
    tau : float
        maturity, in years
    raise_errors : bool
        True to raise errors when more than two argument are missing

    Returns
    -------
    args : dict

    """
    # collect all arguments
    args = {"spot": spot, "forward": forward, "rf": r_counter, "div_yield": r_base}

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
        args["spot"] = forward / np.exp((r_counter - r_base) * tau)
    elif k == "forward":
        args["forward"] = spot * np.exp((r_counter - r_base) * tau)
    elif k == "rf":
        args["rf"] = np.log(forward / spot) / tau + r_base
    elif k == "div_yield":
        args["div_yield"] = r_counter - np.log(forward / spot) / tau

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
    two_ivs = [
        atm_vol + b + 0.5 * r,
        atm_vol + b - 0.5 * r
    ]

    # if delta was not supplied, return list
    if delta is None:
        return two_ivs
    else:
        # deltas
        two_deltas = [delta, 1 - delta]

        return dict(zip(two_deltas, two_ivs))
