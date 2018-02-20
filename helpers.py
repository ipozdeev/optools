import numpy as np
from scipy.special import erf


def fast_norm_cdf(x):
    """Calculate normal cdf.

    Parameters
    ----------
    x : numpy.ndarray

    Uses error function from scipy.special (o(1) faster)
    """
    res = 1/2 * (1 + erf(x / np.sqrt(2)))

    return res


def maturity_str_to_float(mat):
    """

    Parameters
    ----------
    mat: str

    Returns
    -------

    """
    int_part = int(mat[:-1])

    if mat.endswith('m'):
        scale = 12.0
    elif mat.endswith('y'):
        scale = 1.0
    elif mat.endswith('w'):
        scale = 52.0
    else:
        raise ValueError("Maturity string must end with 'm', 'y' or 'w'!")

    res = int_part / scale

    return res


def disc_to_cont(rate, tau):
    """Transfrom a dicrete rate to continuously compounded one.

    Only 30/360 convention is implemented.

    Parameters
    ----------
    rate : float or numpy.ndarray-like
        rate, in (frac of 1) p.a.
    tau : float
        maturity, in years

    Returns
    -------
    res : float or numpy.ndarray
        of continuously compounded rates

    """
    res = np.log(1 + rate*tau) / tau

    return res


def strike_range(strike, k_min=None, k_max=None, step=None):
    """

    Parameters
    ----------
    strike
    k_min
    k_max
    step

    Returns
    -------

    """
    # range
    strike_rng = np.ptp(strike)

    # min, max
    if k_min is None:
        k_min = max(strike_rng / 2, min(strike) - strike_rng * 2)
    if k_max is None:
        k_max = max(strike) + strike_rng * 2
    if step is None:
        step = strike_rng / 200

    strike_new = np.arange(k_min, k_max, step)

    # reindex, assign a socialistic name; this will be sorted!
    res = np.union1d(strike, strike_new).astype(np.float)

    return res
