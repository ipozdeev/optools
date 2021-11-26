import pandas as pd
from pandas.tseries.offsets import BDay
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


def maturity_float_to_str(mat):
    """

    Parameters
    ----------
    mat

    Returns
    -------

    """
    map_dict = {"1m": 22, "2m": 44, "3m": 64, "4m": 84, "5m": 104,
                "6m": 126, "7m": 148, "8m": 170, "9m": 190, "10m": 212,
                "11m": 232, "12m": 254}

    res = map_dict[mat]

    return res


def maturity_str_to_float(mat, to_freq='Y'):
    """Convert maturity to fractions of a period.

    Parameters
    ----------
    mat: str
    to_freq : str
        character, pandas frequency

    Returns
    -------
    res : float

    """
    if (to_freq.upper() == 'B') and (mat[-1].upper() == 'M'):
        map_dict = {"1m": 22, "2m": 44, "3m": 64, "4m": 84, "5m": 104,
                    "6m": 126, "7m": 148, "8m": 170, "9m": 190, "10m": 212,
                    "11m": 232, "12m": 254}
        return map_dict[mat]

    scale_matrix = pd.DataFrame(
        index=['D', 'B', 'Y'],
        columns=['W', 'M', 'Y'],
        data=np.array([[1/7, 1/30, 1/365],
                       [1/5, 1/22, 1/254],
                       [52, 12, 1]], dtype=float))

    int_part = int(mat[:-1])
    scale = scale_matrix.loc[to_freq, mat[-1].upper()]

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


def construct_new_x(old_x, x_min=None, x_max=None, step=None) -> np.ndarray:
    """Construct a new range of values to be used in smile interpolation.

    Parameters
    ----------
    old_x : array-like
    x_min : float
        minimum value of the new x
    x_max : float
        maximum value of the new x
    step : float
        increment
    """
    # range
    x_range = np.ptp(old_x)

    # min, max
    if x_min is None:
        x_min = max(min(old_x) / 2, min(old_x) - x_range * 2)
    if x_max is None:
        x_max = max(old_x) + x_range * 2
    if step is None:
        step = np.diff(old_x).mean() / 500

    res = np.arange(x_min, x_max, step)

    # # reindex, assign a socialistic name; this will be sorted!
    # res = np.union1d(old_x, res).astype(np.float)

    return res


def ndays_from_dateoffset(t, dateoffset):
    """Calculate the no of business days from the next day onwards."""
    res = len(pd.date_range(t, BDay().rollforward(t + dateoffset),
                            freq='B')) - 1

    return res