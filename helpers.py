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