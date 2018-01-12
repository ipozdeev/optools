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
    if mat.endswith('m'):
        res = int(mat[:-1]) / 12.0
    elif mat.endswith('y'):
        res = float(mat[:-1])
    else:
        raise ValueError("Maturity string must end with 'm' or 'y'!")

    return res