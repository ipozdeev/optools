# miscellaneous utility functions
import pandas as pd
import numpy as np
from scipy.optimize import fsolve
import lnmix

def fetch_density_quantiles(par, domain=None, perc=None):
    """ For each index in par calculates rn density and quantiles
    Parameters
    ----------
    par: pandas.DataFrame
        of parameters with each row being [mu1, mu2, s1, s2, w1, w2]
    domain: numpy.ndarray-like
        values at which risk-neutral pdf to be calculated
    perc: numpy.ndarray-like
        percentiles of risk-neutral pdf to calculate
    Returns
    -------
    dens: pandas.DataFrame
        of density at points provided in `domain`
    quant: pandas.DataFrame
        of quantiles calculated at `perc`
    """
    if domain is None:
        domain = np.arange(0.8, 1.5, 0.005)
    if perc is None:
        perc = np.array([0.1, 0.5, 0.9])

    # allocate space
    dens = pd.DataFrame(
        data=np.empty(shape=(len(par), len(domain))),
        index=par.index,
        columns=domain)
    quant = pd.DataFrame(
        data=np.empty(shape=(len(par), len(perc))),
        index=par.index,
        columns=perc)

    # loop over rows of par
    for idx, res in par.iterrows():
        # init lognormal_mixture object
        mu = res.values[:2]
        sigma = res.values[2:4]
        wght = res.values[4:]
        ln_mix = lnmix.lognormal_mixture(mu, sigma, wght)

        # density
        p = ln_mix.pdf(domain)

        # issue warning is density integrates to something too far off from 1
        intg = np.trapz(p, domain)
        if abs(intg-1) > 1e-05:
            logger.warning(
                "Large deviation of density integral from one: {:6.5f}".\
                format(intg))

        # quantiles
        q = ln_mix.quantile(perc)

        # store
        dens.loc[idx,:] = p
        quant.loc[idx,:] = q

    return dens, quant
