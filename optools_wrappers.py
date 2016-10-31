# wrappers
import pandas as pd
import numpy as np
import lnmix
import warnings
import misc

def wrapper_density_from_par(par, domain=None):
    """ Calculate density over `domain`.

    Parameters
    ----------
    par : (2*K,) numpy.ndarray
        of parameters of K components ([mu1, mu2, s1, s2, w1, w2] for K=2)
    domain: numpy.ndarray-like
        values at which risk-neutral pdf to be calculated

    Returns
    -------
    density : pandas.Series
        of density at points provided in `domain`
    """
    # break par into mu, sigma and weigh
    mu = par[:2]
    sigma = par[2:4]
    wght = par[4:]

    # init lognormal mixture object
    ln_mix = lnmix.lognormal_mixture(mu, sigma, wght)

    # if not provided, density is Ex plus/minus 5std
    if domain is None:
        E_x, Var_x = ln_mix.moments()
        domain = np.linspace(E_x-10*np.sqrt(Var_x), E_x+10*np.sqrt(Var_x), 250)

    # calculate density
    p = ln_mix.pdf(domain)

    # issue warning is density integrates to something too far off from 1
    intg = np.trapz(p, domain)
    if abs(intg-1) > 1e-05:
        warnings.warn(
            "Large deviation of density integral from one: {:6.5f}".\
            format(intg))

    # to pandas.Series
    p = pd.Series(data=p, index=domain)

    return p

def wrapper_quantiles_from_par(par, prob=None):
    """ Calculate quantiles at each of `prob`.

    Parameters
    ----------
    par : (2*K,) numpy.ndarray
        of parameters of K components ([mu1, mu2, s1, s2, w1, w2] for K=2)
    prob : (M,) numpy.ndarray
        probabilities at which quantiles to be calculated

    Returns
    -------
    q : pandas.Series
        of quantiles calculated at `prob`, with q.columns = `prob`
    """
    # if not provided, take the usual suspects
    if prob is None:
        prob = np.array([0.1, 0.5, 0.9])

    # break par into mu, sigma and weigh
    mu = par[:2]
    sigma = par[2:4]
    wght = par[4:]

    # init lognormal mixture object
    ln_mix = lnmix.lognormal_mixture(mu, sigma, wght)

    # quantiles
    q = ln_mix.quantile(prob)

    # to pandas.Series
    q = pd.Series(data=q, index=prob)

    return q

def wrapper_variance_of_logx(par):
    """ Calculate variance of log(x) when x ~ log-normal mixture.

    Parameters
    ----------
    par : (2*K,) numpy.ndarray
        of parameters of K components ([mu1, mu2, s1, s2, w1, w2] for K=2)

    Returns
    -------
    Var_x : float
        variance of log(x)

    """
    # break par into mu, sigma and weight
    mu = par[:2]
    sigma = par[2:4]
    wght = par[4:]

    # init lognormal mixture object
    ln_mix = lnmix.lognormal_mixture(mu, sigma, wght)

    # estimate moments
    _, Var_x = ln_mix.moments_of_log()

    return Var_x

def wrapper_implied_co_mat(variances):
    """ Calculate covariance and correlation of currencies w.r.t a common one.

    Parameters
    ----------
    variances : pandas.Series
        of variances, labeled with currency pairs
    """
    # all pairs
    pairs_all = list(variances.index)
    # pairs not containing "usd"
    pairs_nonusd = [p for p in pairs_all if "usd" not in p]
    # pairs containing "usd"
    pairs_usd = [p for p in pairs_all if "usd" in p]
    # currencies except usd
    currencies = [p.replace("usd", '') for p in pairs_usd]
    N = len(currencies)

    covmat = pd.DataFrame(
        data=np.empty(shape=(N,N)),
        index=currencies,
        columns=currencies)
    cormat = pd.DataFrame(
        data=np.diag(np.ones(N)),
        index=currencies,
        columns=currencies)

    # loop over currency indices (except usd, of course)
    for cur1 in range(N):
        # cur1 = 1
        # get its name
        xxx = currencies[cur1]
        # find "xxxusd" or "usdxxx"
        xxx_vs_usd = next(p for p in pairs_usd if xxx in p)
        # set corresponding diagonal element of `covmat` to variance of
        #   xxx_vs_usd
        covmat.loc[xxx,xxx] = variances[xxx_vs_usd]
        # loop over the other currency indices
        for cur2 in range(cur1+1,N):
            # cur2 = 4
            # get its name
            yyy = currencies[cur2]
            # find "xxxyyy" or "yyyxxx"
            xxx_vs_yyy = xxx+yyy if xxx+yyy in pairs_nonusd else yyy+xxx
            # find "yyyusd" or "usdyyy"
            yyy_vs_usd = next(p for p in pairs_usd if yyy in p)
            # if "usd" parts are not aligned
            reverse_sign = xxx_vs_usd.find("usd") == yyy_vs_usd.find("usd")
            # calculate covariance
            cov_q, cor_q = wrapper_implied_co(
                varAC=variances[xxx_vs_yyy],
                varAB=variances[xxx_vs_usd],
                varBC=variances[yyy_vs_usd],
                reverse_sign=reverse_sign)
            # store it
            covmat.loc[xxx,yyy] = cov_q
            covmat.loc[yyy,xxx] = cov_q
            cormat.loc[xxx,yyy] = cor_q
            cormat.loc[yyy,xxx] = cor_q

    # raise warning if matrix is not positive definite
    tmp_covmat = \
        covmat.ix[np.isfinite(covmat).all(), np.isfinite(covmat).all()]
    try:
        d = np.linalg.det(tmp_covmat*10000)
        if not d > 0:
            warnings.warn("Matrix is not positive definite" + \
                " with det = {}".format(round(d, 1)))
    except np.linalg.LinAlgError:
        warnings.warn("Too many missing values")

    return covmat, cormat

def wrapper_implied_co(varAC, varAB, varBC, reverse_sign):
    """ Calculate covaraince and correlation implied by three variances.

    Builds on var[AC] = var[AB] + var[BC] + 2*cov[AB,BC] if
    AC = AB + BC to extract covariance and correlation between AB and BC.

    Parameters
    ----------
    varAC: (1,) float
        variance of currency pair consisting of base currency A and counter
        currency C (units of C for one unit of A)
    varAB: (1,) float
        variance of currency pair consisting of base currency A and counter
        currency B (units of B for one unit of A)
    varBC: (1,) float
        variance of currency pair consisting of base currency B and counter
        currency C (units of C for one unit of B)
    reverse_sign: (1,) boolean
        True if instead of _one_ of addends its reciprocal is provided such
        that +2*cov changes to -2*cov

    Returns
    -------
    co_v: float
        implied covariance
    co_r: float
        implied correlation
    """

    co_v = (varAC - varAB - varBC)/2*(-1 if reverse_sign else 1)
    co_r = co_v/np.sqrt(varAB*varBC)

    return co_v, co_r

def wrapper_beta_from_covmat(covmat, wght):
    """ Estimates beta of a number of assets w.r.t. their linear combination.

    Parameters
    ----------
    covmat : (N,N) numpy.ndarray or pandas.DataFrame
        covariance matrix
    wght : (N,) numpy.ndarray or pandas.Series
        weights of each asset in the linear combination

    Returns
    -------
    B : (N,) numpy.ndarray or pandas.Series
        of calculated betas (ordering corresponds to columns of `covmat`)
    """
    # wght = pd.Series(data=np.ones(8), index=covmat.columns)
    # find rows/columns void of values
    good_idx = np.isfinite(covmat).any(axis=0)
    # different indexing if pandas objects or numpy arrays
    if hasattr(covmat, "columns"):
        # get rid of columns/rows with no values at all
        new_covmat = covmat.ix[good_idx,good_idx]
        new_wght = wght.ix[good_idx]/wght.ix[good_idx].sum()

        # on the rest, do the computations
        numerator = new_covmat.dot(new_wght)
        denominator = new_wght.dot(new_covmat.dot(new_wght))
        B = numerator/denominator

        # reindex
        B = B.reindex(covmat.columns)
    else:
        B = np.empty(shape=len(wght))*np.nan
        # get rid of columns/rows with no values at all
        new_covmat = covmat[good_idx,good_idx]
        new_wght = wght[good_idx]/wght[good_idx].sum()

        # on the rest, do the computations
        numerator = new_covmat.dot(new_wght)
        denominator = new_wght.dot(new_covmat.dot(new_wght))
        B[good_idx] = numerator/denominator

    return B
