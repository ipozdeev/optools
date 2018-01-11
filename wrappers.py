import pandas as pd
from scipy import integrate
import optools.functions as op_func
import re
from optools.volsurface import VolatilitySmile
import numpy as np
import warnings

# module with global configurations
# from optools import config
# from optools import lnmix
# import multiprocessing as mproc


def wrapper_mfiv_from_series(series, interpolation_kwargs, estimation_kwargs):
    """Wrapper.

    Parameters
    ----------
    series : pandas.Series
        indexed with
        - spot
        - forward
        - different risk reversals and butterflies, labeled '[0-9]+(rr|bf)'
        - rf
        - div_yield
        - tau
        - atm_vola

    Returns
    -------
    res : scalar
        mfiv, not annualized, in ((frac of 1))^2

    """
    # no-arbitrage relationships --------------------------------------------
    no_arb_dict = dict(
        series.loc[["spot", "forward", "rf", "div_yield", "tau"]])

    no_arb_series = series.update(
        pd.Series(op_func.fill_by_no_arb(**no_arb_dict)))

    # time to drop na (most likely in the combinations) ---------------------
    no_arb_series = no_arb_series.dropna()

    # find combinations: these have to start with digits --------------------
    combies_regex = re.compile("([0-9]+)[a-z]{2}")
    deltas_str = list(set(combies_regex.findall(no_arb_series.index)))

    # dictionary to comply with .by_delta_from_combinations; .pop ensures
    #   that all combies are deleted from the series, and it can be used as
    #   **kwargs later
    combies = dict()
    for d in deltas_str:
        for p in ["rr", "bf"]:
            combies[float(d)/100] = no_arb_series.pop(d+p)

    for _, v in combies.items():
        v.index = v.index.map(lambda x: re.sub("[0-9]+", '', x))

    # vol smile -------------------------------------------------------------
    smile = VolatilitySmile.by_delta_from_combinations(
        combies=combies, is_call=True, **no_arb_series)

    new_strike = np.linspace(min(smile.strike)*2/3, max(smile.strike)*3/2, 200)
    smile_interp = smile.interpolate(new_strike=new_strike,
                                     **interpolation_kwargs)

    res = smile_interp.get_mfiv(**estimation_kwargs)

    return res


def mfiskew_wrapper(iv_surf, forward_p, rf, tau, spot_p, method="spline"):
    """Wrapper.

    """
    # min and max strikes
    k_min = np.min(iv_surf["K"])
    k_max = np.max(iv_surf["K"])

    # interpolate ivs
    iv_interp = op_func.interpolate_iv(iv_surf, method=method,
        x_pred=pd.DataFrame(data=np.linspace(k_min, k_max, 50)))

    # extend beyond limits: constant fashion
    dk = np.diff(iv_interp.index).mean()
    new_idx_left = np.arange(0.66 * forward_p, k_min, dk)
    new_idx_right = np.arange(k_max + dk, forward_p * 1.5, dk)
    new_idx = np.concatenate((new_idx_left, iv_interp.index, new_idx_right))
    iv_interp = iv_interp.reindex(index=new_idx)
    iv_interp.fillna(method="ffill", inplace=True)
    iv_interp.fillna(method="bfill", inplace=True)

    # back to prices
    c_interp = op_func.bs_price(forward_p, iv_interp.index, rf, tau,
                                iv_interp.values)

    # put-call parity: C for for K > spot_p and P for K <= spot_p
    K_P = iv_interp.index[iv_interp.index <= spot_p]
    P = op_func.call_to_put(
        call_p=c_interp[iv_interp.index <= spot_p],
        strike=K_P,
        forward_p=forward_p,
        rf=rf,
        tau=tau)

    K_C = iv_interp.index[iv_interp.index > spot_p]
    C = c_interp[iv_interp.index > spot_p]

    # cubic contract
    yC = (6 * np.log(K_C / spot_p) - 3 * np.log(K_C / spot_p) ** 2) /\
        K_C**2 * C
    yP = (6 * np.log(spot_p / K_P) + 3 * np.log(spot_p / K_P) ** 2) /\
        K_P**2 * P
    W = integrate.simps(yC, K_C) - integrate.simps(yP, K_P)

    # quadratic contract
    V = mfiv_wrapper(iv_surf, forward_p, rf, tau, method)

    # quartic contract
    yC = (12 * np.log(K_C / spot_p) ** 2 - 4 * np.log(K_C / spot_p)**3) /\
        K_C**2 * C
    yP = (12 * np.log(spot_p / K_P) ** 2 + 4 * np.log(spot_p / K_P)**3) /\
        K_P**2 * P
    X = integrate.simps(yC, K_C) + integrate.simps(yP, K_P)

    # mu
    mu = np.exp(rf*tau) - 1 - np.exp(rf*tau)/2*V - np.exp(rf*tau)/6*W -\
        np.exp(rf*tau)/24*X

    # all together
    mfiskew = (np.exp(rf*tau)*W - 3*mu*np.exp(rf*tau)*V + 2*mu**3)/\
        (np.exp(rf*tau)*V - mu**2)**(3/2)

    return mfiskew


def wrapper_implied_co_mat(variances, ccur):
    """ Calculate covariance and correlation of currencies w.r.t a common one.

    Parameters
    ----------
    variances : pandas.Series
        of variances, labeled with currency pairs
    ccur : str
        counter currency
    """
    # all pairs
    pairs_all = list(variances.index)
    # pairs not containing "usd"
    pairs_nonusd = [p for p in pairs_all if ccur not in p]
    # pairs containing "usd"
    pairs_usd = [p for p in pairs_all if ccur in p]
    # currencies except usd
    currencies = [p.replace(ccur, '') for p in pairs_usd]
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
            reverse_sign = xxx_vs_usd.find(ccur) == yyy_vs_usd.find(ccur)
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
    """ Calculate covariance and correlation implied by three variances.

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
    co_r = co_v/np.sqrt(varAB*varBC)*(1 if reverse_sign else -1)

    return co_v, co_r


def wrapper_beta_from_covmat(covmat, wght, exclude_self=False):
    """ Estimates beta of a number of assets w.r.t. their linear combination.

    Parameters
    ----------
    covmat : pandas.DataFrame
        covariance matrix
    wght : pandas.Series
        weights of each asset in the linear combination
    zero_cost : boolean
        True if the portfolio defined with `wght` is zero-cost (sum(wght)=0)

    Returns
    -------
    B : pandas.Series
        of calculated betas (ordering corresponds to columns of `covmat`)
    """
    # wght = pd.Series(data=np.ones(8), index=covmat.columns)
    # # new weight
    # if zero_cost:
    #     new_wght = wght[covmat_trim.columns] - wght[covmat_trim.columns].sum()
    # else:
    #     new_wght = wght[covmat_trim.columns]/wght[covmat_trim.columns].sum()

    # do the computations
    if exclude_self:
        B = pd.Series(index=wght.index)
        D = pd.Series(index=wght.index)
        for c in wght.index:
            tmp_wght = wght.copy()
            tmp_wght.loc[c] = 0.0
            tmp_wght = normalize_weights(tmp_wght)
            this_num = covmat.dot(tmp_wght)
            denominator = tmp_wght.dot(covmat.dot(tmp_wght))
            B.loc[c] = this_num.loc[c]/denominator
            D.loc[c] = denominator
    else:
        numerator = covmat.dot(wght)
        D = wght.dot(covmat.dot(wght))
        B = numerator/D

    # reindex back
    B = B.reindex(covmat.columns)

    return B, D


def normalize_weights(wght):
    """
    """
    if (wght == 0).all():
        res = wght
    if (wght < 0).any() & (wght > 0).any():
        short_leg = wght.where(wght < 0)
        long_leg = wght.where(wght >= 0)

        short_leg = short_leg / np.abs(short_leg).sum()
        long_leg = long_leg / np.abs(long_leg).sum()

        res = short_leg.fillna(long_leg)
    else:
        res = wght / np.abs(wght).sum()

    return res


def wrapper_beta_of_portfolio(covmat, wght_p, wght_m):
    """ Estimates beta of a number of assets w.r.t. their linear combination.

    TODO: fix this description

    Parameters
    ----------
    covmat : pandas.DataFrame
        covariance matrix
    wght : pandas.Series
        weights of each asset in the linear combination

    Returns
    -------
    B : pandas.Series
        of calculated betas (ordering corresponds to columns of `covmat`)
    """
    # wght = pd.Series(data=np.ones(8), index=covmat.columns)
    # trim nans in a smart way
    covmat_trim = covmat.copy()
    # init count of nans
    nan_count_total = pd.isnull(covmat_trim).sum().sum()
    # while there are nans in covmat, remove columns with max number of nans
    while nan_count_total > 0:
        # detect rows where number of nans is less than maximum
        nan_max = pd.isnull(covmat_trim).sum()
        nan_max_idx = max([(p,q) for q,p in enumerate(nan_max)])[1]

        # nan_max_idx = pd.isnull(covmat_trim).sum() < \
        #     max(pd.isnull(covmat_trim).sum())
        # covmat_trim = covmat_trim.ix[nan_max_idx,nan_max_idx]

        covmat_trim.drop(covmat_trim.columns[nan_max_idx],axis=0,inplace=True)
        covmat_trim.drop(covmat_trim.columns[nan_max_idx],axis=1,inplace=True)

        # new count of nans
        nan_count_total = pd.isnull(covmat_trim).sum().sum()

    # new weight
    new_wght_m = wght_m[covmat_trim.columns]/wght_m[covmat_trim.columns].sum()
    new_wght_p = wght_p.reindex(index=covmat_trim.columns,fill_value=0.0)
    new_wght_p /= new_wght_p.sum()

    # do the computations
    numerator = new_wght_p.dot(covmat_trim.dot(new_wght_m))
    denominator = new_wght_m.dot(covmat_trim.dot(new_wght_m))
    B = numerator/denominator

    return B


# def aux_fun(x):
#     """run_one_row to use in multiprocessing
#     """
#     return run_one_row(x, **config.cfg_dict)
#
#
# def wrapper_density_from_par(par, domain=None):
#     """ Calculate density over `domain`.
#
#     Parameters
#     ----------
#     par : (2*K,) numpy.ndarray
#         of parameters of K components ([mu1, mu2, s1, s2, w1, w2] for K=2)
#     domain: numpy.ndarray-like
#         values at which risk-neutral pdf to be calculated
#
#     Returns
#     -------
#     density : pandas.Series
#         of density at points provided in `domain`
#     """
#     # break par into mu, sigma and weigh
#     mu = par[:2]
#     sigma = par[2:4]
#     wght = par[4:]
#
#     # init lognormal mixture object
#     ln_mix = lnmix.lognormal_mixture(mu, sigma, wght)
#
#     # if not provided, density is Ex plus/minus 5std
#     if domain is None:
#         E_x, Var_x = ln_mix.moments()
#         domain = np.linspace(E_x-10*np.sqrt(Var_x), E_x+10*np.sqrt(Var_x), 250)
#
#     # calculate density
#     p = ln_mix.pdf(domain)
#
#     # issue warning is density integrates to something too far off from 1
#     intg = np.trapz(p, domain)
#     if abs(intg-1) > 1e-05:
#         warnings.warn(
#             "Large deviation of density integral from one: {:6.5f}".\
#             format(intg))
#
#     # to pandas.Series
#     p = pd.Series(data=p, index=domain)
#
#     return p
#
#
# def wrapper_quantiles_from_par(par, prob=None):
#     """ Calculate quantiles at each of `prob`.
#
#     Parameters
#     ----------
#     par : (2*K,) numpy.ndarray
#         of parameters of K components ([mu1, mu2, s1, s2, w1, w2] for K=2)
#     prob : (M,) numpy.ndarray
#         probabilities at which quantiles to be calculated
#
#     Returns
#     -------
#     q : pandas.Series
#         of quantiles calculated at `prob`, with q.columns = `prob`
#     """
#     # if not provided, take the usual suspects
#     if prob is None:
#         prob = np.array([0.1, 0.5, 0.9])
#
#     # break par into mu, sigma and weigh
#     mu = par[:2]
#     sigma = par[2:4]
#     wght = par[4:]
#
#     # init lognormal mixture object
#     ln_mix = lnmix.lognormal_mixture(mu, sigma, wght)
#
#     # quantiles
#     q = ln_mix.quantile(prob)
#
#     # to pandas.Series
#     q = pd.Series(data=q, index=prob)
#
#     return q
#
#
# def wrapper_variance_of_logx(par):
#     """ Calculate variance of log(x) when x ~ log-normal mixture.
#
#     Parameters
#     ----------
#     par : (2*K,) numpy.ndarray
#         of parameters of K components ([mu1, mu2, s1, s2, w1, w2] for K=2)
#
#     Returns
#     -------
#     Var_x : float
#         variance of log(x)
#
#     """
#     # break par into mu, sigma and weight
#     mu = par[:2]
#     sigma = par[2:4]
#     wght = par[4:]
#
#     # init lognormal mixture object
#     ln_mix = lnmix.lognormal_mixture(mu, sigma, wght)
#
#     # estimate moments
#     _, Var_x = ln_mix.moments_of_log()
#
#     return Var_x
#
#
# def estimation_wrapper(data, tau, parallel=False):
#     """ Loop over rows in `data` to estimate RND of the underlying.
#
#     Allow for parallel and standard (loop) regime. If parallel, uses
#     multiprocessing (does not work from IPython).
#
#     TODO: relocate to wrappers
#
#     Parameters
#     ----------
#     data: pandas.DataFrame
#         with rows for 5 option contracts, spot price, forward price and two
#         risk-free rates called rf for domestic and y for foreign
#     tau: float
#         maturity of options, in frac of year
#     parallel: boolean
#         whether to parallelize or not
#
#     Returns
#     -------
#     par: pandas.DataFrame
#         of estimated parameters: [mu', sigma', w']
#
#     """
#     # delete bad rows
#     data.dropna(inplace=True)
#
#     if parallel:
#         # call to multiprocessing; create pool with this many processes
#         n_proc = 4
#         pool = mproc.Pool(n_proc)
#
#         # iterator is (conveniently) pd.iterrows
#         iterator = data.iterrows()
#
#         # function to apply is run_one_row, but it should be picklable,
#         # hence the story with it being located right above
#         out = list(zip(*pool.map(aux_fun, iterator, n_proc)))
#
#         # output is (index, data) -> need to unpack & name
#         par = pd.DataFrame(np.array([*out[1]]), index=out[0],
#             columns=["mu1", "mu2", "sigma1", "sigma2", "w1", "w2"])
#
#     else:
#         # allocate space for parameters (6 of them now)
#         par = pd.DataFrame(
#             data=np.empty(shape=(len(data), 6)),
#             index=data.index,
#             columns=["mu1", "mu2", "sigma1", "sigma2", "w1", "w2"])
#         # estimate in a loop
#         for idx_row in data.iterrows():
#             # estimate rnd for this index; unpack the dict in config to get
#             # hold of constraints and bounds
#             # pdb.set_trace()
#             idx, res = run_one_row(idx_row, **config.cfg_dict)
#             # store
#             par.loc[idx, :] = res
#
#     return par
#
#
# def run_one_row(idx_row, tau, opt_meth, constraints):
#     """ Wrapper around estimate_rnd working on (idx, row) from pandas.iterrows
#     TODO: relocate to wrappers.py
#     """
#     # unpack the tuple
#     idx = idx_row[0]
#     row = idx_row[1]
#
#     logger.info("doing row %.10s..." % str(idx))
#     logger.debug(("This row:\n"+"%.4f "*len(row)+"\n") % tuple(row.values))
#
#     # fetch wings
#     deltas, ivs = get_wings(
#         row["rr25d"],
#         row["rr10d"],
#         row["bf25d"],
#         row["bf10d"],
#         row["atm"],
#         row["y"],
#         tau)
#
#     logger.debug(("IVs:\n"+"%.2f "*len(ivs)+"\n") % tuple(ivs*100))
#     logger.debug(("Deltas:\n"+"%.2f "*len(deltas)+"\n") % tuple(deltas))
#
#     # to strikes
#     K = strike_from_delta(
#         deltas,
#         row["s"],
#         row["rf"],
#         row["y"],
#         tau,
#         ivs,
#         True)
#
#     logger.debug(("K:\n"+"%.2f "*len(K)+"\n") % tuple(K))
#
#     # weighting matrix: inverse squared vegas
#     W = bs_vega(
#         row["f"],
#         K,
#         row["rf"],
#         row["y"],
#         tau,
#         ivs)
#     W = np.diag(1/(W*W))
#
#     logger.debug(("Vegas:\n"+"%.2f "*len(np.diag(W))+"\n") %
#         tuple(np.diag(W)))
#
#     # estimate rnd!
#     res = estimate_rnd(
#         ivs*np.sqrt(tau),
#         row["f"],
#         K,
#         row["rf"]*tau,
#         True,
#         W,
#         opt_meth=opt_meth,
#         constraints=constraints)
#
#     return idx, res


# def wrapper_rnd_nonparametric(day_panel, s, maturity, h=None):
#     """
#
#     Parameters
#     ----------
#     maturity : float
#         in years
#     """
#     rf = day_panel.loc[maturity,:,"rf_base"].mean()
#     maturity = misc.maturity_from_string(maturity)
#
#     # stack everything together ---------------------------------------------
#     df_stacked = pd.DataFrame(columns=["iv", "f", "K", "tau"])
#
#     # within-day loop over maturities
#     for tau_str, df in day_panel.iteritems():
#
#         # get maturity in years
#         tau = misc.maturity_from_string(tau_str)
#
#         # loop over time stamps for each maturity
#         for time_idx, row in df.iterrows():
#             # get deltas and ivs
#             deltas, ivs = op.get_wings(
#                 row["rr25d"],row["rr10d"],row["bf25d"],row["bf10d"],row["atm"],
#                 row["rf_counter"], tau)
#             # transform deltas to strikes
#             strikes = op.strike_from_delta(delta=deltas,
#                 X=s.loc[time_idx], rf=row["rf_base"], y=row["rf_counter"],
#                 tau=tau, sigma=ivs, is_call=True)
#             # store everything
#             tmp_df = pd.DataFrame.from_dict(
#                 {
#                     "iv" : ivs,
#                     "f" : np.ones(5)*row["f"],
#                     "K" : strikes,
#                     "tau" : np.ones(5)*tau
#                 })
#             # merge with df_stacked
#             df_stacked = pd.concat((df_stacked, tmp_df), axis=0,
#                 ignore_index=True)
#
#     # collect response and regressors ---------------------------------------
#     y = df_stacked["iv"]
#     X = df_stacked.drop(["iv",], axis=1)
#
#     # prepare values at which predictions to be made
#     # strikes are equally spaced [min(K), max(K)]
#     dK = 1e-05
#     K_pred = np.arange(min(df_stacked["K"]), max(df_stacked["K"]), dK)
#     # forward is mean forward price over that day
#     f_pred = np.ones(len(K_pred))*df_stacked["f"].mean()
#     # maturity
#     tau_pred = np.ones(len(K_pred))*maturity
#     # all together
#     X_pred = np.stack((K_pred, f_pred, tau_pred), axis=1)
#
#     # estimate
#     res = op.rnd_nonparametric(y, X, X_pred, rf, maturity, is_iv=True, h=h,
#         f=f_pred, K=K_pred)
#
#     return res
