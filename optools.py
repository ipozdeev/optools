#
import pandas as pd
import numpy as np
from scipy.special import erf
from scipy.stats import norm
from scipy.optimize import fsolve, minimize, differential_evolution
from scipy import interpolate, integrate
from statsmodels.nonparametric.kernel_regression import KernelReg
import multiprocessing as mproc
import sys

# from ip_econometrics import RegressionModel as regm
# import ipdb

# module with global configurations
from optools import config

# logging module
import logging
logger = logging.getLogger()

def aux_fun(x):
    """ run_one_row to use in multiprocessing
    """
    return(run_one_row(x, **config.cfg_dict))

def mfiv_wrapper(iv_surf, f, rf, tau, method="spline"):
    """
    Returns
    -------
    res : scalar
        mfiv, not annualized, in (fractions of 1)^2
    """
    # min and max strikes
    K_min = np.min(iv_surf["K"])
    K_max = np.max(iv_surf["K"])

    # interpolate ivs
    iv_interp = interpolate_iv(iv_surf, method=method)

    # extend beyond limits: constant fashion
    dK = np.diff(iv_interp.index).mean()
    new_idx_left = np.arange(0.66*f, K_min, dK)
    new_idx_right = np.arange(K_max+dK, f*1.5, dK)
    new_idx = np.concatenate((new_idx_left, iv_interp.index, new_idx_right))
    iv_interp = iv_interp.reindex(index=new_idx)
    iv_interp.fillna(method="ffill", inplace=True)
    iv_interp.fillna(method="bfill", inplace=True)

    # back to prices
    C_interp = bs_price(f, iv_interp.index, rf, tau, iv_interp.values)

    # integrate
    y = (C_interp*np.exp(rf*tau) - \
        np.maximum(np.zeros(shape=(len(C_interp),)), f-new_idx))/\
            (new_idx*new_idx)
    res = integrate.simps(y, new_idx)*2

    return res

def interpolate_iv(iv_surf, method="spline", X_pred=None):
    """ Interpolate iv surface.
    Spline interpolation is only supported for 2D data: iv ~ strikes.
    Kernel regression interpolation is supported for 2D and 3D data:
    iv ~ strikes + tau.
    Parameters
    ----------
    iv_surf : pandas.DataFrame
        with columns "K" for strikes, "iv" for implied volatilities, and
        (optional) "tau" for maturity, in years.
    method : str
        "spline" or "kernel".
    X_pred : pandas.DataFrame
        with the same columns as `iv_surf` except for "iv".

    Returns
    -------
    rse : pd.Series
        indexed with new strikes and containing interpolated iv
    """
    # sort values: needed for spline to work properly
    iv_surf = iv_surf.sort_values(by="K", axis=0)

    # response is iv
    endog = iv_surf["iv"].values

    # if K_pred missing, use linspace over range of provided K;
    # also, everything should be lists (for kernel regression to work)
    if X_pred is None:
        X_pred = [np.linspace(np.min(iv_surf["K"]), np.max(iv_surf["K"]),
            np.ceil(np.ptp(iv_surf["K"])/0.01)), ]
    else:
        # turn columns of df into list elements
        X_pred = list(X_pred.values.T)

    # var_type for kernel regression: see KernelReg for details
    var_type = ['c',] + (['u',] if iv_surf.shape[1] > 2 else [])

    # regressors, as a list
    exog = list(iv_surf.drop(["iv",], axis=1).values.T)

    # if iv_surf.shape[1] > 2:
    #     if tau is None:
    #         raise NameError("Maturity of interpolated series not provided")
    #     else:
    #         # make list out of provided maturity
    #         exog += [iv_surf["tau"].values, ]
    #         X_pred += [np.array([tau,]*len(K_pred)),]
    #     var_type += ['u',]

    # method
    if method == "spline":
        if iv_surf.shape[1] > 2:
            raise NotImplementedError("Not possible")
            # # convert to ndarrays
            # exog_df = pd.DataFrame(index=np.unique(exog[0]),
            #     columns=np.unique(exog[1]), dtype=np.float)*np.nan
            # for p in range(len(exog[0])):
            #     exog_df.loc[exog[0][p], exog[1][p]] = endog[p]
            #
            # endog_mat = exog_df.values
            #
            # x, y = np.meshgrid(exog_df.index, exog_df.columns, indexing="ij")
            # tck = interpolate.bisplrep(x, y, endog_mat, s=0)
            #
            # print(X_pred[0])
            # # xnew, ynew = np.meshgrid(X_pred[0], X_pred[1], indexing="ij")
            # # print(xnew)
            # # print(ynew)
            # znew = interpolate.bisplev(X_pred[0], [tau,], tck)
            # print(znew)
        else:
            # estimate iv ~ strikes
            tck = interpolate.splrep(exog[0], endog, s=0)
            # predict
            znew = interpolate.splev(X_pred[0], tck, der=0)

    elif method == "kernel":
        # estimate endog must be a list of one elt
        kr = KernelReg(endog=[endog,], exog=exog, var_type=var_type,
            reg_type="ll")
        # fit
        znew, _ = kr.fit(data_predict=X_pred)

    # return, squeeze jsut in case
    res = pd.Series(index=X_pred[0], data=znew.squeeze())

    return res

def mfiskew_wrapper(iv_surf, f, rf, tau, S, method="spline"):
    """
    """
    # min and max strikes
    K_min = np.min(iv_surf["K"])
    K_max = np.max(iv_surf["K"])

    # interpolate ivs
    iv_interp = interpolate_iv(iv_surf, method=method,
        X_pred=pd.DataFrame(data=np.linspace(K_min,K_max,50)))

    # extend beyond limits: constant fashion
    dK = np.diff(iv_interp.index).mean()
    new_idx_left = np.arange(0.66*f, K_min, dK)
    new_idx_right = np.arange(K_max+dK, f*1.5, dK)
    new_idx = np.concatenate((new_idx_left, iv_interp.index, new_idx_right))
    iv_interp = iv_interp.reindex(index=new_idx)
    iv_interp.fillna(method="ffill", inplace=True)
    iv_interp.fillna(method="bfill", inplace=True)

    # back to prices
    C_interp = bs_price(f, iv_interp.index, rf, tau, iv_interp.values)

    # put-call parity: C for for K > S and P for K <= S
    K_P = iv_interp.index[iv_interp.index <= S]
    P = call_to_put(
        C = C_interp[iv_interp.index <= S],
        K = K_P,
        f = f,
        rf = rf,
        tau = tau)

    K_C = iv_interp.index[iv_interp.index > S]
    C = C_interp[iv_interp.index > S]

    # cubic contract
    yC = (6*np.log(K_C/S) - 3*np.log(K_C/S)**2)/K_C**2 * C
    yP = (6*np.log(S/K_P) + 3*np.log(S/K_P)**2)/K_P**2 * P
    W = integrate.simps(yC, K_C) - integrate.simps(yP, K_P)

    # quadratic contract
    V = mfiv_wrapper(iv_surf, f, rf, tau, method)

    # quartic contract
    yC = (12*np.log(K_C/S)**2 - 4*np.log(K_C/S)**3)/K_C**2 * C
    yP = (12*np.log(S/K_P)**2 + 4*np.log(S/K_P)**3)/K_P**2 * P
    X = integrate.simps(yC, K_C) + integrate.simps(yP, K_P)

    # mu
    mu = np.exp(rf*tau) - 1 - np.exp(rf*tau)/2*V - np.exp(rf*tau)/6*W -\
        np.exp(rf*tau)/24*X

    # all together
    mfiskew = (np.exp(rf*tau)*W - 3*mu*np.exp(rf*tau)*V + 2*mu**3)/\
        (np.exp(rf*tau)*V - mu**2)**(3/2)

    return mfiskew

def call_to_put(C, K, f, rf, tau):
    """ Put-call parity relation

    Vectorized for C, K
    """
    P = C - f*np.exp(-rf*tau) + K*np.exp(-rf*tau)

    return P

def estimation_wrapper(data, tau, parallel=False):
    """ Loop over rows in `data` to estimate RND of the underlying.

    Allow for parallel and standard (loop) regime. If parallel, uses
    multiprocessing (does not work from IPython).

    TODO: relocate to wrappers

    Parameters
    ----------
    data: pandas.DataFrame
        with rows for 5 option contracts, spot price, forward price and two
        risk-free rates called rf for domestic and y for foreign
    tau: float
        maturity of options, in frac of year
    parallel: boolean
        whether to parallelize or not

    Returns
    -------
    par: pandas.DataFrame
        of estimated parameters: [mu', sigma', w']

    """
    # delete bad rows
    data.dropna(inplace=True)

    if parallel:
        # call to multiprocessing; create pool with this many processes
        n_proc = 4
        pool = mproc.Pool(n_proc)

        # iterator is (conveniently) pd.iterrows
        iterator = data.iterrows()

        # function to apply is run_one_row, but it should be picklable,
        # hence the story with it being located right above
        out = list(zip(*pool.map(aux_fun, iterator, n_proc)))

        # output is (index, data) -> need to unpack & name
        par = pd.DataFrame(np.array([*out[1]]), index=out[0],
            columns=["mu1", "mu2", "sigma1", "sigma2", "w1", "w2"])

    else:
        # allocate space for parameters (6 of them now)
        par = pd.DataFrame(
            data=np.empty(shape=(len(data), 6)),
            index=data.index,
            columns=["mu1", "mu2", "sigma1", "sigma2", "w1", "w2"])
        # estimate in a loop
        for idx_row in data.iterrows():
            # estimate rnd for this index; unpack the dict in config to get
            # hold of constraints and bounds
            # pdb.set_trace()
            idx, res = run_one_row(idx_row, **config.cfg_dict)
            # store
            par.loc[idx,:] = res

    return(par)

def estimate_rnd(c_true, f_true, K, rf, is_iv, W, opt_meth, **kwargs):
    """ Fit parameters of mixture of two log-normals to market data.

    Everything is per period.

    Returns
    -------
    res: numpy.ndarray
        [mu1, mu2, sigma1, sigma2, w1, w2]

    """
    # if c_true is not IV, convert it to IV
    if not is_iv:
        c_true = bs_iv(c_true, f_true, K, rf, tau = 1)

    # 1) using one log-normal, come up with initial guess
    # some really simple starting values
    proto_x = np.array([np.log(f_true), np.median(c_true)], dtype = float)

    logger.debug(("Proto-x:\n"+"%.4f "*len(proto_x)+"\n") % tuple(proto_x))

    # objective function now is mixture of log-normals with 1 component
    wght = np.array([1,])
    obj_fun = lambda x: \
        objective_for_rnd(x, wght, K, rf, c_true, f_true, True, W)

    # optimization problem
    first_guess = minimize(obj_fun, proto_x, method = "SLSQP",
        bounds = [(0,proto_x[0]*2), (0,proto_x[1]*2)])

    logger.debug(("First guess:\n"+"%.2f "*len(first_guess.x)+"\n") %
        tuple(first_guess.x))

    # starting value for optimization
    x0 = first_guess.x

    # switch to 2
    x0 = [x0[0]*np.array([1.05, 1/1.05]), x0[1]*np.array([1, 1])]

    # bounds: [mu1, mu2, sigma1, sigma2]
    bounds = [(np.log(f_true*0.9), np.log(f_true/0.9))]*2 +\
        [(0, proto_x[1]*3)]*2

    # 2) using this initial guess, cook up a more sophisticated problem
    # space for parameters and loss function value
    if opt_meth == "differential_evolution":
        # try differential_evolution
        try:
            x, w = optimize_for_mixture_par(x0, K, rf, c_true, f_true, True,
                W, "differential_evolution", bounds=bounds)
        except:
            # if fails, stick to the standard stuff
            # err = sys.exc_info()[0]
            logger.error("differential evolution aborted:\n")
            x, w = optimize_for_mixture_par(x0, K, rf, c_true, f_true, True,
                W, opt_meth, bounds=bounds, **kwargs)
        # end of try/except
    else:
        x, w = optimize_for_mixture_par(x0, K, rf, c_true, f_true, True,
            W, opt_meth, **kwargs)

    logger.debug("Weight: %.2f\n" % w[0])
    logger.debug(("Par: "+"%.2f "*len(x)+"\n") % tuple(x))

    return np.concatenate((x, w))

def objective_for_rnd(par, wght, K, rf, c_true, f_true, is_iv, W = None):
    """Compute objective function for minimization problem in RND estimation.

    Objective function is loss function of errors between prices (IVs) of options priced under mixture of normals vs. true provided pricse (IVs).

    Parameters
    ----------
    par: numpy.ndarray
        [[means], [stds]] of individual components, (2,N)
    wght: numpy.ndarray
        (N,) weights of each component (in frac of 1)
    K: numpy.ndarray
        (M,) array of strike prices
    rf: float
        risk-free rate, per period (in frac of 1)
    c_true: numpy.ndarray
        (M,) array of real-world option prices (or IVs)
    f_true: float
        real-world price of forward contract on underlying
    is_iv: boolean
        True if `c_true` are option IVs rather than prices
    W: numpy.ndarray_like
        weights to components of loss function

    Return
    ------
    res: float
        loss function value
    """
    if W is None:
        W = np.diag(np.ones(len(c_true)))

    # number of components
    N = len(wght)

    # decompose par into mu and sigma
    mu = par[:N]
    sigma = par[N:]

    # fitted values
    c_hat = price_under_mixture(K, rf, mu, sigma, wght)
    f_hat = np.dot(wght, np.exp(mu + 0.5*sigma*sigma))

    # if implied_vol, transform to iv and log-prices
    if is_iv:
        # tau = 1 to avoid rescaling; make sure rf is per period!
        c_hat = bs_iv(c_hat, f_hat, K, rf, tau = 1)
        f_hat = np.log(f_hat); f_true = np.log(f_true)

    # pack into objective
    res = loss_fun(c_true, c_hat, f_true, f_hat, W)

    return res

def loss_fun(c_true, c_hat, f_true, f_hat, W = None):
    """Compute value of loss function.

    Quadratic loss with weights defined in `W`. Weight of forward pricing error is exactly 1, so rescale `W` accordingly.

    Parameters
    ----------
    c_true: numpy.ndarray
        (M,) array of real-world option prices (IVs)
    c_hat: numpy.ndarray
        (M,) array of fitted option prices (IVs)
    f_true: float
        real-world price of forward contract on underlying
    f_hat: float
        fitted price of forward contract on underlying
    W: numpy.ndarray_like
        weights to components of loss function

    Return
    ------
    loss: float
        value of loss function
    """
    # if no weighting matrix provided, use equal weighting
    if W is None:
        W = np.eye(len(c_hat))

    # deviations from options prices (ivs)
    cDev = c_true - c_hat

    # deviations from forward price (log-price)
    fDev = f_true - f_hat

    # loss: quadratic form of deviations with weights in W
    loss = 1e04*(np.dot(cDev, np.dot(W, cDev)) + fDev*fDev)

    return loss

def bs_iv(c, f, K, rf, tau, **kwargs):
    """Compute Black-Scholes implied volatility.

    Inversion of Black-Scholes formula to obtain implied volatility. Saddle point is calculated and used as initial guess for x.

    Parameters
    ----------
    c: numpy.ndarray
        (M,) array of fitted option prices (IVs)
    f: float
        forward price of underlying
    K: numpy.ndarray
        (M,) array of strike prices
    rf: float
        risk-free rate, per period (in frac of 1)
    tau: float
        time to maturity, in years
    **kwargs:
        other arguments to fsolve

    Return
    ------
    res: numpy.ndarray
        (M,) array of implied volatilities
    """
    # fprime: derivative of bs_price, or vega
    # f*e^{-rf*tau} is the same as S*e^{-y*tau}
    # lower part is dplus
    fPrime = lambda x: np.diag(f*np.exp(-rf*tau)*np.sqrt(tau)*norm.pdf(
        (np.log(f/K) + x*x/2*tau)/(x*np.sqrt(tau))))

    # saddle point (Wystup (2006), p. 19)
    saddle = np.sqrt(2/tau * np.abs(np.log(f/K)))

    # make sure it is positive, else set it next to 0
    saddle *= 0.9
    saddle[saddle <= 0] = 0.1

    # solve with fsolve
    res = fsolve(func = lambda x: bs_iv_objective(c, f, K, rf, tau, x),
        x0 = saddle, fprime = fPrime, **kwargs)

    return res

def bs_iv_objective(c_hat, f, K, rf, tau, sigma):
    """Compute discrepancy between calculated option price and provided `c_hat`.

    Parameters
    ----------
    same as in
    """
    c = bs_price(f, K, rf, tau, sigma) - c_hat

    return c

    # return c.dot(np.eye(len(sigma)).dot(c))

def price_under_mixture(K, rf, mu, sigma, wght):
    """
    Computes the price of a call option under assumption that the underlying follows a mixture of lognormal distributions with parameters specified in `mu` and `sigma` and weights specified in `w`.

    Everything is per period

    Parameters
    ----------
    K: numpy.ndarray
        of strike prices
    rf: float
        risk-free rate, in fractions of 1, per period
    mu: numpy.ndarray
        of means of log-normal distributions in the mixture
    sigma: numpy.ndarray
        of st. deviations of log-normal distributions in the mixture
    wght: numpy.ndarray
        of component weights
    Returns
    -------
    c: numpy.ndarray
        of call option prices
    p: numpy.ndarray
        of put option prices
    """
    N = len(K)
    M = len(mu)

    # tile with K a matrix with rows for distributional components
    K = np.array([K,]*M)

    # tile with mu and sigma matrices with columns for strikes
    mu = np.array([mu,]*N).transpose()
    # %timeit sigma = np.tile(sigma[np.newaxis].T, (1, len(K)))
    sigma = np.array([sigma,]*N).transpose()

    # calculate forward price based on distributional assumptions
    f = np.exp(mu + 0.5*sigma*sigma)

    # finally, call prices
    c = bs_price(f, K, rf, tau = 1, sigma = sigma)

    # result
    res = wght.dot(c)

    return res

def bs_price(f, K, rf, tau, sigma):
    """
    Black-Scholes formula in terms of forward price *f*. Definitions are from Wystup (2006).
    """

    # d+ and d-
    dplus = (np.log(f/K) + sigma*sigma/2*tau)/(sigma*np.sqrt(tau))
    dminus = dplus - sigma*np.sqrt(tau)

    res = np.exp(-rf*tau)*(f*fast_norm_cdf(dplus) - K*fast_norm_cdf(dminus))

    # return
    return res

def get_wings(r25, r10, b25, b10, atm, y, tau):
    """Finds no-arbitrage quotes of single options from quotes of contracts.

    Following Malz (2014), one can recover prices (in terms of implied vol) of
    the so-called wing options, or individual options entering the risk
    reversals and strangles.

    Everything relevant is annualized.

    Parameters
    ----------
    r25: numpy.ndarray
        iv of 25-delta risk reversals
    atm: numpy.ndarray
        iv of ATMF
    y: float
        dividend yield, in fractions of 1
    tau: float
        time to maturity

    Return
    ------
    deltas: numpy.ndarray
        of deltas (delta of ATM is re-calculated)
    ivs: numpy.ndarray
        of implied volatilities of wing options
    """
    # slightly different delta of atm option
    atm_delta = np.exp(-y*tau)*fast_norm_cdf(0.5*atm*np.sqrt(tau))

    # deltas
    deltas = np.array([0.1, 0.25, atm_delta, 0.75, 0.9])

    #
    ivs = np.array([
        atm + b10 + 0.5*r10,
        atm + b25 + 0.5*r25,
        atm,
        atm + b25 - 0.5*r25,
        atm + b10 - 0.5*r10
    ])

    return(deltas, ivs)

def strike_from_delta(delta, X, rf, y, tau, sigma, is_call):
    """ Retrieve strike prices given deltas and IV.

    Everything relevant is annualized.

    Parameters
    ----------
    delta: numpy.ndarray
        of option deltas
    X: float
        underlying price
    rf: float
        risk-free rate, in fractions of 1
    y: float
        dividend yield, in fractions of 1
    tau: float
        time to maturity, in years
    sigma: numpy.ndarray
        implied vol
    is_call: boolean
        whether options are call options

    Return
    ------
    K: numpy.ndarray
        of strike prices
    """
    # +1 for calls, -1 for puts
    phi = is_call*2-1.0

    theta_plus = (rf-y)/sigma+sigma/2

    # eq. (1.44) in Wystup
    K = X*np.exp(-phi*norm.ppf(phi*delta*np.exp(y*tau))*sigma*np.sqrt(tau) + \
        sigma*theta_plus*tau)

    return(K)

def bs_vega(f, K, rf, y, tau, sigma):
    """ Compute Black-Scholes vegas as in Wystup (2006)
    For each strike in `K` and associated `sigma` computes sensitivity of
    option to changes in volatility.

    Parameters
    ----------
    f: float
        forward price of the underlying
    K: numpy.ndarray
        of strike prices
    rf: float
        risk-free rate (domestic interest rate)
    y: float
        dividend yield (foreign interest rate)
    tau: float
        time to maturity, in years
    sigma: numpy.ndarray
        volatilities

    Returns
    -------
    vega: numpy.ndarray
        vegas
    """
    dplus = (np.log(f/K) + sigma*sigma/2*tau)/(sigma*np.sqrt(tau))
    vega = f*np.exp(-rf*tau)*np.sqrt(tau)*norm.pdf(dplus)

    return(vega)

def fast_norm_cdf(x):
    """
    Using error function from scipy.special (o(1) faster)
    TODO: relocate to misc.py
    """
    return(1/2*(1 + erf(x/np.sqrt(2))))

def run_one_row(idx_row, tau, opt_meth, constraints):
    """ Wrapper around estimate_rnd working on (idx, row) from pandas.iterrows
    TODO: relocate to optools_wrappers.py
    """
    # unpack the tuple
    idx = idx_row[0]
    row = idx_row[1]

    logger.info("doing row %.10s..." % str(idx))
    logger.debug(("This row:\n"+"%.4f "*len(row)+"\n") % tuple(row.values))

    # fetch wings
    deltas, ivs = get_wings(
        row["rr25d"],
        row["rr10d"],
        row["bf25d"],
        row["bf10d"],
        row["atm"],
        row["y"],
        tau)

    logger.debug(("IVs:\n"+"%.2f "*len(ivs)+"\n") % tuple(ivs*100))
    logger.debug(("Deltas:\n"+"%.2f "*len(deltas)+"\n") % tuple(deltas))

    # to strikes
    K = strike_from_delta(
        deltas,
        row["s"],
        row["rf"],
        row["y"],
        tau,
        ivs,
        True)

    logger.debug(("K:\n"+"%.2f "*len(K)+"\n") % tuple(K))

    # weighting matrix: inverse squared vegas
    W = bs_vega(
        row["f"],
        K,
        row["rf"],
        row["y"],
        tau,
        ivs)
    W = np.diag(1/(W*W))

    logger.debug(("Vegas:\n"+"%.2f "*len(np.diag(W))+"\n") %
        tuple(np.diag(W)))

    # estimate rnd!
    res = estimate_rnd(
        ivs*np.sqrt(tau),
        row["f"],
        K,
        row["rf"]*tau,
        True,
        W,
        opt_meth=opt_meth,
        constraints=constraints)

    return((idx, res))

def optimize_for_mixture_par(x0, K, rf, c_true, f_true, is_iv, W,
    opt_meth, **kwargs):
    """
    TODO: relocate to wrappers
    """
    if opt_meth == "differential_evolution":
        # redefine bounds - now need one on weight now
        new_bounds = kwargs["bounds"] + [(0.01, 0.49)]

        # new objective: function of [mu1,mu2,s1,s2,w1]
        def new_obj_fun(x):
            # mus, sigma
            x1 = x[:4]
            # weight
            x2 = np.array([x[4], 1-x[4]])

            return objective_for_rnd(x1, x2, K, rf, c_true, f_true, is_iv, W)

        # optimize
        res = differential_evolution(new_obj_fun, new_bounds)
        # unpack weight
        w = np.array([res.x[-1], 1-res.x[-1]])
        # unpack mu, sigma
        x = res.x[:4]

    else:
        # allocate space for grid search over weight
        xs = {}
        loss = {}
        # loop over weights
        for p in range(1,50,2):
            # two weights
            wght = np.array([p/100, 1-p/100])

            # objective changes with each weight
            obj_fun = lambda x: \
                objective_for_rnd(x, wght, K, rf, c_true, f_true, True, W)

            # optimize (**kwargs are for bounds and constraints)
            res = minimize(obj_fun, x0, method=opt_meth, **kwargs)

            # store parameters, value
            xs.update({p/100 : res.x})
            loss.update({res.fun : p/100})

        logger.debug(("Losses over w:\n"+"%04d "*len(loss.keys())+"\n") %
            tuple(range(1,50,2)))
        logger.debug(("\n"+"%4.3f "*len(loss.keys())+"\n") %
            tuple(loss.keys()))

        # find minimum of losses
        best_p = loss[min(loss.keys())]
        w = np.array([best_p, 1-best_p])

        # mu, sigma
        x = xs[best_p]

    # warning if weight is close to 0 or 0.5
    if (w[0] < 0.02) or (w[0] > 0.48):
        logger.warning("Weight of the first component is at the boundary: {}".\
            format(w[0]))

    return x, w

# def rnd_nonparametric(y, X, X_pred, rf, tau, is_iv=True, h=None, **kwargs):
#     """
#
#     Parameters
#     ----------
#     X_pred :
#         strikes must be in column 0 and equally spaced
#     **kwargs : dict
#         (if `is_iv` is True) all arguments to `bs_price` except `sigma`
#     """
#     # init model
#     mod = regm.KernelRegression(y0=y, X0=X)
#
#     # use cross_validation if needed, bleach is automatically done
#     if h is None:
#         h = mod.cross_validation(k=10)*5
#     else:
#         # if no cross_validation, need to bleach
#         mod.bleach(z_score=True, add_constant=False, dropna=True)
#
#     # fit curve
#     y_hat = mod.fit(X_pred=X_pred, h=h)
#
#     # from iv to price if needed
#     if is_iv:
#         y_hat = bs_price(rf=rf, tau=tau, sigma=y_hat, **kwargs)
#
#     # differentiate with respect to K ---------------------------------------
#     K_pred = X_pred[1:-1,0]
#     # assumed K_pred are equally spaced, calculate dK
#     dK = K_pred[1]-K_pred[0]
#     # second difference
#     d2C = np.exp(rf*1/12)*(y_hat[2:] - 2*y_hat[1:-1] + y_hat[:-2])/dK**2
#     # get rid of negative values: if after truncation there are negative
#     #    values, redo the estimation with a higher k
#     if (d2C < 0).any():
#         med_K = np.median(K_pred)
#         before_idx = K_pred <= med_K
#         after_idx = K_pred > med_K
#         body_start = np.max(np.where(d2C[before_idx] <= 0))+1
#         body_end = sum(before_idx)+np.max(np.where(d2C[after_idx] > 0))+1
#         # trim
#         d2C[:body_start] = 0
#         d2C[body_end:] = 0
#
#     # check how much we have truncated
#     intgr = np.trapz(d2C, K_pred)
#     if 1-intgr > 1e-04:
#         logger.warning("Density integrates to {:5.4f}".format(intgr))
#
#     # rescale
#     d2C = d2C/intgr
#
#     # result to a DataFrame
#     res = pd.Series(data=d2C, index=K_pred)
#
#     return res
