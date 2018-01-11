import pandas as pd
import numpy as np
from optools.helpers import fast_norm_cdf
from scipy.stats import norm
from scipy.optimize import fsolve
from scipy import interpolate, integrate
from statsmodels.nonparametric.kernel_regression import KernelReg

# from scipy.optimize import minimize, differential_evolution

# logging
import logging
logger = logging.getLogger()


def bs_price(forward_p, strike_p, rf, tau, vola):
    """Compute Black-Scholes price (forward price-based).

    Definitions are as in Wystup (2006).

    Parameters
    ----------
    forward_p : float
        forward price
    strike_p : float or numpy.ndarray
        strikes prices
    rf : float
        risk-free rate, in (frac of 1) p.a.
    tau : float
        maturity, in years
    vola : float or numpy.ndarray
        volatility, in (frac of 1) p.a.

    Returns
    -------
    res : float
        price
    """
    # d+ and d-
    d_plus = (np.log(forward_p / strike_p) + vola ** 2 / 2 * tau) / \
             (vola * np.sqrt(tau))
    d_minus = d_plus - vola * np.sqrt(tau)

    res = np.exp(-rf * tau) *\
        (forward_p * fast_norm_cdf(d_plus) - strike_p * fast_norm_cdf(d_minus))

    # return
    return res


def call_to_put(call_p, strike_p, forward_p, rf, tau):
    """Calculate price of put from the put-call parity relation.

    Vectorized for call_p, strike

    Parameters
    ----------
    call_p : float or numpy.array
        call price
    strike_p : float or numpy.array
        strike price
    forward_p : float
        forward price of the underlying
    rf : float
        risk-free rate, in (frac of 1) p.a.
    tau : float
        maturity, in years

    Returns
    -------
    p : float or numpy.array
        put price
    """
    p = call_p - forward_p * np.exp(-rf * tau) + strike_p * np.exp(-rf * tau)

    return p


def interpolate_iv(iv_surf, method="spline", x_pred=None):
    """Interpolate iv surface.

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
    x_pred : pandas.DataFrame
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
    if x_pred is None:
        x_pred = [np.linspace(np.min(iv_surf["K"]), np.max(iv_surf["K"]),
                              np.ceil(np.ptp(iv_surf["K"])/0.005)), ]
    else:
        # turn columns of df into list elements
        x_pred = list(x_pred.values.T)

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
    #         x_pred += [np.array([tau,]*len(K_pred)),]
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
            # print(x_pred[0])
            # # xnew, ynew = np.meshgrid(x_pred[0], x_pred[1], indexing="ij")
            # # print(xnew)
            # # print(ynew)
            # znew = interpolate.bisplev(x_pred[0], [tau,], tck)
            # print(znew)
        else:
            # estimate iv ~ strikes
            tck = interpolate.splrep(exog[0], endog, s=0)
            # predict
            znew = interpolate.splev(x_pred[0], tck, der=0)

    elif method == "kernel":
        # estimate endog must be a list of one elt
        kr = KernelReg(endog=[endog,], exog=exog, var_type=var_type,
            reg_type="ll")
        # fit
        znew, _ = kr.fit(data_predict=x_pred)

    # return, squeeze jsut in case
    res = pd.Series(index=x_pred[0], data=znew.squeeze())

    return res


def bs_iv(call_p, forward_p, strike, rf, tau, **kwargs):
    """Compute Black-Scholes implied volatility.

    Inversion of Black-Scholes formula to obtain implied volatility. Saddle
    point is calculated and used as the initial guess.

    Parameters
    ----------
    call_p: numpy.ndarray
        (M,) array of fitted option prices (IVs)
    forward_p: float
        forward price of underlying
    strike: numpy.ndarray
        (M,) array of strike prices
    rf: float
        risk-free rate, (in frac of 1) per period tau
    tau: float
        time to maturity, in years
    **kwargs: dict
        other arguments to fsolve

    Return
    ------
    res: numpy.ndarray
        (M,) array of implied volatilities
    """
    # fprime: derivative of bs_price, or vega
    # forward_p*e^{-rf*tau} is the same as S*e^{-y*tau}
    # lower part is dplus
    def f_prime(x):
        x_1 = forward_p * np.exp(-rf * tau) * np.sqrt(tau)
        x_2 = norm.pdf(
            (np.log(forward_p/strike) + x * x/2 * tau) / (x * np.sqrt(tau)))

        val = np.diag(x_1 * x_2)

        return val

    # saddle point (Wystup (2006), p. 19)
    saddle = np.sqrt(2 / tau * np.abs(np.log(forward_p / strike)))

    # make sure it is positive, else set it next to 0
    saddle *= 0.9
    saddle[saddle <= 0] = 0.1

    # objective
    def f_obj(x):
        val = bs_iv_objective(call_p, forward_p, strike, rf, tau, x)

        return val

    # solve with fsolve, use f_prime for gradient
    res = fsolve(func=f_obj, x0=saddle, fprime=f_prime, **kwargs)

    return res


def bs_iv_objective(c_hat, forward_p, strike, rf, tau, sigma):
    """Compute discrepancy between the calculated option price and `c_hat`.

    Parameters
    ----------
    c_hat : float
    forward_p
    strike
    rf : float
        risk-free rate, in (frac of 1) p.a.
    tau
    sigma : float
        implied vola, in (frac of 1) p.a.

    Returns
    -------

    """
    c = bs_price(forward_p, strike, rf, tau, sigma) - c_hat

    return c


def bs_vega(forward_p, strike, y, tau, sigma):
    """Compute Black-Scholes vega as in Wystup (2006)

    For each strike in `K` and associated `sigma` computes sensitivity of
    option to changes in volatility.

    Parameters
    ----------
    forward_p: float
        forward price of the underlying
    strike: numpy.ndarray
        of strike prices
    y: float
        dividend yield (foreign interest rate)
    tau: float
        time to maturity, in years
    sigma: numpy.ndarray
        implied vola, in (frac of 1) p.a.

    Returns
    -------
    vega: numpy.ndarray
        vegas
    """
    dplus = (np.log(forward_p / strike) + sigma**2 / 2 * tau) / \
        (sigma * np.sqrt(tau))
    vega = forward_p * np.exp(-y * tau) * np.sqrt(tau) * norm.pdf(dplus)

    return vega


def wings_iv_from_combies_iv(rr, bf, atm, delta=None):
    """Calculate implied vola of calls from that of put/call combinations.

    See Wystup, p.24.

    Parameters
    ----------
    atm : float
        implied vola of the at-the-money option
    rr : float
        implied vola of the risk reversal
    bf : float
        implied vola of the butterfly contract
    delta : float
        delta, in ((frac of 1)), e.g. 0.25 or 0.10


    Returns
    -------
    res : list or pandas.Series
        of implied volas; pandasSeries indexed by delta if `delta` was provided

    """
    # implied volas
    two_ivs = np.array([
        atm + bf + 0.5 * rr,
        atm + bf - 0.5 * rr
    ])

    # if delta was not supplied, return list
    if delta is None:
        return two_ivs

    # deltas
    two_deltas = [delta, 1 - delta]

    # create a Series
    res = pd.Series(index=two_deltas, data=two_ivs).rename("iv")
    res.index.name = "delta"

    return res


def strike_from_delta(delta, spot, rf, div_yield, tau, vola, is_call):
    """Calculate strike prices given delta and implied vola.

    Everything relevant is annualized.

    Parameters
    ----------
    delta: float or numpy.ndarray
        of option deltas, in (frac of 1)
    spot: float
        underlying price
    rf: float
        risk-free rate, in (frac of 1) p.a.
    div_yield: float
        dividend yield, in (frac of 1) p.a.
    tau: float
        time to maturity, in years
    vola: float or numpy.ndarray
        implied vol
    is_call: bool
        whether options are call options

    Return
    ------
    k: float or numpy.ndarray
        of strike prices
    """
    # +1 for calls, -1 for puts
    phi = is_call*2 - 1.0

    theta_plus = (rf - div_yield) / vola + vola / 2

    # eq. (1.44) in Wystup
    k = spot * \
        np.exp(-phi * norm.ppf(phi * delta * np.exp(div_yield * tau)) *
               vola * np.sqrt(tau) + vola * theta_plus * tau)

    return k


def mfiv(call_p, strike, forward_p, rf, tau, method="jiang_tian"):
    """

    Parameters
    ----------
    call_p : numpy.array
        of call option prices
    strike : numpy.array
        of strike prices
    forward_p : float
        forward price
    rf : float
        risk-free rate, in (frac of 1) p.a.
    tau : float
        maturity, in years
    method : str
        'jiang_tian' or 'sarno'

    Returns
    -------
    res : float
        mfiv, in (frac of 1) p.a.

    """
    if method == "jiang_tian":
        # integrate
        integrand = (call_p * np.exp(rf * tau) -\
            np.maximum(np.zeros(shape=(len(call_p), )), forward_p-strike)) /\
            (strike * strike)

        res = integrate.simps(integrand, strike) * 2

    elif method == "sarno":
        # part of prices to puts
        put_p = call_to_put(call_p, strike, forward_p, rf, tau)

        # out-of-the-money calls and puts
        call_strike = strike[strike >= forward_p]
        otm_call_p = call_p[strike >= forward_p]
        put_strike = strike[strike < forward_p]
        otm_put_p = put_p[strike < forward_p]

        # integrate
        res = integrate.simps(otm_put_p / put_strike**2, put_strike) + \
            integrate.simps(otm_call_p / call_strike**2, call_strike)

        res *= 2*np.exp(rf*tau)

    else:
        raise NameError("Method not allowed!")

    # annualize
    res /= tau

    return res


def fill_by_no_arb(spot, forward, rf, div_yield, tau):
    """

    Parameters
    ----------
    spot
    forward
    rf : float
        in (frac of 1) p.a.
    div_yield : float
        in (frac of 1) p.a.
    tau : float
        maturity, in years

    Returns
    -------
    args : dict
        same arguments,
    """
    if (tau is None) | np.isnan(tau):
        raise ValueError("Maturity not provided!")

    # collect all arguments
    args = locals()

    # find one nan
    where_nan = {k: v for k, v in args.items() if np.isnan(v)}

    if len(where_nan) > 1:
        raise ValueError("Only one argument can be missing!")

    k, v = list(where_nan.items())[0]

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

# def estimate_rnd(c_true, f_true, strike, rf, is_iv, W, opt_meth, **kwargs):
#     """Fit parameters of mixture of two log-normals to market data.
# 
#     Everything is per period.
# 
#     Returns
#     -------
#     res: numpy.ndarray
#         [mu1, mu2, sigma1, sigma2, w1, w2]
# 
#     """
#     # if c_true is not IV, convert it to IV
#     if not is_iv:
#         c_true = bs_iv(c_true, f_true, strike, rf, tau = 1)
# 
#     # 1) using one log-normal, come up with initial guess
#     # some really simple starting values
#     proto_x = np.array([np.log(f_true), np.median(c_true)], dtype = float)
# 
#     logger.debug(("Proto-x:\n"+"%.4f "*len(proto_x)+"\n") % tuple(proto_x))
# 
#     # objective function now is mixture of log-normals with 1 component
#     wght = np.array([1,])
#     obj_fun = lambda x: \
#         objective_for_rnd(x, wght, strike, rf, c_true, f_true, True, W)
# 
#     # optimization problem
#     first_guess = minimize(obj_fun, proto_x, method = "SLSQP",
#         bounds = [(0,proto_x[0]*2), (0,proto_x[1]*2)])
# 
#     logger.debug(("First guess:\n"+"%.2f "*len(first_guess.x)+"\n") %
#         tuple(first_guess.x))
# 
#     # starting value for optimization
#     x0 = first_guess.x
# 
#     # switch to 2
#     x0 = [x0[0]*np.array([1.05, 1/1.05]), x0[1]*np.array([1, 1])]
# 
#     # bounds: [mu1, mu2, sigma1, sigma2]
#     bounds = [(np.log(f_true*0.9), np.log(f_true/0.9))]*2 +\
#         [(0, proto_x[1]*3)]*2
# 
#     # 2) using this initial guess, cook up a more sophisticated problem
#     # space for parameters and loss function value
#     if opt_meth == "differential_evolution":
#         # try differential_evolution
#         try:
#             x, w = optimize_for_mixture_par(x0, strike, rf, c_true, f_true,
#                                             True, W, "differential_evolution",
#                                             bounds=bounds)
#         except:
#             # if fails, stick to the standard stuff
#             # err = sys.exc_info()[0]
#             logger.error("differential evolution aborted:\n")
#             x, w = optimize_for_mixture_par(x0, strike, rf, c_true, f_true,
#                                             True, W, opt_meth, bounds=bounds,
#                                             **kwargs)
#         # end of try/except
#     else:
#         x, w = optimize_for_mixture_par(x0, strike, rf, c_true, f_true, True,
#                                         W, opt_meth, **kwargs)
# 
#     logger.debug("Weight: %.2f\n" % w[0])
#     logger.debug(("Par: "+"%.2f "*len(x)+"\n") % tuple(x))
# 
#     return np.concatenate((x, w))
#
#
# def objective_for_rnd(par, wght, strike, rf, c_true, f_true, is_iv, W=None):
#     """Compute objective function for minimization problem in RND estimation.
#
#     Objective function is loss function of errors between prices (IVs) of
#     options priced under mixture of normals vs. true provided pricse (IVs).
#
#     Parameters
#     ----------
#     par: numpy.ndarray
#         [[means], [stds]] of individual components, (2,N)
#     wght: numpy.ndarray
#         (N,) weights of each component (in frac of 1)
#     strike: numpy.ndarray
#         (M,) array of strike prices
#     rf: float
#         risk-free rate, per period (in frac of 1)
#     c_true: numpy.ndarray
#         (M,) array of real-world option prices (or IVs)
#     f_true: float
#         real-world price of forward contract on underlying
#     is_iv: boolean
#         True if `c_true` are option IVs rather than prices
#     W: numpy.ndarray-like
#         weights to components of loss function
#
#     Return
#     ------
#     res: float
#         loss function value
#     """
#     if W is None:
#         W = np.diag(np.ones(len(c_true)))
#
#     # number of components
#     N = len(wght)
#
#     # decompose par into mu and sigma
#     mu = par[:N]
#     sigma = par[N:]
#
#     # fitted values
#     c_hat = price_under_mixture(strike, rf, mu, sigma, wght)
#     f_hat = np.dot(wght, np.exp(mu + 0.5*sigma*sigma))
#
#     # if implied_vol, transform to iv and log-prices
#     if is_iv:
#         # tau = 1 to avoid rescaling; make sure rf is per period!
#         c_hat = bs_iv(c_hat, f_hat, strike, rf, tau = 1)
#         f_hat = np.log(f_hat); f_true = np.log(f_true)
#
#     # pack into objective
#     res = loss_fun(c_true, c_hat, f_true, f_hat, W)
#
#     return res
#
#
# def loss_fun(c_true, c_hat, f_true, f_hat, W=None):
#     """Compute value of loss function.
#
#     Quadratic loss with weights defined in `W`. Weight of forward pricing
#     error is exactly 1, so rescale `W` accordingly.
#
#     Parameters
#     ----------
#     c_true: numpy.ndarray
#         (M,) array of real-world option prices (IVs)
#     c_hat: numpy.ndarray
#         (M,) array of fitted option prices (IVs)
#     f_true: float
#         real-world price of forward contract on underlying
#     f_hat: float
#         fitted price of forward contract on underlying
#     W: numpy.ndarray_like
#         weights to components of loss function
#
#     Return
#     ------
#     loss: float
#         value of loss function
#     """
#     # if no weighting matrix provided, use equal weighting
#     if W is None:
#         W = np.eye(len(c_hat))
#
#     # deviations from options prices (ivs)
#     c_dev = c_true - c_hat
#
#     # deviations from forward price (log-price)
#     f_dev = f_true - f_hat
#
#     # loss: quadratic form of deviations with weights in W
#     loss = 1e04*(np.dot(c_dev, np.dot(W, c_dev)) + f_dev*f_dev)
#
#     return loss
#
#
# def price_under_mixture(strike, rf, mu, sigma, wght):
#     """
#     Computes the price of a call option under assumption that the underlying 
#     follows a mixture of lognormal distributions with parameters specified in 
#     `mu` and `sigma` and weights specified in `w`.
# 
#     Everything is per period
# 
#     Parameters
#     ----------
#     strike: numpy.ndarray
#         of strike prices
#     rf: float
#         risk-free rate, in (frac of 1), per period
#     mu: numpy.ndarray
#         of means of log-normal distributions in the mixture
#     sigma: numpy.ndarray
#         of st. deviations of log-normal distributions in the mixture
#     wght: numpy.ndarray
#         of component weights
#     Returns
#     -------
#     c: numpy.ndarray
#         of call option prices
#     p: numpy.ndarray
#         of put option prices
#     """
#     N = len(strike)
#     M = len(mu)
# 
#     # tile with strike a matrix with rows for distributional components
#     strike = np.array([strike, ] * M)
# 
#     # tile with mu and sigma matrices with columns for strikes
#     mu = np.array([mu,]*N).transpose()
#     # %timeit sigma = np.tile(sigma[np.newaxis].T, (1, len(strike)))
#     sigma = np.array([sigma,]*N).transpose()
# 
#     # calculate forward price based on distributional assumptions
#     f = np.exp(mu + 0.5*sigma*sigma)
# 
#     # finally, call prices
#     c = bs_price(f, strike, rf, tau = 1, sigma = sigma)
# 
#     # result
#     res = wght.dot(c)
# 
#     return res
#
#
# def get_wings(r25, r10, b25, b10, atm, y, tau):
#     """Find no-arbitrage quotes of single options from quotes of contracts.
#
#     Following Malz (2014), one can recover prices (in terms of implied vol)
#     of the so-called wing options, or individual options entering the risk
#     reversals and strangles.
#
#     Everything relevant is annualized.
#
#     Parameters
#     ----------
#     r25: numpy.ndarray
#         iv of 25-delta risk reversals
#     atm: numpy.ndarray
#         iv of ATMF
#     y: float
#         dividend yield, in (frac of 1)
#     tau: float
#         time to maturity
#
#     Return
#     ------
#     deltas: numpy.ndarray
#         of deltas (delta of ATM is re-calculated)
#     ivs: numpy.ndarray
#         of implied volatilities of wing options
#     """
#     # slightly different delta of atm option
#     atm_delta = np.exp(-y*tau) * fast_norm_cdf(0.5*atm*np.sqrt(tau))
#
#     # deltas
#     deltas = np.array([0.1, 0.25, atm_delta, 0.75, 0.9])
#
#     #
#     ivs = np.array([
#         atm + b10 + 0.5*r10,
#         atm + b25 + 0.5*r25,
#         atm,
#         atm + b25 - 0.5*r25,
#         atm + b10 - 0.5*r10
#     ])
#
#     return deltas, ivs
#
#
# def optimize_for_mixture_par(x0, K, rf, c_true, f_true, is_iv, W,
#     opt_meth, **kwargs):
#     """
#     TODO: relocate to wrappers
#     """
#     if opt_meth == "differential_evolution":
#         # redefine bounds - now need one on weight now
#         new_bounds = kwargs["bounds"] + [(0.01, 0.49)]
# 
#         # new objective: function of [mu1,mu2,s1,s2,w1]
#         def new_obj_fun(x):
#             # mus, sigma
#             x1 = x[:4]
#             # weight
#             x2 = np.array([x[4], 1-x[4]])
# 
#             return objective_for_rnd(x1, x2, K, rf, c_true, f_true, is_iv, W)
# 
#         # optimize
#         res = differential_evolution(new_obj_fun, new_bounds)
#         # unpack weight
#         w = np.array([res.x[-1], 1-res.x[-1]])
#         # unpack mu, sigma
#         x = res.x[:4]
# 
#     else:
#         # allocate space for grid search over weight
#         xs = {}
#         loss = {}
#         # loop over weights
#         for p in range(1,50,2):
#             # two weights
#             wght = np.array([p/100, 1-p/100])
# 
#             # objective changes with each weight
#             obj_fun = lambda x: \
#                 objective_for_rnd(x, wght, K, rf, c_true, f_true, True, W)
# 
#             # optimize (**kwargs are for bounds and constraints)
#             res = minimize(obj_fun, x0, method=opt_meth, **kwargs)
# 
#             # store parameters, value
#             xs.update({p/100 : res.x})
#             loss.update({res.fun : p/100})
# 
#         logger.debug(("Losses over w:\n"+"%04d "*len(loss.keys())+"\n") %
#             tuple(range(1,50,2)))
#         logger.debug(("\n"+"%4.3f "*len(loss.keys())+"\n") %
#             tuple(loss.keys()))
# 
#         # find minimum of losses
#         best_p = loss[min(loss.keys())]
#         w = np.array([best_p, 1-best_p])
# 
#         # mu, sigma
#         x = xs[best_p]
# 
#     # warning if weight is close to 0 or 0.5
#     if (w[0] < 0.02) or (w[0] > 0.48):
#         logger.warning("Weight of the first component is at the boundary: {}".\
#             format(w[0]))
# 
#     return x, w
#
#
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

if __name__ == "__main__":
    fill_by_no_arb()