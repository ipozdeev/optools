#
import numpy as np
from scipy.special import erf
from scipy.stats import norm
from scipy.optimize import fsolve, minimize, fmin

def estimate_rnd(cTrue, fTrue, K, rf, tau, is_iv, W, **kwargs):
    """
    everything is per period!
    TODO: THIS FUNCTION IS NOW HALF-WAY DONE
    """
    # # weight by vegas
    # vega = 1/(vega**vega)
    # vega = vega/np.max(vega)
    # W = diag(vega)
    #
    # # upper bound

    if not is_iv:
        cTrue = bs_iv(cTrue, fTrue, K, rf, tau)

    # initial values for mu and sigma
    par0 = np.array([np.log(fTrue), np.median(cTrue)], dtype = float)

    # objective function
    obj_fun = lambda x: \
        objective_for_rnd(x, np.array([1,]), K, rf, cTrue, fTrue, True, W)

    # optimization problem
    clever_guess = minimize(obj_fun, par0, method = "Powell", **kwargs)

    return(clever_guess.x)

def objective_for_rnd(par, wght, K, rf, cTrue, fTrue, is_iv, W = None):
    """Compute objective function for minimization problem in RND estimation.

    Objective function is loss function of errors between prices (IVs) of options priced under mixture of normals vs. true provided pricse (IVs).

    Parameters
    ----------
    par: numpy.ndarray
        [[means], [stds]] of individual components, (2,N)
    wght: numpy.array
        (N,) weights of each component (in frac of 1)
    K: numpy.array
        (M,) array of strike prices
    rf: float
        risk-free rate, per period (in frac of 1)
    cTrue: numpy.array
        (M,) array of real-world option prices (IVs)
    fTrue: float
        real-world price of forward contract on underlying
    is_iv: boolean
        True if `cTrue` are option IVs rather than prices
    W: numpy.ndarray_like
        weights to components of loss function

    Return
    ------
    res: float
        loss function value
    """
    # number of components
    N = len(wght)
    # decompose par into mu and sigma
    mu = par[:N]
    sigma = par[N:]

    # fitted values
    cHat = price_under_mixture(K, rf, mu, sigma, wght)
    fHat = np.dot(wght, np.exp(mu + 0.5*sigma*sigma))

    # if implied_vol, transform to iv and log-prices
    if is_iv:
        # tau = 1 to avoid rescaling; make sure rf is per period!
        cHat = bs_iv(cHat, fHat, K, rf, tau = 1)
        fHat = np.log(fHat); fTrue = np.log(fTrue)

    # pack into objective
    res = loss_fun(cTrue, cHat, fTrue, fHat, W)

    return res

def loss_fun(cTrue, cHat, fTrue, fHat, W = None):
    """Compute value of loss function.

    Quadratic loss with weights defined in `W`. Weight of forward pricing error is exactly 1, so rescale `W` accordingly.

    Parameters
    ----------
    cTrue: numpy.array
        (M,) array of real-world option prices (IVs)
    cHat: numpy.array
        (M,) array of fitted option prices (IVs)
    fTrue: float
        real-world price of forward contract on underlying
    fHat: float
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
        W = np.eye(len(cHat))

    # deviations from options prices (ivs)
    cDev = cTrue - cHat

    # deviations from forward price (log-price)
    fDev = fTrue - fHat

    # loss: quadratic form of deviations with weights in W
    loss = 1e04*(np.dot(cDev, np.dot(W, cDev)) + fDev*fDev)

    return loss

def bs_iv(c, f, K, rf, tau, **kwargs):
    """Compute Black-Scholes implied volatility.

    Inversion of Black-Scholes formula to obtain implied volatility. Saddle point is calculated and used as initial guess for x.

    Parameters
    ----------
    c: numpy.array
        (M,) array of fitted option prices (IVs)
    f: float
        forward price of underlying
    K: numpy.array
        (M,) array of strike prices
    rf: float
        risk-free rate, per period (in frac of 1)
    tau: float
        time to maturity, in years
    **kwargs:
        other arguments to fsolve

    Return
    ------
    res: numpy.array
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

def bs_iv_objective(cHat, f, K, rf, tau, sigma):
    """Compute discrepancy between calculated option price and provided `cHat`.

    Parameters
    ----------
    same as in
    """
    c = bs_price(f, K, rf, tau, sigma) - cHat

    return c

    # return c.dot(np.eye(len(sigma)).dot(c))

def price_under_mixture(K, rf, mu, sigma, wght):
    """
    Computes the price of a call option under assumption that the underlying follows a mixture of lognormal distributions with parameters specified in `mu` and `sigma` and weights specified in `w`.
    Parameters
    ----------
    K: numpy.array
        of strike prices
    rf: float
        risk-free rate, in fractions of 1, per period
    mu: numpy.array
        of means of log-normal distributions in the mixture
    sigma: numpy.array
        of st. deviations of log-normal distributions in the mixture
    wght: numpy.array
        of component weights
    Returns
    -------
    c: numpy.array
        of call option prices
    p: numpy.array
        of put option prices
    """
    # tile with K a matrix with rows for distributional components
    K = np.array([K,]*len(mu))

    # tile with mu and sigma matrices with columns for strikes
    mu = np.array([mu,]*len(K)).transpose()
    # %timeit sigma = np.tile(sigma[np.newaxis].T, (1, len(K)))
    sigma = np.array([sigma,]*len(K)).transpose()

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

# def bs_greeks(x = None, f = None, K, rf, T, t, sigma, y, is_call):
#     """
#     """
#     tau = T-t            # time to maturity
#     phi = is_call*2 - 1  # +1 for call, -1 for put
#
#     # if no forward was provided, calculate it
#     if f is None:
#         f = x*np.exp((rf-y)*tau)
#     if x is None:
#         x = f*np.exp((y-rf)*tau)
#
#     dplus = (np.log(f/K) + sigma*sigma/2*tau)/(sigma*np.sqrt(tau))
#     dminus = dplus - sigma*np.sqrt(tau)
#
#     delta = phi*np.exp(-y*tau)*fast_norm_cdf(phi*dplus)
#     gamma = no.exp(-y*tau)*norm.pdf(dplus)/(x*sigma*np.sqrt(tau))

# def bs_vega(x = None, f = None, K, rf = None, y = None, tau, sigma,
#     dplus = None):
#     """
#     """
#     if dplus is None:
#         dplus = (np.log(f/K) + sigma*sigma/2*tau)/(sigma*np.sqrt(tau)))
#     vega = np.diag(f*np.exp(-rf*tau)*np.sqrt(tau)*norm.pdf(dplus)


def fast_norm_cdf(x):
    """
    Using error function from scipy.special (o(1) faster)
    """
    return(1/2*(1 + erf(x/np.sqrt(2))))
