#
import numpy as np
from scipy.special import erf
from scipy.stats import norm
from scipy.optimize import fsolve, minimize, fmin

def estimate_rnd(c_true, f_true, K, rf, is_iv, W, **kwargs):
    """
    everything is per period!

    """
    # # weight by vegas
    # vega = 1/(vega**vega)
    # vega = vega/np.max(vega)
    # W = diag(vega)
    #
    # # upper bound

    # if c_true is not IV, convert it to IV
    if not is_iv:
        c_true = bs_iv(c_true, f_true, K, rf, tau = 1)

    # 1) using one log-normal, come up with initial guess
    # some really simple starting values
    proto_x = np.array([np.log(f_true), np.median(c_true)], dtype = float)

    # objective function is mixture of log-normals with 1 component
    wght = np.array([1,])
    obj_fun = lambda x: \
        objective_for_rnd(x, wght, K, rf, c_true, f_true, True, W)

    # optimization problem
    first_guess = minimize(obj_fun, proto_x, method = "SLSQP",
        bounds = [(0,proto_x[0]*2), (0,proto_x[1]*2)])

    # starting value for optimization
    x0 = first_guess.x

    # switch to 2
    x0 = [x0[0]*np.array([1.05, 1/1.05]), x0[1]*np.array([1, 1])]

    # # constraints of the form Ax >= 0
    # A = np.hstack((
    #     np.zeros((2,2)), np.array([[-1, 4/3], [4/3, -1]])
    #     ))
    # con_fun = lambda x: A.dot(x)  # returns 1D numpy.array
    # con = {"type": "ineq", "fun": con_fun}

    # # bounds
    # bounds = [(np.log(f_true*0.9), np.log(f_true*1.1))]*2 +\
    #     [(0, proto_x[1])]*2

    # 2) using this initial guess, cook up a more sophisticated problem
    # space for parameters and loss function value
    xs = {}
    loss = {}
    for p in range(1,48,2):
        # two probabilities
        wght = np.array([p/100, 1-p/100])

        # objective
        obj_fun = lambda x: \
            objective_for_rnd(x, wght, K, rf, c_true, f_true, True, W)

        # optimize
        second_guess = minimize(obj_fun, x0, method = "SLSQP", **kwargs)

        # store parameters, value
        xs.update({p/100 : second_guess.x})
        loss.update({second_guess.fun : p/100})

    # find minimum of losses
    best_p = loss[min(loss.keys())]

    # and parameters of interest
    x = xs[best_p]

    return((np.array([best_p, 1-best_p]),x))

def objective_for_rnd(par, wght, K, rf, c_true, f_true, is_iv, W = None):
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
    c_true: numpy.array
        (M,) array of real-world option prices (IVs)
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
    c_true: numpy.array
        (M,) array of real-world option prices (IVs)
    c_hat: numpy.array
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

    Following Malz (2014), one can recover prices (in terms of implied vol) of the so-called wing options, or individual options entering the risk reversals and strangles.

    Parameters
    ----------
    r25: numpy.array
        iv of 25-delta risk reversals
    atm: numpy.array
        iv of ATMF
    y: float
        dividend yield, in fractions of 1
    tau: float
        time to maturity

    Return
    ------
    deltas: numpy.array
        of deltas (delta of ATM is re-calculated)
    ivs: numpy.array
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
    """Retrieves strike prices given deltas and IV.

    Parameters
    ----------
    delta: numpy.array
        of option deltas
    X: float
        underlying price
    rf: float
        risk-free rate, in fractions of 1, annualized
    y: float
        dividend yield, in fractions of 1, annualized
    tau: float
        time to maturity, in years
    sigma: numpy.array
        implied vol
    is_call: boolean
        whether options are call options

    Return
    ------
    K: numpy.array
        of strike prices
    """
    # +1 for calls, -1 for puts
    phi = is_call*2-1.0

    theta_plus = (rf-y)/sigma+sigma/2

    # eq. (1.44) in Wystup
    K = X*np.exp(-phi*norm.ppf(phi*delta*np.exp(y*tau))*sigma*np.sqrt(tau) + \
        sigma*theta_plus*tau)

    return(K)

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

def bs_vega(f, K, rf, y, tau, sigma):
    """
    """
    dplus = (np.log(f/K) + sigma*sigma/2*tau)/(sigma*np.sqrt(tau))
    vega = f*np.exp(-rf*tau)*np.sqrt(tau)*norm.pdf(dplus)

    return(vega)

def fast_norm_cdf(x):
    """
    Using error function from scipy.special (o(1) faster)
    """
    return(1/2*(1 + erf(x/np.sqrt(2))))

class lognormal_mixture():
    """ Guess what.

    """
    def __init__(self, mu, sigma, wght):
        """
        """
        self.mu = mu
        self.sigma = sigma
        self.wght = wght

    def pdf(self, x):
        """ PDF at point x

        Parameters
        ----------
        x: numpy.array
            of points

        Return
        ------
        p: numpy.array
            of probability densities

        """
        # dimensions
        M = len(self.wght)
        N = len(x)

        # broadcast
        mu = np.array([self.mu,]*N).transpose()
        sigma = np.array([self.sigma,]*N).transpose()
        x = np.array([x,]*M)

        # densities
        arg = (np.log(x) - mu)/sigma
        p = 1/(x*sigma*np.sqrt(2*np.pi))*np.exp(-arg*arg/2)

        # weighted average
        p = self.wght.dot(p)

        return(p)
