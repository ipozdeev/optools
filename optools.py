#
import pandas as pd
import numpy as np
from scipy.special import erf
from scipy.stats import norm
from scipy.optimize import fsolve, minimize, fmin

import logging
logger = logging.getLogger()

def estimate_rnd(c_true, f_true, K, rf, is_iv, W, **kwargs):
    """ Fit parameters of mixture of two log-normals to market data.

    Everything is per period.

    Returns
    -------
    res:

    """
    # if c_true is not IV, convert it to IV
    if not is_iv:
        c_true = bs_iv(c_true, f_true, K, rf, tau = 1)

    # 1) using one log-normal, come up with initial guess
    # some really simple starting values
    proto_x = np.array([np.log(f_true), np.median(c_true)], dtype = float)

    # objective function now is mixture of log-normals with 1 component
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

    # bounds
    bounds = [(np.log(f_true*0.9), np.log(f_true/0.9))]*2 +\
        [(0, proto_x[1]*3)]*2

    # 2) using this initial guess, cook up a more sophisticated problem
    # space for parameters and loss function value
    xs = {}
    loss = {}
    for p in range(3,48,2):
        # two probabilities
        wght = np.array([p/100, 1-p/100])

        # objective
        obj_fun = lambda x: \
            objective_for_rnd(x, wght, K, rf, c_true, f_true, True, W)

        # optimize
        second_guess = minimize(obj_fun, x0,
            method = "SLSQP", bounds = bounds, **kwargs)

        # store parameters, value
        xs.update({p/100 : second_guess.x})
        loss.update({second_guess.fun : p/100})

    # find minimum of losses
    best_p = loss[min(loss.keys())]
    w = np.array([best_p, 1-best_p])

    # warning if weight is close to 0 or 0.5
    if (best_p < 0.04) or (best_p > 0.47):
        logger.warning("Weight of one component is at the boundary: {}".\
            format(best_p))

    # and parameters of interest
    x = xs[best_p]

    return(w, x)

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

    Everything is per period

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

    Everything relevant is annualized.

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

    Everything relevant is annualized.

    Parameters
    ----------
    delta: numpy.array
        of option deltas
    X: float
        underlying price
    rf: float
        risk-free rate, in fractions of 1
    y: float
        dividend yield, in fractions of 1
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
    Parameters
    ----------
    mu: float
        mean
    sigma: float
        standard deviation
    """
    def __init__(self, mu, sigma, wght):
        """
        """
        assert all([len(p) == len(wght) for p in [mu, sigma]])

        self.mu = mu
        self.sigma = sigma
        self.wght = wght

    def moments(self):
        """
        """
        Ex = np.exp(self.mu+self.sigma*self.sigma/2).dot(self.wght)

        return((Ex,))


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
        flag = False
        if not (type(x) is np.array):
            flag = True
            x = np.array([x,])

        # dimensions
        M = len(self.wght)
        N = len(x)

        # broadcast
        mu = np.array([self.mu,]*N).transpose()
        sigma = np.array([self.sigma,]*N).transpose()
        x = np.array([x,]*M)

        # densities
        arg = (np.log(x) - mu)/sigma
        d = 1/(x*sigma*np.sqrt(2*np.pi))*np.exp(-arg*arg/2)

        # weighted average
        d = self.wght.dot(d)

        return(d[0] if flag else d)

    def cdf(self, x):
        """ CDF
        """
        flag = False
        if not (type(x) is np.ndarray):
            flag = True
            x = np.array([x,])

        # dimensions
        M = len(self.wght)
        N = len(x)

        # broadcast
        mu = np.array([self.mu,]*N).transpose()
        sigma = np.array([self.sigma,]*N).transpose()
        x = np.array([x,]*M)

        p = 0.5*(1+erf((np.log(x)-mu)/(np.sqrt(2)*sigma)))
        p = self.wght.dot(p)

        return(p[0] if flag else p)

    def quantile(self, p):
        """ Quantile function of mixture of log-normals
        """
        flag = False
        if not (type(p) is np.ndarray):
            flag = True
            p = np.array([p,])

        # find root of CDF(x)=p
        obj_fun = lambda x: self.cdf(x) - p

        # starting value: exp{quantiles of single normal} TODO: think about it
        x0 = np.exp(norm.ppf(p,
            loc=self.mu.dot(self.wght),
            scale=self.sigma.dot(self.wght)))

        # fprime is pdf
        fprime = lambda x: np.diag(self.pdf(x))

        # solve
        q = fsolve(func=obj_fun, x0=x0, fprime=fprime)

        return(q[0] if flag else q)


def estimation_wrapper(data, tau, constraints, domain=None, perc=None):
    """ Loop over rows in `data` to estimate RND of the underlying.

    Everything is annualized

    Parameters
    ----------
    data: pandas.DataFrame
        with rows for 5 option contracts, spot price, forward price and two
        risk-free rates
    tau: float
        maturity of options, in frac of year
    constraints: dict
        of constraints as in scipy.optimize.minimize
    domain: numpy.ndarray-like
        values at which risk-neutral pdf to be calculated
    perc: numpy.ndarray-like
        percentiles of risk-neutral pdf to calculate

    Returns
    -------
    dens: pandas.DataFrame
        of density at points provided in `domain`
    par: pandas.DataFrame
        of estimated parameters: [mu', sigma', w']

    TODO: handling bad data rows
    """
    if domain is None:
        domain = np.arange(0.8, 1.5, 0.005)
    if perc is None:
        perc = np.array([0.1, 0.5, 0.9])

    # allocate space
    dens = pd.DataFrame(
        data=np.empty(shape=(len(data), len(domain))),
        index=data.index,
        columns=domain)
    quant = pd.DataFrame(
        data=np.empty(shape=(len(data), len(perc))),
        index=data.index,
        columns=perc)
    par = pd.DataFrame(
        data=np.empty(shape=(len(data), 6)),
        index=data.index,
        columns=["mu1", "mu2", "sigma1", "sigma2", "w1", "w2"])

    # estimate in a loop
    for idx, row in data.iterrows():

        logger.info("doing row %.10s..." % str(idx))

        # fetch wings
        deltas, ivs = get_wings(
            row["rr25d"],
            row["rr10d"],
            row["bf25d"],
            row["bf10d"],
            row["atm"],
            row["eur"],
            tau)

        # # from annual to by-period
        # ivs = ivs*tau

        # to strikes
        K = strike_from_delta(
            deltas,
            row["s"],
            row["chf"],
            row["eur"],
            tau,
            ivs,
            True)

        # weighting matrix: inverse squared vegas
        W = bs_vega(
            row["f"],
            K,
            row["chf"],
            row["eur"],
            tau,
            ivs)
        W = np.diag(1/(W*W))

        # estimate rnd!
        res = estimate_rnd(
            ivs*np.sqrt(tau),
            row["f"],
            K,
            row["chf"]*tau,
            True,
            W,
            constraints=constraints)

        # init lognormal_mixture object
        ln_mix = lognormal_mixture(res[1][:2], res[1][2:], res[0])

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
        par.loc[idx,:] = np.concatenate([res[1], res[0]])
        dens.loc[idx,:] = p
        quant.loc[idx,:] = q

    return(dens, par, quant)

# if __name__ == "__main__":
#     from import_data import data
#     data = data.ix[1000:1004,:]
#
#     # constraints
#     C = np.array([
#         [0, 0],
#         [0, 0],
#         [-1, 1],
#         [4/3, -3/4]])
#     constraints = {
#         "type" : "ineq",
#         "fun" : lambda x: x.dot(C)}
#
#     p = estimation_wrapper(data, constraints)
