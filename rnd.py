import numpy as np
from scipy.optimize import minimize
from tensorflow_probability import distributions as tfd

from .blackscholes import option_price as bs_price
from .blackscholesmix import \
    option_price as bsmix_option_price, \
    forward_price as bsmix_forward_price


def fit_lognormal_mix(option_price, strike, is_call: bool, forward, rf,
                      x0: np.ndarray=None, weights: np.ndarray=None) \
        -> np.ndarray:
    """Fit a mixuture of 2 log-normal densities to option prices.

    Assuming the distribution of the forward price at expiration is a
    mixture of two log-normal distributions defined by the parameter vector
    (w, mu_1, mu_2, sigma_1, sigma_2), fits these parameters to observed
    `option_price` of with corresponding `strike` and the forward price
    itself by minimizing the (possibly weighted using `weights`) sum of
    squared pricing errors. Imposes bounds on parameters: weight between 0.1
    and 0.5, sigma's nonnegative

    Parameters
    ----------
    option_price : np.ndarray
        option prices (premium)
    strike : np.ndarray
        strike prices, same length as `option_price`
    is_call : bool
    forward : float
        forward price
    rf : float
        risk-free rate (rate of teh counter currency), in frac of 1 p.a.
    x0 : np.ndarray
        initial values of (w, mu_1, mu_2, sigma_1, sigma_2)
    weights : np.ndarray
        weights to use to place more/less importance on certain elements of
        (forward, option_1, ... option_n); the first element corresponds to
        the forward pricing errors

    Returns
    -------
    np.ndarray
        of parameters (w, mu_1, mu_2, sigma_1, sigma_2)

    """
    # parameters [w, mu_1, mu_2, sigma_1, sigma_2]
    if x0 is None:
        x0 = np.array([0.34,
                       np.log(forward)/2, np.log(forward)/2,
                       0.2, 0.2])

    # equal weighting if not stated otherwise
    if weights is None:
        weights = np.ones(shape=(len(strike), ))

    # objective fun is the 2-norm of the diff between observed and fitted
    # prices, including the forward
    def obj_fun(x_):
        y_hat = np.concatenate((
            [bsmix_forward_price(w=np.array([x_[0], 1-x_[0]]),
                                mu=x_[1:3],
                                vol=x_[3:])],
            bsmix_option_price(w=np.array([x_[0], 1-x_[0]]),
                               mu=x_[1:3],
                               vol=x_[3:],
                               strike=strike,
                               rf=rf,
                               is_call=True),
        ))
        y = np.concatenate(([forward], option_price))
        res_ = np.linalg.norm((y_hat - y) * weights)
        return res_

    # weights are in (0.1, 0.5)
    bounds = [(0.1, 0.5),
              (-np.inf, np.inf), (-np.inf, np.inf),
              (1e-06, np.inf), (1e-06, np.inf)]

    # mimimize!
    res = minimize(obj_fun, x0, bounds=bounds).x

    return res
