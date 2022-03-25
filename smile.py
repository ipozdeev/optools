import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.misc import derivative
from scipy.integrate import simps
from scipy.optimize import least_squares, fsolve
from statsmodels.nonparametric.kernel_regression import KernelReg
import matplotlib.pyplot as plt

from optools.helpers import construct_new_x
from optools.blackscholes import option_price as bs_price
from optools.implied import mfivariance, simple_var_swap_rate
from optools.strike import strike_from_delta, strike_from_atm


class VolatilitySmile:
    """Volatility smile as a mapping from strikes to Black-Scholes IVs.

    Parameters
    ----------
    mapping : callable
        mapping from strike to vola (in percent p.a.)
    tau : float
        in years
    """
    def __init__(self, mapping: callable, tau: float):
        self.mapping = mapping
        self.tau = tau

    def __call__(self, *args, **kwargs):
        return self.mapping(*args, **kwargs)

    def get_mfivariance(self, forward, r_counter, svix=False, limits=None,
                        dk=1e-04) -> float:
        """Calculate the model-free implied variance.

        The mfiv is calculated as the integral over call prices weighted by
        strikes (for details see Jiang and Tian (2005) and Martin (2017)).
        The volas are transformed to the prices of vanillas, then the
        integration is done using Simpson's rule.

        Parameters
        ----------
        forward : float
        r_counter : float
            in frac of 1 p.a.
        svix : bool
            True to calculate Martin (2017) simple variance swap rates
        limits : tuple of floats
            limits of integration; defaults to (forward/3, forward*2)
        dk : float
            integration step size

        Returns
        -------
        float
            mfiv, in (frac of 1) p.a.
        """
        if self.tau is None:
            raise ValueError("parameter `tau` must be set on instantiation.")

        # call pricer
        def call_pricer(k_):
            return bs_price(forward=forward, strike=k_,
                            r_counter=r_counter, tau=self.tau, vola=self(k_), is_call=True)

        # put pricer
        def put_pricer(k_):
            return bs_price(forward=forward, strike=k_,
                            r_counter=r_counter, tau=self.tau, vola=self(k_),
                            is_call=False)

        dk = 1e-04
        if limits is None:
            k_min, k_max = forward / 3, forward * 2
        else:
            k_min, k_max = limits

        k_put = np.arange(k_min, forward, dk)
        k_call = np.arange(forward, k_max, dk)
        v = 2 * np.exp(r_counter * self.tau) * (
            simps(y=put_pricer(k_put) / k_put ** 2, x=k_put) +
            simps(y=call_pricer(k_call) / k_call ** 2, x=k_call)
        )

        # mfiv
        if svix:
            v = 2 * np.exp(r_counter * self.tau) / (forward ** 2) * (
                simps(y=put_pricer(k_put), x=k_put) +
                simps(y=call_pricer(k_call), x=k_call)
            )
        else:
            v = 2 * np.exp(r_counter * self.tau) * (
                simps(y=put_pricer(k_put) / k_put ** 2, x=k_put) +
                simps(y=call_pricer(k_call) / k_call ** 2, x=k_call)
            )

        # annualize
        v /= self.tau

        return v

    def get_rnd(self, r_counter, forward=None, spot=None, r_base=None) -> \
            callable:
        """Get risk-neutral density estimator of Breeden and Litzenberger.

        Parameters
        ----------
        r_counter : float
            risk-free rate (rate in counter currency), in frac of 1 p.a.
        forward : float
        spot : float
        r_base : float
            dividend yield (rate in base currency), in frac of 1 p.a.

        Returns
        -------
        callable
            function to compute density at a given point
        """
        # the Black-Schole formula at K is being differentiated twice
        def func_to_diff(x_):
            c_ = bs_price(strike=x_, r_counter=r_counter, tau=self.tau,
                          vola=self(x_), forward=forward, is_call=True)
            return c_

        # step of 1e-04 should be enough
        def res(x_):
            res_ = derivative(func_to_diff, x_, dx=1e-04, n=2) \
                * np.exp(r_counter * self.tau)

            return res_

        return res


    def plot(self, domain):
        """Plot."""
        fig, ax = plt.subplots()
        iv = self(domain)

        ax.plot(domain, iv)

        return fig, ax


class SABR(VolatilitySmile):
    """SABR volatility smile.

    https://en.wikipedia.org/wiki/SABR_volatility_model

    Parameters
    ----------
    tau : float
        maturity, in years
    forward : float
    init_vola : float
    volvol : float
    rho : float
    beta : int
        1 is good
    """
    def __init__(self,
                 tau: float,
                 forward: float,
                 init_vola: float = 1.0,
                 volvol: float = 0.5,
                 rho: float = 0.0,
                 beta: float = 1):
        # shorthand
        f, v, a, r, b = \
            forward, volvol, init_vola, rho, beta

        def chi(x_):
            res_ = np.log(
                (np.sqrt(1 - 2 * r * x_ + x_ ** 2) + x_ - r) / (1 - r)
            )
            return res_

        def mapping(k):

            z = v / a * (f * k) ** ((1 - b) / 2) * np.log(f / k)

            res_ = a / \
                (
                    (f * k) ** ((1 - b) / 2) *
                    (
                        1 +
                        ((1 - b) ** 2 / 24) * np.log(f / k) ** 2 +
                        ((1 - b) ** 4 / 1920) * np.log(f / k) ** 4
                    )
                ) * \
                (z / chi(z)) * \
                (
                    1 + (
                        (1 - b) ** 2 / 24 * a ** 2 / (f * k) ** (1 - b) +
                        0.25 * (r * b * v * a / (f * k) ** ((1 - b) / 2)) +
                        (2 - 3 * r ** 2) / 24 * v ** 2
                    ) * self.tau
                )

            res_ = np.where(
                np.equal(k, f),
                a / f ** (1 - b) * (
                    1 +
                    (
                        (1 - b) ** 2 / 24 * a ** 2 / f ** (2 - 2 * b) +
                        0.25 * r * b * a * v / f ** (1 - b) +
                        (2 - 3 * r ** 2) / 24 * v ** 2
                    ) * self.tau
                ),
                res_
            )

            return res_

        super(SABR, self).__init__(tau=tau, mapping=mapping)

        self.forward = forward
        self.tau = tau
        self.init_vola = init_vola
        self.volvol = volvol
        self.rho = rho
        self.beta = beta

    def __str__(self):
        b, a, v, r = self.beta, np.round(self.init_vola, 2), \
            np.round(self.volvol, 2), np.round(self.rho, 2)
        res = f"SABR (beta={b}) with a={a}, volvol={v}, rho={r}."

        return res

    @classmethod
    def fit_to_fx(cls, tau, v_atm: float, contracts: dict,
                  spot=None, forward=None, r_counter=None, r_base=None,
                  delta_conventions=None, beta=1.0):
        """Fit to FX contracts (ATM, market strangles and risk reversals).

        Fully based on the algorithm in Clark (2011), ch. 3.7.1

        Parameters
        ----------
        tau : float
            maturity, in years
        v_atm : float
            at-the-money volatility
        contracts : dict
            {delta: {'ms': float, 'rr': float}}, where
                - delta: float, in frac of 1;
                - 'ms' keys the quote of the market strangle (usually as a
                premium to the atm vola);
                - 'rr' keys the quoteof the risk reversal (call vola minus
                put vola);
            follows 'Foreign Exchange Option Pricing' by Clark (2011), ch.3
        spot : float
            spot quote
        forward : float
            forward outright quote
        r_counter : float
            risk-free rate in the counter currency, cont. comp.,
            in frac of 1 p.a.
        r_base : float
            risk-free rate in the base currency, cont. comp.,
            in frac of 1 p.a.
        delta_conventions : dict
            {'atm_def': str, 'is_forward': bool, 'is_premiumadj': bool}
        beta : float
            beta parameter in SABR
        """
        # number of different deltas
        n = len(contracts)

        # unpack conventions
        is_forward = delta_conventions["is_forward"]
        is_premiumadj = delta_conventions["is_premiumadj"]
        atm_def = delta_conventions["atm_def"]

        # pack all the other data as a shorthand
        data_rest = {"spot": spot,
                     "forward": forward,
                     "r_counter": r_counter,
                     "r_base": r_base}

        # determine atm strike; SABR IV at this level should match `v_atm`
        k_atm = strike_from_atm(atm_def, is_premiumadj=is_premiumadj,
                                spot=spot, forward=forward, vola=v_atm,
                                tau=tau)

        d = []  # deltas
        k_ms = []  # market strangle strikes as [d1 call, put; d2 call, put...]
        v_tgt = []  # market strangle price (premium) as [d1 v; d2 v...]
        sigma_ms = []  # market strangle quote; same
        sigma_rr = []  # risk reversal quote; same

        for d_, c_ in contracts.items():
            # for each delta, evaluate the above
            sigma_ms_ = v_atm + c_["ms"]

            # call delta is d_, put delta is -d_
            k_ms_ = strike_from_delta(np.array([d_, -d_]), tau, sigma_ms_,
                                      is_call=np.array([True, False]),
                                      is_forward=is_forward,
                                      is_premiumadj=is_premiumadj,
                                      **data_rest)

            # market strangle price (premium) is the sum of call and put;
            v_tgt_ = bs_price(strike=k_ms_, tau=tau, vola=sigma_ms_,
                              is_call=np.array([True, False]),
                              forward=forward, r_counter=r_counter).sum()

            d.append(d_)
            k_ms += k_ms_.tolist()
            v_tgt.append(v_tgt_)
            sigma_ms.append(sigma_ms_)
            sigma_rr.append(c_["rr"])

        # convert every list to array
        d = np.array(d)
        k_ms = np.array(k_ms)
        v_tgt = np.array(v_tgt)
        sigma_ms = np.array(sigma_ms)
        sigma_rr = np.array(sigma_rr)

        def get_sabr(par_) -> callable:
            """SABR getter.

            Parameters
            ----------
            par_: tuple
                (init_vola, volvol, rho)
            """
            res_ = SABR(
                forward=data_rest["forward"], tau=tau, beta=beta,
                init_vola=par_[0], volvol=par_[1], rho=par_[2]
            )

            return res_

        # main objective
        def main_obj(par_):
            sigma_ss = par_[3:]
            sabr_par = par_[:3]
            sabr_ = get_sabr(sabr_par)

            # functions
            def k_obj(x_):
                k_ = strike_from_delta(
                    np.concatenate(list(zip(d, -d))),
                    tau=tau, vola=sabr_(x_),
                    is_call=np.array([True, False] * n),
                    is_forward=is_forward,
                    is_premiumadj=is_premiumadj,
                    **data_rest
                )
                res__ = x_ - k_
                return res__

            # solve for market strangle strikes
            k = fsolve(k_obj, x0=np.array([k_atm, ] * 2 * n))

            sigma_x_atm = sabr_(k_atm)
            sigma_x = sabr_(k)
            sigma_x_rr = sigma_x[0::2] - sigma_x[1::2]
            sigma_x_ss = 0.5 * (sigma_x[0::2] + sigma_x[1::2]) - sigma_x_atm

            v_x = bs_price(strike=k_ms, tau=tau, vola=sabr_(k_ms),
                           is_call=np.array([True, False] * n),
                           forward=forward, r_counter=r_counter) \
                .reshape(-1, 2) \
                .sum(axis=1)

            val_model = np.concatenate(
                ([sigma_x_atm], sigma_x_rr, sigma_x_ss, v_x)
            )
            val_observed = np.concatenate(
                ([v_atm], sigma_rr, sigma_ss, v_tgt)
            )
            res_ = val_model - val_observed

            return res_

        # bounds: vol and volvol are positive; -1 < rho < 1
        bounds = [
            (0, 0, -1, *np.zeros(n).tolist()),
            (np.inf, np.inf, 1, *(np.ones(n) + (np.inf)).tolist())
        ]

        # start with the default optimization method
        par_t = least_squares(main_obj,
                              x0=np.array([v_atm, v_atm, 0.0, *sigma_ms]),
                              bounds=bounds)

        # check if the solution makes sense (-1 < rho < 1);
        # if not, use levenberg-marquardt
        if (not par_t.success) or (np.abs(par_t.x[2]) > 0.99):
            par_t = least_squares(main_obj,
                                  x0=np.array([v_atm, 0.5, 0.0, *sigma_ms]),
                                  method="lm")

        # the resulting SABR
        sabr = get_sabr(par_t.x)

        # # checks
        # v_trial = bs_price(strike=k_ms, tau=tau,
        #                    vola=sabr(k_ms),
        #                    is_call=np.array([True, False] * n),
        #                    **data_rest) \
        #     .reshape(-1, 2) \
        #     .sum(axis=1)

        # print("v_trial:\n")
        # print(v_trial)
        # print("v_tgt:\n")
        # print(v_tgt)

        return sabr
