import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
from statsmodels.nonparametric.kernel_regression import KernelReg
import matplotlib.pyplot as plt

from optools.functions import (bs_price, strike_from_delta, mfivariance,
                               mfiskewness, wings_iv_from_combies_iv)
from optools.helpers import fast_norm_cdf


class VolatilitySmile:
    """Volatility smile as a mapping from strike to vola.

    In the simplest specification, only volas and strikes are needed; optional
    arguments are spot and forward prices etc., making computation of mfiv
    and other values possible.

    All options are by defaults call options.

    TODO: think about forward, spot, rf and div_yield defaults.

    Parameters
    ----------
    vola: numpy.ndarray
        implied vol
    strike: numpy.ndarray
        of option strike prices
    spot: float, optional
        underlying price
    forward : float, optional
        forward price
    rf: float, optional
        risk-free rate, in (frac of 1) p.a.
    div_yield: float, optional
        dividend yield, in (frac of 1) p.a.
    tau: float, optional
        time to maturity, in years

    """
    def __init__(self, vola, strike, spot=None, forward=None, rf=None,
                 div_yield=None, tau=None):
        """
        """
        # construct a pandas.Series -> easier to plot and order
        smile = pd.Series(data=vola.astype(float),
                          index=strike.astype(float))

        # sort index, rename
        smile = smile.loc[sorted(smile.index)].rename("vola")

        # # default forward or spot?
        # if forward is None:
        #     forward = spot * np.exp((rf - div_yield) * tau)

        # save to attributes
        self.smile = smile

        self.vola = smile.values
        self.strike = np.array(smile.index)
        self.spot = spot
        self.forward = forward
        self.rf = rf
        self.div_yield = div_yield
        self.tau = tau

    def __str__(self):
        """
        """
        return str(self.smile)

    @classmethod
    def by_delta(cls, vola, delta, spot, forward, rf, div_yield, tau, is_call):
        """Construct VolatilitySmile from delta-vola relation.

        Converts Balck-Scholes deltas to strikes (see Wystup (2006), eq. 1.44)
        and use those to construct a VolatilitySmile

        Parameters
        ----------
        vola: numpy.ndarray
            implied vol
        delta: numpy.ndarray
            of option deltas, in (frac of 1)
        spot: float
            underlying price
        forward : float
            forward price
        rf: float
            risk-free rate, in (frac of 1) p.a.
        div_yield: float
            dividend yield, in (frac of 1) p.a.
        tau: float
            time to maturity, in years
        is_call: bool
            whether options are call options

        Returns
        -------
        res : VolatilitySmile
            instance

        """
        strike = strike_from_delta(delta, spot, rf, div_yield, tau,
                                   vola, is_call)

        res = cls(vola, strike, spot, forward, rf, div_yield, tau)

        return res

    @classmethod
    def by_delta_from_combinations(cls, combies, atm_vola, spot, forward, rf,
                                   div_yield, tau):
        """Construct VolatilitySmile from delta-vola of option combinations.

        Essentially a wrapper around .by_delta() condtructor, conveniently
        accepting deltas and volas of option contracts such as risk reversals,
        butterfly spreads and at-the-money vanillas (details are in
        Wystup (2006), pp. 22-23). The first step is to recover the vola of
        the underlying vanilla call options (eqs. 1.97-1.100 in Wystup (
        2006)), the second step is to apply .to_delta to these.

        Parameters
        ----------
        combies : dict
            of (delta: combi) pairs where combi is a dict-like as follows:
                {'rr': iv of the risk reversal,
                 'bf': iv of the butterfly}
            all ivs are in (frac of 1) p.a.
        atm_vola : float
            at-the-money call volatility, in (frac of 1) p.a.
        spot: float
            underlying price
        forward : float
            forward price
        rf: float
            risk-free rate, in (frac of 1) p.a.
        div_yield : float
            div yield (rf rate of the base currency), in (frac of 1) p.a.
        tau : float
            time to maturity, in years

        Returns
        -------
        res : VolatilitySmile
            instance

        """
        # for each delta, calculate iv of respective call, concat to a Series
        volas = list()

        for k, v in combies.items():
            volas.append(wings_iv_from_combies_iv(atm=atm_vola, delta=k, **v))

        volas = pd.concat(volas)

        # add delta of atm (slightly different than 0.5, as in
        #   Wystup (2006), eq. 1.96)
        atm_delta = np.exp(-div_yield * tau) * \
            fast_norm_cdf(0.5 * atm_vola * np.sqrt(tau))

        volas.loc[atm_delta] = atm_vola

        res = cls.by_delta(volas.values, np.array(volas.index),
                           spot, forward, rf, div_yield, tau, is_call=True)

        return res

    def interpolate(self, new_strike, in_method="spline",
                    ex_method="const", **kwargs):
        """Interpolate volatility smile.

        Spline interpolation (exact fit to existing data) or kernel
        regression interpolation (approximate fit to existing data) is
        implemented.

        Parameters
        ----------
        new_strike : numpy.ndarray
            of strike prices over which the interpolation takes place
        in_method : str
            method of interpolation; 'spline' and 'kernel' is supported
        ex_method : str or None
            method of extrapolation; None to skip extrapolation, 'const' for
            extrapolation with endpoint values
        **kwargs : any
            additional argument to the interpolation function,
            e.g. bc_type='clamped' for a clamped smile

        Returns
        -------
        res : VolatilitySmile
            a new instance of VolatilitySmile
        """
        # reindex, assign a socialistic name; this will be sorted!
        strike_union = np.union1d(self.strike, new_strike).astype(np.float)

        # interpolate -------------------------------------------------------
        if in_method == "spline":
            # estimate
            cs = CubicSpline(self.strike, self.vola,
                             extrapolate=False, **kwargs)
            # fit
            vola_interpolated = cs(strike_union)

        elif in_method == "kernel":
            # estimate endog must be a list of one element
            kr = KernelReg(endog=[self.vola, ], exog=[self.strike, ],
                           reg_type="ll", var_type=['c', ])

            # fit
            vola_interpolated, _ = kr.fit(data_predict=strike_union)

        else:
            raise NotImplementedError("Interpolation method not implemented!")

        # extrapolate -------------------------------------------------------
        if ex_method is None:
            pass

        elif ex_method == "constant":
            # use pandas.Series functionality to extrapolate but check if the
            #   strikes are sorted first
            tmp = pd.Series(index=strike_union, data=vola_interpolated)
            if not np.array_equal(tmp.index, sorted(tmp.index)):
                raise ValueError("Strikes not sorted before extrapolation!")

            # fill beyond endpoints
            tmp.loc[tmp.index < min(self.strike)] = tmp.loc[min(self.strike)]
            tmp.loc[tmp.index > max(self.strike)] = tmp.loc[max(self.strike)]

            # disassemble again
            strike_union = np.array(tmp.index)
            vola_interpolated = tmp.values

        else:
            raise NotImplementedError("Extrapolation method not implemented!")

        # construct another VolatilitySmile instance
        res = VolatilitySmile(vola=vola_interpolated, strike=strike_union,
                              spot=self.spot,
                              forward=self.forward, rf=self.rf,
                              div_yield=self.div_yield, tau=self.tau)

        return res

    def get_mfivariance(self, method="jiang_tian"):
        """Calculate the model-free implied variance.

        The mfiv is calculated as the integral over call prices weighted by
        strikes (for details see Jiang and Tian (2005)). This method first
        transforms the volas to the prices of vanillas, then does the
        integration using Simpson's rule.

        Parameters
        ----------
        method : str
            'jiang_tian' (integral with call options only) and 'sarno' (both
            call and put options) currently implemented

        Returns
        -------
        res : float
            mfiv, in (frac of 1) p.a.

        """
        # from volas to call prices
        call_p = bs_price(forward=self.forward, strike=self.strike,
                          rf=self.rf, tau=self.tau, vola=self.vola)

        # mfiv
        res = mfivariance(call_p, self.strike, self.forward, self.rf, self.tau,
                          method=method)

        return res

    def get_mfiskewness(self):
        """

        Returns
        -------

        """
        # from volas to call prices
        call_p = bs_price(forward=self.forward, strike=self.strike,
                          rf=self.rf, tau=self.tau, vola=self.vola)

        # mfiv
        res = mfiskewness(call_p=call_p, strike=self.strike, spot=self.spot,
                          forward=self.forward, rf=self.rf, tau=self.tau)

        return res

    def plot(self, **kwargs):
        """Plot the smile.

        Parameters
        ----------
        **kwargs : any
            arguments to matplotlib.pyplot.plot(); can contain an instance
            of Axes to iuse for plotting

        Returns
        -------
        fig : matplotlib.pyplot.figure
        ax : matplotlib.pyplot.Axes

        """
        # watch out for cases when `ax` was provided in kwargs
        ax = kwargs.pop("ax", None)

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        # plot
        self.smile.plot(ax=ax, **kwargs)

        return fig, ax


if __name__ == "__main__":
    # vola = np.array([0.08, 0.10, 0.07, 0.068, 0.075])
    vola = np.array([0.08, ]*5)
    strike = np.array([0.9, 0.8, 0.95, 1.1, 1.21])
    vs = VolatilitySmile(vola, strike, spot=1.0,
                         forward=1.0*np.exp((0.01 - 0.001)*0.25), rf=0.01,
                         div_yield=.001, tau=0.25)

    # clamped
    smile_in = vs.interpolate(new_strike=np.linspace(0.6, 1.5, 100),
                              ex_method="constant", bc_type="clamped")
    fig, ax = smile_in.plot(color="blue")
    mfiv_1 = smile_in.get_mfivariance()
    skew_1 = smile_in.get_mfiskewness()

    # # natural
    # smile_in = vs.interpolate(new_strike=np.linspace(0.6, 1.5, 100),
    #                           ex_method="constant", bc_type="natural")
    # fig, ax = smile_in.plot(ax=ax, color="black")
    # mfiv_2 = smile_in.get_mfivariance()

    # # kernel
    # smile_in = vs.interpolate(new_strike=np.linspace(0.6, 1.5, 100),
    #                           in_method="kernel",
    #                           ex_method="constant")
    # fig, ax = smile_in.plot(ax=ax, color="black")

    # vs.plot(ax=ax, linestyle="none", marker='o', color="red")

    # plt.show()

    # print([mfiv_1, mfiv_2])

