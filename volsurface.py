import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
from statsmodels.nonparametric.kernel_regression import KernelReg
import matplotlib.pyplot as plt

from optools.functions import (bs_price, strike_from_delta, mfiv,
                               wings_iv_from_combies_iv)
from optools.helpers import fast_norm_cdf


class VolatilitySmile():
    """

    TODO: think about foreward, spot, rf and div_yield - defaults?

    Parameters
    ----------
    vola: numpy.ndarray
        implied vol
    strike: numpy.ndarray
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

    """
    def __init__(self, vola, strike, spot, forward, rf, div_yield, tau,
                 is_call):
        """
        """
        # construct a pandas.Series -> easier to plot and order
        smile = pd.Series(data=vola.astype(float),
                          index=strike.astype(float))

        # sort index, rename
        smile = smile.loc[sorted(smile.index)].rename("vola")

        # default forward or spot?
        if forward is None:
            forward = spot * np.exp((rf - div_yield) * tau)

        # save to attributes
        self.smile = smile

        self.vola = smile.values
        self.strike = np.array(smile.index)
        self.spot = spot
        self.forward = forward
        self.rf = rf
        self.div_yield = div_yield
        self.tau = tau
        self.is_call = is_call

    @classmethod
    def by_delta(cls, vola, delta, spot, forward, rf, div_yield, tau, is_call):
        """Construct volatility smile from delta-vola relation.

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

        """
        strike = strike_from_delta(delta, spot, rf, div_yield, tau,
                                   vola, is_call)

        res = cls(vola, strike, spot, forward, rf, div_yield, tau, is_call)

        return res

    @classmethod
    def by_delta_from_combinations(cls, combies, atm_vola, spot, forward, rf,
                                   div_yield, tau):
        """

        Parameters
        ----------
        combies : dict
            of (delta: combi) pairs where combi is a Series as follows:
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
        res : any
            dependent on `return_as`

        """
        # for each delta, calculate iv of call, concat to a Series
        vola_series = pd.concat(
            [wings_iv_from_combies_iv(atm=atm_vola, delta=k, **v)
             for k, v in combies.items()],
            axis=0)

        # add delta of atm (slightly different one of the atm option)
        atm_delta = np.exp(-div_yield * tau) * \
            fast_norm_cdf(0.5 * atm_vola * np.sqrt(tau))

        vola_series.loc[atm_delta] = atm_vola

        res = cls.by_delta(vola_series.values, np.array(vola_series.index),
                           spot, forward, rf, div_yield, tau, is_call=True)

        return res

    def interpolate(self, new_strike, in_method="spline",
                    ex_method="const", **kwargs):
        """Interpolate volatility.

        Spline interpolation (exact fit to existing data) or kernel
        regression interpolation (approximate fit to existing data) is
        implemented.

        Parameters
        ----------
        new_strike : numpy.ndarray
            of strikes
        in_method : str
            method of interpolation; 'spline' and 'kernel' is supported
        ex_method : str
            method of extrapolation; 'const' is supported
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
        if ex_method == "constant":
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
        res = VolatilitySmile(vola_interpolated, strike_union,
                              self.spot, self.forward, self.rf,
                              self.div_yield, self.tau, self.is_call)

        return res

    def get_mfiv(self, method="jiang_tian"):
        """

        Parameters
        ----------
        method : str
            'jiang_tian' and 'sarno' currently implemented

        Returns
        -------
        res : float
            mfiv, in (frac of 1) p.a.

        """
        # from volas to call prices
        call = bs_price(self.forward, self.strike, self.rf, self.tau,
                        self.vola)

        # mfiv
        res = mfiv(call, self.strike, self.forward, self.rf, self.tau,
                   method=method)

        return res

    def plot(self, **kwargs):
        """Plot smile.

        Parameters
        ----------
        **kwargs : any
            arguments to matplotlib.pyplot.plot()

        Returns
        -------

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
    vola = np.array([0.08, 0.10, 0.07, 0.068, 0.075])
    strike = np.array([0.9, 0.8, 0.95, 1.1, 1.21])
    vs = VolatilitySmile(vola, strike, 1.0, None, 0.01, 0.001, 0.25, True)

    # clamped
    smile_in = vs.interpolate(new_strike=np.linspace(0.6, 1.5, 100),
                              ex_method="constant", bc_type="clamped")
    fig, ax = smile_in.plot(color="blue")
    mfiv_1 = smile_in.get_mfiv()

    # natural
    smile_in = vs.interpolate(new_strike=np.linspace(0.6, 1.5, 100),
                              ex_method="constant", bc_type="natural")
    fig, ax = smile_in.plot(ax=ax, color="black")
    mfiv_2 = smile_in.get_mfiv()

    # # kernel
    # smile_in = vs.interpolate(new_strike=np.linspace(0.6, 1.5, 100),
    #                           in_method="kernel",
    #                           ex_method="constant")
    # fig, ax = smile_in.plot(ax=ax, color="black")

    vs.plot(ax=ax, linestyle="none", marker='o', color="red")

    plt.show()

    print([mfiv_1, mfiv_2])

