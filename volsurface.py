import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
from scipy import interpolate
import matplotlib.pyplot as plt

import optools as ops


class VolatilitySmile():
    """

    Parameters
    ----------
    vola: numpy.ndarray
        implied vol
    strike: numpy.ndarray
        of option deltas, in (frac of 1)
    spot: float
        underlying price
    rf: float
        risk-free rate, in (frac of 1) p.a.
    div_yield: float
        dividend yield, in (frac of 1) p.a.
    tau: float
        time to maturity, in years
    is_call: bool
        whether options are call options

    """
    def __init__(self, vola, strike, rf, div_yield, tau):
        """
        """
        self.vola = vola.astype(np.float)
        self.strike = strike.astype(np.float)
        self.rf = rf
        self.div_yield = div_yield
        self.tau = tau

    @property
    def smile(self):
        # construct a Series
        res = pd.Series(data=self.vola, index=self.strike).rename("vola")

        # sort strikes
        res = res.loc[sorted(res.index)]

        return res

    @classmethod
    def by_delta(cls, vola, delta, spot, rf, div_yield, tau, is_call):
        """Construct volatility smile from delta-vola relation.

        Parameters
        ----------
        vola: numpy.ndarray
            implied vol
        delta: numpy.ndarray
            of option deltas, in (frac of 1)
        spot: float
            underlying price
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
        strike = ops.strike_from_delta(delta, spot, rf, div_yield, tau,
                                       vola, is_call)

        res = cls(vola, strike, rf, div_yield, tau)

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

        Returns
        -------

        """
        # reindex
        new_idx = pd.Index(new_strike.astype(np.float)).union(self.smile.index)
        smile_reixed = self.smile.reindex(index=new_idx)

        # interpolate
        if in_method == "spline":
            # res = smile_reixed.interpolate(method="spline", ext=ex_method,
            #                                order=3, **kwargs)

            # estimate
            cs = CubicSpline(x=self.smile.index, y=self.smile.values,
                             bc_type="clamped", extrapolate=False)
            # fit
            new_y = cs(new_idx)

            res = pd.Series(index=new_idx, data=new_y)

        else:
            raise NotImplementedError("Not implemented!")

        # extrapolate
        if ex_method == "constant":
            res.where(pd.Series(True, index=new_idx[new_idx < min()]))

        return res

    def plot(self):
        """

        Returns
        -------

        """
        return self.smile.plot()


if __name__ == "__main__":
    vola = np.array([0.10, 0.08, 0.07, 0.068, 0.075])
    strike = np.array([0.8, 0.9, 0.95, 1.1, 1.21])
    vs = VolatilitySmile(vola, strike, 0.01, 0.001, 0.25)
    # vs.plot()
    # plt.show()
    vs.interpolate(new_strike=np.linspace(0.6, 1.5, 100)).plot()
    plt.show()

