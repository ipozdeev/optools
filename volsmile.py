import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.misc import derivative
from statsmodels.nonparametric.kernel_regression import KernelReg
import matplotlib.pyplot as plt
from numba import jit

from .helpers import construct_new_x
from .blackscholes import option_price
from .implied import mfivariance, simple_var_swap_rate


class VolatilitySmile:
    """Volatility smile as a mapping from strike to vola.

    In the simplest specification, only volas and strikes are needed; optional
    arguments are spot and forward prices etc., making computation of mfiv
    and other values possible.

    All options are by defaults call options.

    Parameters
    ----------
    vol_series: pandas.Series
        vol smile, indexed by strike or delta
    tau: float, optional
        time to maturity, in years

    """
    def __init__(self, vol_series=None, strike=None, delta=None, vol=None,
                 tau=None):

        if vol_series is not None:
            if not all((strike is None, delta is None, vol is None)):
                raise ValueError("if `vol_series` is defined, impossible "
                                 "to have any of (strike, delta, vol)")

            if vol_series.index.name not in ("strike", "delta"):
                raise ValueError("index name must be one of "
                                 "('strike', 'delta')")

            # sort, convert to float, rename
            smile = vol_series.astype(float)

        else:
            if vol is None:
                raise ValueError("`vol` must be defined")
            if not (strike is None) ^ (delta is None):
                raise ValueError("only one of (`strike`, `delta`) "
                                 "can be defined")
            if strike is not None:
                smile = pd.Series(index=pd.Index(strike, name="strike"),
                                  data=vol, dtype=float)
            if delta is not None:
                smile = pd.Series(index=pd.Index(delta, name="strike"),
                                  data=vol, dtype=float)

        # save to attributes
        smile = smile.sort_index()
        self.smile = smile

        self.vola = smile.values
        self.x = np.array(smile.index)
        self.tau = tau

    @property
    def by_strike(self):
        return self.smile.index.name == "strike"

    def __str__(self):
        """
        """
        return str(self.smile)

    def interpolate(self, kind="cubic", new_x=None, extrapolate=False,
                    **kwargs):
        """Interpolate volatility smile.

        Spline interpolation (exact fit to existing data) or kernel
        regression interpolation (approximate fit to existing data) is
        implemented.

        Parameters
        ----------
        kind : str
            'kernel' or one of valid kinds in interp1d, e.g. 'cubic'
        new_x : numpy.ndarray
            of strikes or deltas over which the interpolation takes place
        extrapolate : bool
            False to extrapolate with endpoint values
        kwargs
            additional arguments to scipy.interpolate.interp1d

        Returns
        -------
        res : VolatilitySmile
            a new instance of VolatilitySmile
        """
        # defaults
        if new_x is None:
            if extrapolate:
                scale = 1 if self.tau is None else np.sqrt(self.tau)
                x_min = min(self.x) - max(self.vola) * scale * 3
                x_max = max(self.x) + max(self.vola) * scale * 3
            else:
                x_min = min(self.x)
                x_max = max(self.x)

            new_x = construct_new_x(self.x, x_min=x_min, x_max=x_max)

        # interpolate -------------------------------------------------------
        if kind == "kernel":
            # estimate endog must be a list of one element
            kr = KernelReg(endog=self.vola, exog=self.x, reg_type="ll",
                           var_type='c')

            # fit
            new_vol, _ = kr.fit(data_predict=new_x)

        else:
            # if extrapolation is w/constant values
            min_v, max_v = self.smile.iloc[0], self.smile.iloc[-1]

            # estimate
            f = interp1d(self.x, self.vola, kind=kind,
                         bounds_error=(not extrapolate),
                         fill_value=(min_v, max_v), **kwargs)

            # fit
            new_vol = f(new_x)

        # construct another VolatilitySmile instance
        vol_series = pd.Series(
            new_vol, index=pd.Index(new_x, name=self.smile.index.name)
        )
        res = VolatilitySmile(vol_series, tau=self.tau)

        return res

    def get_mfivariance(self, forward, rf, svix=False):
        """Calculate the model-free implied variance.

        The mfiv is calculated as the integral over call prices weighted by
        strikes (for details see Jiang and Tian (2005) and Martin (2017)).
        The volas are transformed to the prices of vanillas, then the
        integration is done using Simpson's rule.

        Parameters
        ----------
        svix : bool
            True to calculate Martin (2017) simple variance swap rates

        Returns
        -------
        res : float
            mfiv, in (frac of 1) p.a.

        """
        if self.tau is None:
            raise ValueError("parameter `tau` must be set on instantiation.")

        # if strike price is not on the x-axis, need to convert
        if not self.by_strike:
            raise ValueError("the smile must be by strike price. convert "
                             "deltas to strikes first using "
                             "`greeks.strike_from_delta`.")

        # from volas to call prices
        call_p = option_price(forward=forward, strike=self.x,
                              rf=rf, tau=self.tau, vol=self.vola)

        # mfiv
        if svix:
            res = simple_var_swap_rate(call_p, self.x, forward, rf, self.tau)
        else:
            res = mfivariance(call_p, self.x, forward, rf, self.tau)

        return res

    def get_rnd(self, forward=None, rf=None, div_yield=None, spot=None,
                is_call=True, x: np.ndarray=None) -> callable:
        """
        """
        if not self.by_strike:
            raise ValueError("smile must defined over strike prices for this!")

        min_v, max_v = self.smile.iloc[0], self.smile.iloc[-1]

        # if extrapolation is w/constant values
        f = interp1d(self.x, self.vola, kind="cubic",
                     bounds_error=False,
                     fill_value=(min_v, max_v))

        # estimate
        def func_to_diff(x_):
            c_ = option_price(x_, rf, self.tau, f(x_), div_yield,
                              spot, forward, is_call)
            return c_

        def res(x_):
            res_ = derivative(func_to_diff, x_, dx=1e-04, n=2) \
                * np.exp(-rf * self.tau)
            return res_

        if x is not None:
            return res(x)

        return res


    def plot(self, **kwargs):
        """Plot the smile.

        Parameters
        ----------
        **kwargs : any
            arguments to matplotlib.pyplot.plot(); can contain an instance
            of Axes to use for plotting

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

