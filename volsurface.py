import pandas as pd
import numpy as np
import copy
from functools import reduce
from scipy.interpolate import CubicSpline
from statsmodels.nonparametric.kernel_regression import KernelReg
import matplotlib.pyplot as plt
from optools.helpers import strike_range
from mpl_toolkits.mplot3d import Axes3D

from optools.pricing import (bs_price, strike_from_delta, mfivariance,
                             mfiskewness, vanillas_from_combinations)
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
    def __init__(self, vola_series, spot=None, forward=None, rf=None,
                 div_yield=None, tau=None):
        """
        """
        # sort, convert to float, rename
        smile = vola_series.astype(float).sort_index().rename(tau)

        # save to attributes
        self.smile = smile

        self.vola = smile.values
        self.strike = np.array(smile.index)
        self.spot = spot
        self.forward = forward
        self.rf = rf
        self.div_yield = div_yield
        self.tau = tau

    def dropna(self, from_index=False):
        """
        """
        res = copy.deepcopy(self)

        if from_index:
            res.smile = res.smile.loc[res.smile.index.dropna()]
            res.vola = res.smile.values
            res.strike = np.array(res.smile.index)
        else:
            res.smile = res.smile.dropna()

        return res

    def __repr__(self):
        """
        """
        return self.smile.__repr__()

    def __str__(self):
        """
        """
        return str(self.smile)

    @classmethod
    def by_delta(cls, vola_series, spot, forward, rf, div_yield, tau,
                 is_call):
        """Construct VolatilitySmile from delta-vola relation.

        Converts Balck-Scholes deltas to strikes (see Wystup (2006), eq. 1.44)
        and use those to construct a VolatilitySmile

        Parameters
        ----------
        vola_series: pandas.Series
            of vola, in percent p.a., indexed by option deltas, in frac of 1
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
        # rename index to be able to pick it up as a column later
        vola_series.index.name = "delta"

        # strikes from deltas
        strike = strike_from_delta(vola_series.index, spot, rf, div_yield, tau,
                                   vola_series.values, is_call)

        # concat vola-by-delta and strikes to have the delta-to-strike mapping
        df = pd.concat({
            "vola": vola_series,
            "strike": pd.Series(strike, index=vola_series.index)},
            axis=1).reset_index().set_index("strike").sort_index()

        res = cls(df.loc[:, "vola"], spot, forward, rf, div_yield, tau)

        res.delta = df.loc[:, "delta"]

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
            volas.append(vanillas_from_combinations(atm=atm_vola,
                                                    delta=k, **v))

        volas = pd.concat(volas)

        # add delta of atm (slightly different than 0.5, as in
        #   Wystup (2006), eq. 1.96)
        atm_delta = np.exp(-div_yield * tau) * \
            fast_norm_cdf(0.5 * atm_vola * np.sqrt(tau))

        volas.loc[atm_delta] = atm_vola

        res = cls.by_delta(volas,
                           spot, forward, rf, div_yield, tau, is_call=True)

        return res

    @staticmethod
    def interpolate_by_delta(delta, delta_new, delta_atm, sigma_atm, sigma_s,
                             sigma_r):
        """

        Parameters
        ----------
        delta
        delta_new
        delta_atm
        sigma_atm
        sigma_s
        sigma_r

        Returns
        -------

        """
        a = 1.0

        c1_num = a**2*(2*sigma_s + sigma_r) -\
            2*a*(2*sigma_s + sigma_r)*(delta + delta_atm) +\
            2*(delta**2*sigma_r +
                4*sigma_s*delta*delta_atm +
                sigma_r*delta_atm**2)

        c1_den = (2*(2*delta - a)*(delta - delta_atm)*(delta - a + delta_atm))
        c1 = c1_num / c1_den

        c2_num = 4*delta*sigma_s -\
            a*(2*sigma_s + sigma_r) +\
            2*sigma_r*delta_atm
        c2_den = 2*(2*delta - a)*(delta - delta_atm)*(delta - a + delta_atm)
        c2 = c2_num / c2_den

        res = sigma_atm + c1*(delta_new - delta_atm) +\
            c2*(delta_new - delta_atm)**2

        return res

    def interpolate(self, new_strike=None, in_method="spline",
                    ex_method="constant", **kwargs):
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
        # defaults
        if new_strike is None:
            new_strike = strike_range(self.strike)

        # interpolate -------------------------------------------------------
        if in_method == "spline":
            # estimate
            cs = CubicSpline(self.strike, self.vola,
                             extrapolate=False, **kwargs)
            # fit
            vola_interpolated = cs(new_strike)

        elif in_method == "kernel":
            # estimate endog must be a list of one element
            kr = KernelReg(endog=[self.vola, ], exog=[self.strike, ],
                           reg_type="ll", var_type=['c', ])

            # fit
            vola_interpolated, _ = kr.fit(data_predict=new_strike)

        else:
            raise NotImplementedError("Interpolation method not implemented!")

        # extrapolate -------------------------------------------------------
        if ex_method is None:
            pass

        elif ex_method == "constant":
            # use pandas.Series functionality to extrapolate but check if the
            #   strikes are sorted first
            tmp = pd.Series(index=new_strike, data=vola_interpolated)
            if not np.array_equal(tmp.index, sorted(tmp.index)):
                raise ValueError("Strikes not sorted before extrapolation!")

            # fill beyond endpoints
            tmp.loc[tmp.index < min(self.strike)] = tmp.loc[min(self.strike)]
            tmp.loc[tmp.index > max(self.strike)] = tmp.loc[max(self.strike)]

            # disassemble again
            new_strike = np.array(tmp.index)
            vola_interpolated = tmp.values

        else:
            raise NotImplementedError("Extrapolation method not implemented!")

        # construct another VolatilitySmile instance
        vola_series = pd.Series(vola_interpolated, index=new_strike)
        res = VolatilitySmile(vola_series,
                              spot=self.spot,
                              forward=self.forward, rf=self.rf,
                              div_yield=self.div_yield, tau=self.tau)

        return res

    def get_mfivariance(self):
        """Calculate the model-free implied variance.

        The mfiv is calculated as the integral over call prices weighted by
        strikes (for details see Jiang and Tian (2005)). This method first
        transforms the volas to the prices of vanillas, then does the
        integration using Simpson's rule.

        Parameters
        ----------

        Returns
        -------
        res : float
            mfiv, in (frac of 1) p.a.

        """
        # from volas to call prices
        call_p = bs_price(forward=self.forward, strike=self.strike,
                          rf=self.rf, tau=self.tau, vola=self.vola)

        # mfiv
        res = mfivariance(call_p, self.strike, self.forward, self.rf, self.tau)

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


class VolatilitySurface:
    """
    """
    def __init__(self, vola_df, spot=None, forward=None, rf=None,
                 div_yield=None):
        """

        Parameters
        ----------
        vola_df : pandas.DataFrame
        """
        # sort a bit
        self.surface = vola_df.sort_index(axis=0).sort_index(axis=1)

        self.vola = self.surface.values
        self.strike = np.array(self.surface.index)
        self.tau = np.array(self.surface.columns)

        self.spot = spot
        self.forward = forward
        self.rf = rf
        self.div_yield = div_yield

    @property
    def smiles(self):
        res = {
            t: VolatilitySmile(v.dropna(),
                               spot=self.spot.get(t, None),
                               forward=self.forward.get(t, None),
                               rf=self.rf.get(t, None),
                               div_yield=self.div_yield.get(t, None),
                               tau=t)
            for t, v in self.surface.iteritems()
        }

        return res

    def plot(self, **kwargs):
        """Plot the surface in 3d.

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
            fig = plt.figure()
        else:
            fig = ax.figure

        # plot
        ax = fig.add_subplot(111, projection='3d')
        k, t = np.meshgrid(self.strike, self.tau)

        ax.plot_surface(k, t, self.vola.T)

        ax.set_xlabel('strike')
        ax.set_ylabel('tau')
        ax.set_zlabel('vola')

        return fig, ax

    def interpolate_along_tau(self):
        """

        Parameters
        ----------

        Returns
        -------
        res : VolatilitySurface
            a new instance, with interpolated data

        """
        # assert isinstance(self.forward, pd.Series)
        # self.forward = self.forward.reindex(index=self.tau)

        # loop over maturities: for each, there are five unique (in case of
        # fx) strikes, so it is possible to fill up to five new strikes in
        # each other column
        imputed_all = list()

        for tau_star, tau_star_col in self.surface.iteritems():
            # fetch self.forward corresponding to this maturity
            f_star = self.forward.loc[tau_star]
            # strikes in this column
            k_star = np.array(tau_star_col.dropna().index)

            # loop over the other columns (the two closest at most)
            # closest_cols = pd.Series({tau_star: tau_star}).reindex(
            #     index=self.tau).sort_index()
            # closest_cols = closest_cols.ffill(limit=1).bfill(limit=1).dropna()
            # closest_cols = [p for p in closest_cols.index if p != tau_star]
            closest_cols = self.surface.columns

            imputed_tau_star = dict()

            for tau_new, tau_new_col in\
                    self.surface.loc[:, closest_cols].iteritems():

                # fetch self.forward corresponding to this maturity (new one)
                f_new = self.forward.loc[tau_new]

                # calculate new strikes for this new maturity
                k_new = f_new * (k_star / f_star)**(np.sqrt(tau_new/tau_star))

                # calculate new sigmas
                sigma_new = self.surface.loc[f_new, tau_new] +\
                    self.surface.loc[k_star, tau_star].values - \
                    self.surface.loc[f_star, tau_star]

                # assemble in a series
                imputed_tau_star[tau_new] = pd.Series(index=k_new,
                                                      data=sigma_new)

            # concat all freshly imputed dfs
            imputed_all.append(pd.concat(imputed_tau_star, axis=1))

        # merge all dfs with reduce
        def reduce_func(x, y):
            x_new, y_new = x.align(y, join="outer")
            return x_new.fillna(y_new)

        res = VolatilitySurface(
            vola_df=reduce(reduce_func, imputed_all),
            forward=self.forward, spot=self.spot, rf=self.rf,
            div_yield=self.div_yield)

        return res

    def extrapolate(self, other):
        """

        Parameters
        ----------
        other : VolatilitySurface

        Returns
        -------

        """
        # copy surfaces
        self_surf, other_surf = self.surface.align(other.surface,
                                                   axis=0, join="outer")

        # fill na smile by smile, leaving everything within the eisting
        # smile boundaries as is
        for t, smile in self_surf.iteritems():
            # existign smile boundaries
            s_idx = smile.first_valid_index()
            e_idx = smile.last_valid_index()

            # fillna
            self_surf.loc[:, t].fillna(
                other_surf.loc[other_surf.index < s_idx, t], inplace=True)
            self_surf.loc[:, t].fillna(
                other_surf.loc[other_surf.index > e_idx, t], inplace=True)

        res = VolatilitySurface(self_surf, forward=self.forward,
                                spot=self.spot, rf=self.rf,
                                div_yield=self.div_yield)

        return res

    def interpolate(self, new_strike=None, **kwargs):
        """

        Returns
        -------

        """
        if new_strike is None:
            new_strike = {}

        smiles_interp = pd.DataFrame(
            {
                t: v.interpolate(new_strike=new_strike.get(t, None),
                                 **kwargs).smile
                for t, v in self.smiles.items()
            }
        )

        return VolatilitySurface(vola_df=smiles_interp, spot=self.spot,
                                 forward=self.forward, rf=self.rf,
                                 div_yield=self.div_yield)

    def get_mfivariance(self):
        """

        Returns
        -------

        """
        res = pd.Series({t: v.get_mfivariance()
                         for t, v in self.smiles.items()})

        return res


if __name__ == "__main__":
    # vola = np.array([0.08, 0.10, 0.07, 0.068, 0.075])
    vola = np.array([0.08, ]*5)
    strike = np.array([0.9, 0.8, 0.95, 1.1, 1.21])
    vs = VolatilitySmile(pd.Series(vola, index=strike), spot=1.0,
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

