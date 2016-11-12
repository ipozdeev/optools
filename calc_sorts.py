# tests
import pandas as pd
import misc
import h5py
import numpy as np
from assetpricing import portfolio_construction as poco
from assetpricing import tables_and_figures as taf
from scipy import signal as sig
%matplotlib inline

usr = "hsg-spezial"
path = "c:/users/" + usr + "/google drive/" + \
    "personal/option_implied_betas_project/"

if __name__ == "__main__":

    tau_str = "1m"
    opt_meth = "diff"

    # fetch betas
    hangar = pd.HDFStore(path + \
        "data/estimates/betas_"+tau_str+"_"+opt_meth+".h5", mode='r')

    b_impl = hangar["b_impl_eq"]
    s_d = hangar["s_d"]
    s_m = s_d.resample('M').sum()
    fd_d = hangar["fwd_disc_d"]
    fd_mean = fd_d.resample('M').mean()
    rx_m = hangar["rx_m"]

    hangar.close()

    # filter to use ewm within rolling
    my_filter = lambda x: sig.lfilter([0.5,], [1, -0.1], x[np.isfinite(x)])[-1]

    # daily sorts

    # monthly sorts
    # carry
    ps_carry = poco.get_factor_portfolios(
        poco.rank_sort(rx_m, fd_mean.shift(1), 3), True)
    ps_carry.hml.cumsum().plot()
    taf.descriptives(ps_carry, 12)

    # gap betas
    ps_dol_gap = poco.get_factor_portfolios(
        poco.rank_sort(rx_m, b_gap_m.shift(1), 3), True)
    ps_dol_gap.hml.cumsum().plot()
    taf.descriptives(ps_dol_gap, 12)

    # monthly implied betas: ewm over previous 22 days
    b_impl_m = b_impl.rolling(window=22,
        min_periods=11).apply(my_filter).resample('M').first()
    # or simple stuff
    b_impl_m = b_impl.resample('M').mean()
    ps_dol_impl = poco.get_factor_portfolios(
        poco.rank_sort(rx_m, b_impl_m.shift(1), 3), True)
    taf.descriptives(ps, 12)
    ps_dol_impl.cumsum().plot()

    # monthly series
    betas_m = betas_d.resample('M').first()
    s_m = s_d.resample('M').sum()
    betas_m.plot()

    # monthly sorts
    ps = poco.get_factor_portfolios(poco.rank_sort(s_m, betas_m, 3),
        True)
    taf.descriptives(ps)
    ps.cumsum().plot()
