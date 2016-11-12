# ---------------------------------------------------------------------------
#
import scipy.signal as sig
import pandas as pd
import h5py
from pandas.tseries.offsets import MonthBegin, MonthEnd
import misc
from ip_econometrics import RegressionModel as regm

usr = "hsg-spezial"
path = "c:/users/" + usr + "/google drive/" + \
    "personal/option_implied_betas_project/"

if __name__ == "__main__":

    tau_str = "1m"
    opt_meth = "diff"

    # load data -------------------------------------------------------------
    hangar = h5py.File(path + \
        "data/estimates/betas_and_returns_"+tau_str+"_"+opt_meth+".h5",
        mode='r')

    b_gap = misc.hdf_to_pandas(hangar["b_gap"])
    b_roll = misc.hdf_to_pandas(hangar["b_roll"])
    b_impl = misc.hdf_to_pandas(hangar["b_impl_eq"])
    b_impl.dropna(inplace=True)

    hangar.close()

    # monthly implied betas: start-of-month
    b_impl_m = b_impl.resample('M').first()

    # monthly implied betas: ewm over previous 22 days
    my_filter = lambda x: sig.lfilter([0.7,], [1, -0.1], x[np.isfinite(x)])[-1]
    b_impl_m = b_impl.rolling(window=22,
        min_periods=11).apply(my_filter).resample('M').first()


    # forecast betas with previous betas ------------------------------------
    # scatter plot
    # align monthly ``gap`` betas and start-of-month implied betas
    x, y = b_impl_m.align(b_gap, axis=0, join='inner')
    fig, ax = plt.subplots(figsize=(7,7/4*3))
    ax.plot(x["aud"], y["aud"],
        color='b', linestyle='none', marker='o')

    # regression


    b_impl.plot()

    s = misc.hdf_to_pandas(hangar["s_d"])
    fd = misc.hdf_to_pandas(hangar["fwd_disc_d"])
    rx = misc.hdf_to_pandas(hangar["rx_d"])
