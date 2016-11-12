# tests
import pandas as pd
import misc
import h5py
import numpy as np
from assetpricing import portfolio_construction as poco
from assetpricing import tables_and_figures as taf
from ip_econometrics import RegressionModel as regm
from scipy import signal as sig
import matplotlib.pyplot as plt
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

    # implied betas ---------------------------------------------------------
    # filter a bit
    b_impl_m = b_impl.resample('M').apply(my_filter)

    # Fama-MacBeth
    b_panel = pd.Panel.from_dict({"mkt" : b_impl_m.shift(1)}, orient="minor")
    lam_imp, alf_imp = taf.fama_macbeth_second(rx_m, b_panel)
    print(taf.fmb_results(lam_imp, alf_imp))

    # realized betas --------------------------------------------------------
    # in-sample
    # Fama-MacBeth 1st
    b_impl_m, _ = taf.fama_macbeth_first(Y=rx_m.loc["2011-01-01":,:],
        X=rx_m.mean(axis=1))
    b_impl_m.loc[:,:,0]
    # Fama-MacBeth 2nd
    lam_rlz, alf_rlz = taf.fama_macbeth_second(rx_m, b_impl_m)
    print(taf.fmb_results(lam_rlz, alf_rlz,
        lag=np.int(0.75*len(alf_rlz)**(1/3))))
    plt.plot(b_impl_m.loc[:,:,0].mean(), rx_m.mean(), 'bo')

    fig, ax = plt.subplots(figsize=(7,7*3/4))
    (12*alf_imp.mean()).plot(ax=ax, color='b', marker='o', linestyle='none')
    (12*alf_rlz.mean()).plot(ax=ax, color='r', marker='o', linestyle='none')
