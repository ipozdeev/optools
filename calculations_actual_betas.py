# actual monthly betas
import pandas as pd
import numpy as np
import h5py
import misc
import statsmodels.api as sm
from ip_econometrics import RegressionModel as regm
import matplotlib.pyplot as plt
import datetime as dt
from pandas.tseries.offsets import MonthEnd
# %matplotlib inline

usr = "hsg-spezial"
path = "c:/users/" + usr + "/google drive/" + \
    "personal/option_implied_betas_project/"

tau_str = "1m"
opt_meth = "diff"

# fetch betas
hangar = h5py.File(path + \
    "data/estimates/fxpairs_"+tau_str+"_"+opt_meth+"_joint_moments.h5",
    mode='r')
b_impl_eq = misc.hdf_to_pandas(hangar["b_impl_eq"])

# daily spot
s_d = pd.read_csv(path+"data/raw/returns/s_d.csv",
    index_col=0, parse_dates=True)
s_d.columns = [p.lower() for p in s_d.columns]
s_d = s_d.reindex(columns=b_impl_eq.columns)
s_d.dropna(inplace=True)
hangar.close()

# create dollar index: equally weighted spot
dol_d = s_d.mean(axis=1)
dol_d.name = "dol"

# helper function
def aux_reg(yX):
    """ Give one-factor beta
    """
    # try:
    if isinstance(yX, np.ndarray):
        mod = regm.PureOls(y0=yX[:,0], X0=yX[:,1:])
    else:
        mod = regm.PureOls(y0=yX.ix[:,0], X0=yX.ix[:,1:])

    mod.bleach(z_score=False, add_constant=True, dropna=True)
    res = mod.fit()
    # except:
    #     res = [np.nan,]

    return res[-1]

if __name__ == "__main__":

    # loop over currencies, estimate betas
    for col in s_d:
        yX = pd.concat((s_d[col], dol_d), axis=1)
        # estimate monthly betas on daily data within months ----------------
        # group by month of each year
        this_b_gap_m = yX.groupby([lambda x: x.year, lambda x: x.month])\
            .apply(aux_reg)
        # add name
        this_b_gap_m.name = col

        # estimate rolling betas, 22 days window ----------------------------
        this_b_roll_d = yX.rolling(window=22).cov().loc[:,col,"dol"]/ \
            dol_d.rolling(22).var()
        # add name
        this_b_roll_d.name = col

        # if first instance, create dataframe of betas
        if col == s_d.columns[0]:
            b_gap_m = this_b_gap_m
            b_roll_d = this_b_roll_d
        else:
            b_gap_m = pd.concat((b_gap_m, this_b_gap_m), axis = 1)
            b_roll_d = pd.concat((b_roll_d, this_b_roll_d), axis = 1)

    # change index
    new_idx = \
        [MonthEnd().rollforward(dt.date(p[0], p[1], 1)) for p in b_gap_m.index]
    b_gap_m.index = new_idx

    # # store
    # hangar = h5py.File(path + \
    #     "data/estimates/fxpairs_"+tau_str+"_"+opt_meth+"_joint_moments.h5",
    #     mode='a')
    #
    # misc.pandas_to_hdf(group=hangar, pandas_object=b_gap_m,
    #     dset_name="b_gap_m")
    # misc.pandas_to_hdf(group=hangar, pandas_object=b_roll_d,
    #     dset_name="b_roll_d")
    #
    # hangar.close()

    store = pd.HDFStore(path + \
        "data/estimates/betas_"+tau_str+"_"+opt_meth+".h5", mode='w')
    store.put("b_gap_m", b_gap_m)
    store.put("b_roll_d", b_roll_d)
    store.close()
