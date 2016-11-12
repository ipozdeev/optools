# organize_data
# ---------------------------------------------------------------------------
# Collects previously estimated betas, spot returns etc.
import pandas as pd
import h5py
from pandas.tseries.offsets import MonthBegin, MonthEnd
import misc

usr = "hsg-spezial"
path = "c:/users/" + usr + "/google drive/" + \
    "personal/option_implied_betas_project/"

if __name__ == "__main__":

    tau_str = "1m"
    opt_meth = "diff"

    # returns
    rx_m = pd.read_csv(path+"data/raw/returns/rx_m.csv",
        index_col=0, parse_dates=True)
    fwd_disc_d = pd.read_csv(path+"data/raw/returns/fwd_disc_d.csv",
        index_col=0, parse_dates=True)
    s_d = pd.read_csv(path+"data/raw/returns/s_d.csv",
        index_col=0, parse_dates=True)

    # implied betas
    hangar = h5py.File(path + \
        "data/estimates/fxpairs_"+tau_str+"_"+opt_meth+"_joint_moments.h5",
        mode='r')
    b_impl_bis = misc.hdf_to_pandas(hangar["b_impl_bis"])
    b_impl_eq = misc.hdf_to_pandas(hangar["b_impl_eq"])
    hangar.close()

    # actual betas
    hangar = pd.HDFStore(path + \
        "data/estimates/betas_"+tau_str+"_"+opt_meth+".h5", mode='a')

    # # make a smooth calendar index with daily frequency
    # daily_idx = pd.date_range(
    #     start=MonthBegin().rollback(b_impl_eq.first_valid_index()),
    #     end=MonthEnd().rollforward(b_impl_eq.last_valid_index()),
    #     freq='D')

    # # save to a new file
    # hangar = h5py.File(path + \
    #     "data/estimates/betas_and_returns_"+tau_str+"_"+opt_meth+".h5",
    #     mode='w')
    # names = ["rx_m", "fwd_disc_d", "s_d", "b_impl_eq", "b_impl_bis",
    #     "b_gap_m", "b_roll_d"]
    # count = 0
    # for dset in [rx_m, fwd_disc_d, s_d, b_impl_eq, b_impl_bis,
    #     b_gap_m, b_roll_d]:
    #
    #     dset.columns = [c.lower() for c in dset.columns]
    #     dset = dset.loc["2011":,:]
    #     dset = dset.reindex(columns=b_impl_eq.columns)
    #     misc.pandas_to_hdf(
    #         group=hangar,
    #         pandas_object=dset,
    #         dset_name=names[count])
    #     count += 1
    #
    # hangar.close()

    # save to a new file
    names = ["rx_m", "fwd_disc_d", "s_d", "b_impl_eq", "b_impl_bis"]

    count = 0
    for dset in [rx_m, fwd_disc_d, s_d, b_impl_eq, b_impl_bis]:
        dset.columns = [c.lower() for c in dset.columns]
        dset = dset.reindex(columns=b_impl_eq.columns)
        hangar.put(key=names[count], value=dset)
        count += 1

    hangar.close()
