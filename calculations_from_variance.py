# calculation of diverse measures
import pandas as pd
import numpy as np
import os
import misc
import h5py
import optools_wrappers as wrap
# %matplotlib inline

usr = "hsg-spezial"
path = "c:/users/" + usr + "/google drive/" + \
    "personal/option_implied_betas_project/"

if __name__ == "__main__":

    tau_str = "1m"
    opt_meth = "diff"

    # collect all variances into one dataframe ------------------------------
    # fetch .h5 file with estimated variances (from `calcualtions_from_par...`)
    hangar = h5py.File(path + \
        "data/estimates/fxpairs_"+tau_str+"_"+opt_meth+"_statistics.h5",
        mode='r')
    tau_str = hangar.attrs["maturity"]

    all_pairs = list(hangar.keys())

    # loop over time, collect variances
    for fx_pair in all_pairs:
        # fx_pair = all_pairs[0]
        # group "xxxyyy"
        gr = hangar[fx_pair]
        # group "variance"
        Var_x = misc.hdf_to_pandas(gr["variances"])
        # if exists not, create DataFrame for variances
        if fx_pair == all_pairs[0]:
            variances = pd.DataFrame(index=Var_x.index,
                columns=all_pairs)
        # append
        variances[fx_pair] = Var_x

    hangar.close()

    # estimate covariances and correlations ---------------------------------
    for idx, row in variances.iterrows():
        # row = variances.loc["2012-04-13"]
        # estimate
        covmat, cormat = wrap.wrapper_implied_co_mat(row)
        # if exists not, allocate space
        if idx == variances.index[0]:
            covmat_panel = pd.Panel(
                items=variances.index,
                major_axis=covmat.index,
                minor_axis=covmat.index)
            cormat_panel = pd.Panel(
                items=variances.index,
                major_axis=covmat.index,
                minor_axis=covmat.index)

        # save matrices to Panel
        covmat_panel.loc[idx] = covmat
        cormat_panel.loc[idx] = cormat

    # store covmats and correlations
    hangar = h5py.File(path + \
        "data/estimates/fxpairs_"+tau_str+"_"+opt_meth+"_joint_moments.h5",
        mode='w')
    hangar.attrs["maturity"] = tau_str

    misc.pandas_to_hdf(group=hangar, pandas_object=covmat_panel,
        dset_name="covariances")
    misc.pandas_to_hdf(group=hangar, pandas_object=cormat_panel,
        dset_name="correlations")

    hangar.close()
