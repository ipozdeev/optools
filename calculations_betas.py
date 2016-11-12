# calculation of variance
import pandas as pd
import numpy as np
import misc
import os
import h5py
import optools_wrappers as wrap
# %matplotlib inline

usr = "hsg-spezial"
path = "c:/users/" + usr + "/google drive/" + \
    "personal/option_implied_betas_project/"

if __name__ == "__main__":

    tau_str = "1m"
    opt_meth = "diff"

    # load covariances
    hangar = h5py.File(path + \
        "data/estimates/fxpairs_"+tau_str+"_"+opt_meth+"_joint_moments.h5",
        mode='a')
    tau_str = hangar.attrs["maturity"]

    # covmat
    vcv = misc.hdf_to_pandas(hangar["covariances"])

    # estimate betas --------------------------------------------------------
    # weights (approx., from BIS triennial...)
    wght_bis = pd.Series(np.array([7, 31, 13, 2, 5, 5, 22, 2])/100,
        vcv.major_axis)
    wght_bis = wght_bis/wght_bis.sum()

    wght_eq = pd.Series(np.ones(len(vcv.major_axis))/len(vcv.major_axis),
        vcv.major_axis)

    # # store weight as attribute of hangar
    # hangar.attrs["weights"] = wght

    # space for betas
    b_impl_bis = pd.DataFrame(data=np.empty(shape=(vcv.shape[:2])),
        index=vcv.items, columns=vcv.major_axis)
    b_impl_eq = b_impl_bis.copy()
    # loop over dates
    for idx, row in vcv.iteritems():
        # idx = vcv.items[10]
        B_bis = wrap.wrapper_beta_from_covmat(covmat=vcv.loc[idx],
            wght=wght_bis)
        B_eq = wrap.wrapper_beta_from_covmat(covmat=vcv.loc[idx],
            wght=wght_eq)
        b_impl_bis.loc[idx] = B_bis
        b_impl_eq.loc[idx] = B_eq

    # save betas
    misc.pandas_to_hdf(group=hangar, pandas_object=b_impl_bis,
        dset_name="b_impl_bis")
    misc.pandas_to_hdf(group=hangar, pandas_object=b_impl_eq,
        dset_name="b_impl_eq")

    hangar.close()
