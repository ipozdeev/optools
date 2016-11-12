# calculations from parameters
import pandas as pd
import numpy as np
import os
import misc
import lnmix
import h5py
import warnings
import optools_wrappers as wrap
# %matplotlib inline

# ---------------------------------------------------------------------------
# Results are stored in an HDF file with a separate group for every currency
# pair where three arrays are stored_ quantiles, density and variance. Those
# are easily convertible to pandas objects with help of meta information
# stored in attributes.
# Structure is thus as follows
# hangar/
#   <aaabbb>/       .attrs["index"]
#       quantiles   .attrs["prob"]
#       density     .attrs["domain"]
#       variance
#   <cccddd>/
#       quantiles
#       ...
# ---------------------------------------------------------------------------

usr = "hsg-spezial"
path = "c:/users/" + usr + "/google drive/" + \
    "personal/option_implied_betas_project/"

if __name__ == "__main__":

    # optimization method used
    opt_meth = "slsq"

    # fetch all files with estimated parameters
    all_files = list(filter(lambda x: x.endswith(opt_meth+"_par.csv"),
        os.listdir(path + "data/estimates/par/")))

    # automatically detect maturity (from name of 1st file)
    tau_str = all_files[0][7:9]

    # init storage
    hangar = h5py.File(path + \
        "data/estimates/fxpairs_"+tau_str+"_"+opt_meth+"_statistics.h5",
        mode='w')
    hangar.attrs["maturity"] = tau_str

    # loop over currency pairs
    for filename in all_files:
        # filename = all_files[1]
        # filename = "eurchf_1m_slsq_par.csv"
        # create new group for this pair
        gr = hangar.create_group(filename[:6])

        # collect data from this .xlsx file
        pars = pd.read_csv(path + "data/estimates/par/" + filename,
            index_col=0, parse_dates=True)

        # density -----------------------------------------------------------
        # TODO: think of better domain
        # get average estimates of parameters to set density domain
        avg_par = pars.mean().values
        ln_mix = lnmix.lognormal_mixture(avg_par[:2],avg_par[2:4],avg_par[4:])
        E_x, Var_x = ln_mix.moments()
        domain = np.linspace(E_x-15*np.sqrt(Var_x), E_x+15*np.sqrt(Var_x),
            250)

        # allocate space
        density = pd.DataFrame(
            index=pars.index,
            columns=domain,
            dtype=float)

        # loop over dates
        for idx, par in pars.iterrows():
            # idx = "2016-10-05"
            # par = pars.loc[idx,:]
            p = wrap.wrapper_density_from_par(par.values, domain)
            density.loc[idx] = p

        # save to hdf
        dset = gr.create_dataset("density", data=density.values)
        # save columns as attribute to reconstruct pandas.DataFrame later
        dset.attrs["columns"] = density.columns
        # NB: only byte strings are allowed into h5py, so will need to
        #   decode those later
        dset.attrs["index"] = \
            density.index.map(lambda x: x.strftime("%Y%m%d")).astype('S')

        # quantiles ---------------------------------------------------------
        prob = np.array([0.1, 0.5, 0.9])

        # allocate space
        quantiles = pd.DataFrame(
            index=pars.index,
            columns=prob,
            dtype=float)

        # loop over dates
        for idx, par in pars.iterrows():
            q = wrap.wrapper_quantiles_from_par(par.values, prob=prob)

            # store
            quantiles.loc[idx] = q

        # save to hdf
        dset = gr.create_dataset("quantiles", data=quantiles.values)
        # save columns as attribute to reconstruct pandas.DataFrame later
        dset.attrs["columns"] = quantiles.columns
        # NB: only byte strings are allowed into h5py, so will need to
        #   decode those later
        dset.attrs["index"] = \
            quantiles.index.map(lambda x: x.strftime("%Y%m%d")).astype('S')

        # variance of log(x) ------------------------------------------------
        # allocate space
        variance = pd.Series(
            index=pars.index,
            dtype=float)

        # loop over dates
        for idx, par in pars.iterrows():
            Var_x = wrap.wrapper_variance_of_logx(par.values)

            # store
            variance.loc[idx] = Var_x

        # save to hdf
        dset = gr.create_dataset("variances", data=variance.values)
        # NB: only byte strings are allowed into h5py, so will need to
        #   decode those later
        dset.attrs["index"] = \
            quantiles.index.map(lambda x: x.strftime("%Y%m%d")).astype('S')

    # close HDF
    hangar.close()
