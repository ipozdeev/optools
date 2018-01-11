import pandas as pd
import optools.wrappers as op_wrap
from foolbox.data_mgmt import set_credentials as set_cred
path_to_data = set_cred.set_path("option_implied_betas_project/data/raw/",
                                 which="gdrive")

def recipe_mfiv():
    """

    Parameters
    ----------
    deriv
    spot
    rf

    Returns
    -------

    """
    interpolation_kwargs = {
        "in_method": "spline",
        "bc_type": "clamped",
        "ex_method": "const"
    }
    estimation_kwargs = {
        "method": "jiang_tian"
    }

    deriv = pd.read_pickle(path_to_data + "pickles/deriv.p")
    spot = pd.read_pickle(path_to_data + "pickles/spot.p")
    rf = pd.read_pickle(path_to_data + "pickles/ois_bloomi_1w_30y.p")

    for c in deriv.keys():
        xxx, yyy = c[:3].lower(), c[3:].upper()
        for m in ["1m", "3m", "6m", "9m", "1y"]:
            this_data = pd.concat((
                deriv[c][m],
                spot.loc[:, c],
                rf[m].loc[:, xxx].rename("div_yield"),
                rf[m].loc[:, yyy].rename("rf")), axis=1)
            this_data.columns = [p.lower() for p in this_data.columns]

            res = this_data.apply(axis=1, func=op_wrap,
                                  interpolation_kwargs=interpolation_kwargs,
                                  estimation_kwargs=estimation_kwargs)


