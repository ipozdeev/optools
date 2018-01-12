import pandas as pd
import numpy as np
import optools.wrappers as op_wrap
from foolbox.data_mgmt import set_credentials as set_cred
from optools.data_management import organize_data_for_mfiv
from optools.helpers import maturity_str_to_float
from optools.functions import fill_by_no_arb

path_to_data = set_cred.set_path("option_implied_betas_project/data/raw/",
                                 which="gdrive")


def recipe_mfiv(which_mat="1m", which_pair="eurchf", intpl_kwargs=None,
                estim_kwargs=None):
    """

    Parameters
    ----------
    which_mat : str or list
        e.g. '1m' or ['3m', '1y']
    which_pair : str or list
        e.g. ['eurchf', 'usddkk']
    intpl_kwargs : dict
        of keyword args to VolatilitySmile.interpolate
    estim_kwargs : dict
        of keyword args to VolatilitySmile.get_mfiv

    Returns
    -------
    res : pandas.DataFrame or dict thereof
        of mfiv, columned by maturities

    """
    # data ------------------------------------------------------------------
    data = organize_data_for_mfiv(which_pair=which_pair, which_mat=which_mat)

    # CIP relations: rate priority ------------------------------------------
    no_arb_preference = ["chf", "usd", "eur", "gbp", "jpy", "aud", "nzd",
                         "cad", "sek", "nok", "dkk"]
    no_arb_preference = no_arb_preference[::-1]

    # loop over currency pairs ----------------------------------------------
    res = dict()

    for pair, data_pair in data.items():

        # split currencies
        rf_or_div = {pair[:3].lower(): "div_yield", pair[3:].lower(): "rf"}

        # which of these rates should be given lower priority and have its
        #   rf be dermined from CIP
        rate_to_infer = next(c for c in no_arb_preference if c in
                             rf_or_div.keys())

        # loop over maturities ----------------------------------------------
        res_mat = list()

        for mat_str, this_data in data_pair.items():

            # float maturity, in years
            tau = maturity_str_to_float(mat_str)

            # set the rf with lower priority to nan
            this_data.loc[:, rf_or_div[rate_to_infer]] = np.nan

            # loop over time ------------------------------------------------
            this_res = pd.Series(index=this_data.index).rename(mat_str)

            for t, row in this_data.iterrows():

                # if t > pd.to_datetime("2015-01-14"):
                #     print("there")

                # no-arbitrage relationships --------------------------------
                no_arb_dict = dict(
                    row.loc[["spot", "forward", "rf", "div_yield"]])

                no_arb = pd.Series(fill_by_no_arb(tau=tau, **no_arb_dict))

                this_res.loc[t] = \
                    op_wrap.wrapper_mfiv_from_series(row.fillna(no_arb), tau,
                                                     intpl_kwargs,
                                                     estim_kwargs)

            res_mat.append(this_res)

        res[pair] = pd.concat(res_mat, axis=1)

    return res[pair] if len(res) < 2 else res


if __name__ == "__main__":
    # import matplotlib.pyplot as plt
    # import pickle
    #
    # in_kwargs = {
    #     "in_method": "spline",
    #     "ex_method": "constant"
    # }
    #
    # est_kwargs = {
    #     "method": "jiang_tian"
    # }
    #
    # res = dict()
    #
    # for bc_type in ["natural", "clamped"]:
    #     in_kwargs.update({"bc_type": bc_type})
    #
    #     res[bc_type] = recipe_mfiv(which_mat="1m", in_kwargs=in_kwargs,
    #                                est_kwargs=est_kwargs)
    #
    # with open(path_to_data + "../estimates/compare_mfivs.p", mode="wb") as h:
    #     pickle.dump(res, h)

    data = pd.read_pickle(path_to_data + "../estimates/compare_mfivs.p")







