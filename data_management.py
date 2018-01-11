import pandas as pd
import pickle
from foolbox.data_mgmt import set_credentials as set_cred
from foolbox.utils import parse_bloomberg_excel
from os import listdir

path_to_data = set_cred.set_path("option_implied_betas_project/data/raw/",
                                 which="gdrive")


def fetch_imp_exp_data():
    """
    """
    # all files + get rid of temp files
    files = listdir(path_to_data + "raw/imp_exp/")
    files = [p for p in files if p[0] != '~']

    # read in one by one
    all_data = dict()

    for f in files:
        iso = f[:3]

        # columns
        colnames = pd.read_excel(path_to_data + "raw/imp_exp/" + f,
            sheetname="iso", index_col=None, header=0)
        colnames = colnames.columns

        # data
        this_spr = pd.read_excel(path_to_data + "raw/imp_exp/" + f,
            sheetname=["imp", "exp"], index_col=0, skiprows=2, header=None)
        for k in this_spr.keys():
            this_spr[k].columns = colnames
            this_spr[k] = this_spr[k].resample('M').last()

        all_data[iso] = this_spr

    return all_data


def fetch_deriv_data():
    """
    """
    # all files + get rid of temp files
    files = [p for p in listdir(path_to_data + "deriv/") if p[0] != '~']

    res = dict()

    for f in files:
        filename = f[:6]

        data = parse_bloomberg_excel(path_to_data + "deriv/" + f,
                                     data_sheets=None,
                                     colnames_sheet="contracts", space=0,
                                     skiprows=7)

        res[filename] = data

    return res


if __name__ == "__main__":

    # all_data = fetch_imp_exp_data()
    #
    # with open(path_to_data + "raw/pickles/" + "imp_exp.p", mode="wb") as hngr:
    #     pickle.dump(obj=all_data, file=hngr)

    data_deriv = fetch_deriv_data()

    with open(path_to_data + "pickles/" + "deriv.p", mode="wb") as hngr:
        pickle.dump(obj=data_deriv, file=hngr)
