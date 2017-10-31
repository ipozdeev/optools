import pandas as pd
import pickle
from foolbox.data_mgmt import set_credentials as set_cred
from os import listdir

path_to_data = set_cred.set_path(
    "option_implied_betas_project/data/",
    which="gdrive")

def parse_imp_exp_data():
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

        all_data[iso] = this_spr

    return all_data

if __name__ == "__main__":
    all_data = parse_imp_exp_data()
    with open(path_to_data + "raw/pickles/" + "imp_exp.p", mode="wb") as hngr:
        pickle.dump(obj=all_data, file=hngr)
