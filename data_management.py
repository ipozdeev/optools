import pandas as pd
import pickle
from foolbox.data_mgmt import set_credentials as set_cred
from foolbox.utils import parse_bloomberg_excel
from os import listdir
from optools.functions import fill_by_no_arb
from optools.helpers import maturity_str_to_float

# path to all data files
path_to_data = set_cred.set_path("option_implied_betas_project/data/raw/",
                                 which="gdrive")


def fetch_deriv_data():
    """
    """
    # all files + get rid of temp files
    files = [p for p in listdir(path_to_data + "temp_deriv/")
             if (p[0] != '~') & p.endswith("xlsx")]

    deriv = dict()

    for f in files:
        print(f)
        filename = f[:6]

        # excel files
        data = parse_bloomberg_excel(path_to_data + "temp_deriv/" + f,
                                     data_sheets=None,
                                     colnames_sheet="contracts", space=0,
                                     skiprows=7)

        deriv[filename] = data

    with open(path_to_data + "pickles/deriv.p", mode="wb") as hangar:
        pickle.dump(deriv, hangar)


def fetch_spot():
    """
    """
    # spot
    spot = parse_bloomberg_excel(
        path_to_data + "deriv/spot_pairs_2000_2018_d.xlsx",
        data_sheets="spot",
        colnames_sheet="xxxyyy", space=1,
        skiprows=1)

    with open(path_to_data + "pickles/spot.p", mode="wb") as hangar:
        pickle.dump(spot, hangar)


def fetch_rf():
    """
    """
    # ois -------------------------------------------------------------------
    f_ois = set_cred.set_path("research_data/fx_and_events/") + \
        "ois_bloomi_1w_30y.p"
    ois = pd.read_pickle(f_ois)

    # deposit rates ---------------------------------------------------------
    f_depo = path_to_data + "ir/deposit_rates_2000_2017_d.xlsx"
    depo = parse_bloomberg_excel(f_depo, data_sheets=None,
                                 colnames_sheet="iso", space=1, skiprows=1)
    # depo = pd.concat(depo, axis=1)
    # depo.columns.names = ["maturity", "currency"]
    # depo = depo.swaplevel("maturity", "currency", axis=1)
    # depo = {k: depo[k] for k in depo.columns.levels[0]}

    with open(path_to_data + "pickles/depo.p", mode="wb") as hangar:
        pickle.dump(depo, hangar)

    # libor -----------------------------------------------------------------
    f_libor = path_to_data + "ir/libor_2000_2017_d.xlsx"
    libor = parse_bloomberg_excel(f_libor, data_sheets=None,
                                  colnames_sheet="iso", space=1, skiprows=1)
    # libor = pd.concat(libor, axis=1)
    # libor.columns.names = ["maturity", "currency"]
    # libor = libor.swaplevel("maturity", "currency", axis=1)
    # libor = {k: libor[k] for k in libor.columns.levels[0]}

    with open(path_to_data + "pickles/libor.p", mode="wb") as hangar:
        pickle.dump(libor, hangar)

    # merge, set priority to ois - depo - libor
    merged = dict()
    for k in list(set(
            list(libor.keys()) + list(depo.keys()) + list(ois.keys()))):
        this_ois = ois.get(k, pd.DataFrame({}))
        this_libor = libor.get(k, pd.DataFrame({}))
        this_depo = depo.get(k, pd.DataFrame({}))

        this_df = pd.DataFrame(
            index=this_ois.index.union(this_libor.index.union(
                this_depo.index)),
            columns=this_ois.columns.union(this_libor.columns.union(
                this_depo.columns)))

        merged[k] = this_df.fillna(this_ois)\
            .fillna(this_libor)\
            .fillna(this_depo)

    with open(path_to_data + "pickles/merged_ois_depo_libor.p", mode="wb") \
            as hangar:
        pickle.dump(merged, hangar)

    return


def organize_data_for_mfiv(which_pair=None, which_mat=None,
                           rf_pickle=None):
    """
    Parameters
    ----------
    which_pair : str or list
        e.g. 'eurchf'
    which_mat : str or list
        e.g. '2w' or '1y'
    rf_pickle : str
        pickle name with '.p' extension, e.g. 'ois_bloomi_1w_30y.p'
    """
    # flag_1d_pair = False
    # flag_1d_mat = False

    if rf_pickle is None:
        rf_pickle = "merged_ois_depo_libor.p"

    if which_pair is not None:
        if not isinstance(which_pair, (list, tuple)):
            # flag_1d_pair = True
            which_pair = [which_pair, ]

    if which_mat is not None:
        if not isinstance(which_mat, (list, tuple)):
            # flag_1d_mat = True
            which_mat = [which_mat, ]

    # read in ---------------------------------------------------------------
    deriv = pd.read_pickle(path_to_data + "pickles/deriv.p")
    spot = pd.read_pickle(path_to_data + "pickles/spot.p")
    rf = pd.read_pickle(path_to_data + "pickles/" + rf_pickle)

    # loop over currency pairs
    all_pair = dict()

    for pair in (deriv.keys() if which_pair is None else which_pair):

        # select this pair's data
        v = {key.lower(): value for key, value in deriv[pair].items()}

        # base currency, counter currency
        xxx, yyy = pair[:3], pair[3:]

        # scale: a xxxjpy pip is .001
        scale = 100 if yyy == "jpy" else 10000

        # loop over maturities ----------------------------------------------
        all_mat = dict()

        for mat in (v.keys() if which_mat is None else which_mat):

            if any([mat not in p.keys() for p in [rf, v]]):
                continue

            tau = maturity_str_to_float(mat)

            # select this maturity
            vv = v[mat]

            # spot ----------------------------------------------------------
            this_s = spot.loc[:, pair].copy().rename("spot")

            # forward -------------------------------------------------------
            # from forward points to forward prices
            this_f = (this_s + vv.pop("forward") / scale).rename("forward")

            # risk-free rates -----------------------------------------------
            # are usually in percent, not in (frac of 1)
            this_rf = rf[mat].loc[:, yyy].rename("rf") / 100
            this_div = rf[mat].loc[:, xxx].rename("div_yield") / 100

            # try to fill by no-arb relations -------------------------------
            # concat
            sfrd = pd.concat((this_s, this_f, this_rf, this_div), axis=1)

            # loop over time
            sfrd_no_arb = sfrd.copy()
            for t, row in sfrd.iterrows():
                # print(t)
                sfrd_no_arb.loc[t, :] = \
                    fill_by_no_arb(tau=tau, raise_errors=False, **dict(row))

            # combinations --------------------------------------------------
            # drop rows where atm is missing - no use
            vv = vv.dropna(subset=["atm_vola"])

            # group by delta, delete those (rr, bf) pairs where wither rr or
            #   bf is missing
            group_fun = lambda x: x[:2]

            # these will have at least two values in a row
            these_combies = pd.concat(
                [c.dropna() for _, c in
                 vv.drop("atm_vola", axis=1).groupby(by=group_fun, axis=1)],
                axis=1)

            # ffill risk-free rates
            sfrd_no_arb.loc[:, ["rf", "div_yield"]] = \
                sfrd_no_arb.loc[:, ["rf", "div_yield"]].ffill(limit=1)

            # combine everything, drop na beforehand, convert vola to
            #   (frac of 1)
            this_res = pd.concat(
                [these_combies / 100,
                 vv.loc[:, "atm_vola"].dropna() / 100,
                 sfrd_no_arb.dropna()], axis=1, join="inner")

            all_mat[mat] = this_res

        # if flag_1d_mat:
        #     all_mat = all_mat[which_mat[0]]

        all_pair[pair] = all_mat

    # if flag_1d_pair:
    #     all_pair = all_pair[which_pair[0]]

    return all_pair


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


if __name__ == "__main__":

    # all_data = fetch_imp_exp_data()
    #
    # with open(path_to_data + "raw/pickles/" + "imp_exp.p", mode="wb") as hngr:
    #     pickle.dump(obj=all_data, file=hngr)

    # fetch_rf()

    data = pd.read_pickle(path_to_data + "pickles/merged_ois_depo_libor.p")
    data.keys()


    # fetch_spot()

    # ir_0 = pd.read_pickle(path_to_data + "pickles/" + "ir_bloomi.p")
    # ir_1 = pd.read_pickle(set_cred.set_path("research_data/fx_and_events/") +
    #                       "ois_bloomi_1w_30y.p")
    # ir_0 = ir_0["1m"]
    # ir_1 = ir_1["1m"]
    #
    # cur = "chf"
    # both = pd.concat(
    #     (ir_0.loc[:, cur].rename("depo"), ir_1.loc[:, cur].rename("ois")),
    #     axis=1)
    # both.plot()