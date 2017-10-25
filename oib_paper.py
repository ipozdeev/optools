import pandas as pd
import numpy as np
from os import listdir
import pickle

from foolbox.data_mgmt import set_credentials as set_cred

from optools import optools_func as op
from optools import optools_wrappers as wrap
from optools.import_data import *

class OIBPaper():
    """
    """
    def __init__(self, tau_str="1m", ccur="usd", x_curs=[]):
        """
        """
        self.tau_str = tau_str
        self.tau = float(tau_str[:-1])/12.0
        self.ccur = ccur
        self.x_curs = x_curs

        self.path_main = set_cred.gdrive_path("option_implied_betas_project/")
        self.path_to_fx = set_cred.gdrive_path("research_data/fx_and_events/")
        self.path_to_raw = self.path_main + "data/raw/"
        self.path_to_mfiv = self.path_main + "data/estimates/"

        # construct hangar name
        self.hangar_name = self.path_to_mfiv + '_'.join(
            ("data_vs", ccur, tau_str, "x"+ str(len(x_curs)) + "curs.h5"))

    @property
    def mfiv(self):
        if "mfiv_" + self.tau_str + ".h5" in listdir(self.path_to_mfiv):
            with pd.HDFStore(self.path_to_mfiv + "mfiv_" + self.tau_str +
                ".h5", mode='r') as hngr:
                res = hngr["mfiv"]
        else:
            res = self.calculate_mfiv()

        return res

    @property
    def mficov(self):
        try:
            res = self.from_hdf("mficov")
        except (OSError, KeyError) as e:
            res = self.calculate_mficov()
        return res

    def calculate_mfiv(self):
        """
        """
        path_to_options = self.path_to_raw + "options/"

        # interest rates
        with open(self.path_to_raw + "ir/ir_bloomi.p", mode="rb") as hngr:
            ir = pickle.load(hngr)

        ir = ir[self.tau_str]

        # fetch all files with raw data
        files = list(filter(
            lambda x: x.endswith("deriv.xlsx"), listdir(path_to_options)))

        mfiv = pd.DataFrame()

        for filename in files:
            # collect data from .xlsx file
            # filename = files[17]
            # filename = 'usdjpy_fx_deriv.xlsx'
            print(filename)

            data_for_est = import_data(
                data_path=path_to_options,
                filename=filename,
                tau_str=self.tau_str,
                ir_name=ir)

            # e.g. for Denmark
            if data_for_est.empty:
                continue

            this_mfiv = pd.Series(index=data_for_est.index)

            for idx, row in data_for_est.iterrows():
                # idx, row = list(data_for_est.iterrows())[-528]
                # fetch wings
                deltas, ivs = op.get_wings(
                    row["rr25d"],
                    row["rr10d"],
                    row["bf25d"],
                    row["bf10d"],
                    row["atm"],
                    row["y"],
                    self.tau)

                # to strikes
                K = op.strike_from_delta(
                    deltas,
                    row["s"],
                    row["rf"],
                    row["y"],
                    self.tau,
                    ivs,
                    True)

                # concat to pandas object
                vol_surf = pd.DataFrame(
                    data=np.vstack((K, ivs)).T,
                    columns=["K", "iv"])

                # transform & integrate
                res = op.mfiv_wrapper(
                    vol_surf, row["f"], row["rf"], self.tau, "spline")
                this_mfiv[idx] = res

            mfiv[filename[:6]] = this_mfiv

        with pd.HDFStore(self.path_to_mfiv + "mfiv_" + self.tau_str +".h5",
            mode='w') as hngr:
            hngr.put("mfiv", mfiv)

        return mfiv

    def from_hdf(self, what, s_dt=None, e_dt=None):
        """
        """
        with pd.HDFStore(self.hangar_name, mode='r') as hngr:
                res = hngr[what]

        if (s_dt is None) and (e_dt is None):
            return res

        if s_dt is None:
            s_dt = res.index[0]
        if e_dt is None:
            e_dt = res.index[-1]

        return res.loc[s_dt:e_dt]

    def to_hdf(self, what_dict):
        """
        what_dict : dict
            {key : value}
        """
        with pd.HDFStore(self.hangar_name, mode='a') as hngr:
            for k, v in what_dict.items():
                hngr.put(k, v)

    def calculate_mficov(self):
        """
        """
        # x_curs = ["dkk","nok","sek"]
        # collect all variances into one dataframe --------------------------
        variances = self.mfiv

        # exclude some ------------------------------------------------------
        filter_fun = lambda x: all([p not in x for p in self.x_curs])
        all_pairs = list(filter(filter_fun, variances.columns))
        variances = variances[all_pairs]

        # estimate covariances ----------------------------------------------
        unq_curs = []
        for p in variances.columns:
            unq_curs += [p[:3], p[3:]]
        unq_curs = [c for c in list(set(unq_curs)) if c != "usd"]

        covmat_panel = pd.Panel(
            items=variances.index,
            major_axis=unq_curs,
            minor_axis=unq_curs)

        for idx, row in variances.iterrows():
            # row = variances.loc["2012-04-13"]
            covmat, _ = wrap.wrapper_implied_co_mat(row, self.ccur)

            # save matrices to Panel
            covmat_panel.loc[idx] = covmat

        # drop na
        covmat_panel = covmat_panel.dropna(axis="items", how="all")

        # store covmats and correlations
        self.to_hdf({"mficov": covmat_panel})

        with pd.HDFStore(self.hangar_name, mode='a') as hngr:
            hngr.root.mficov._v_attrs.x_curs = self.x_curs

        return covmat_panel

    def calculate_mfibetas(self, wght=None, trim_vcv=False, exclude_self=False,
        mnemonic=None):
        """
        """
        # fetch covariances
        vcv = self.mficov

        # record names, for later use (this will be an Index)
        curs = vcv.minor_axis

        if wght is None:
            if mnemonic is not None:
                raise ValueError("For undefined weights, equally-weighted " +
                    "betas are calculated and stored under 'eq'.")
            else:
                mnemonic = "eq"
            wght = pd.DataFrame(1.0, index=vcv.items, columns=curs)
        else:
            if mnemonic is None:
                raise ValueError("Provide mnemonic for identification of " +
                    "the calculated series in the HDF.")
            if isinstance(wght, pd.Series):
                wght = pd.DataFrame(
                    data=np.array([wght.values, ]*len(vcv.items)),
                    index=vcv.items,
                    columns=curs)

        # estimate betas ----------------------------------------------------
        B = pd.DataFrame(index=vcv.items, columns=curs)
        vix = pd.Series(index=vcv.items)

        # loop over dates
        for idx, row in vcv.iteritems():
            # idx = vcv.items[10]

            # skip this matrix if any NA is present
            this_vcv = row.copy()
            this_wght = wght.loc[idx].copy()

            if this_vcv.isnull().any().any():
                continue

            # trim if necessary (+reweight!)
            if trim_vcv:
                this_vcv = self.trim_covmat(this_vcv)
                this_wght = wrap.normalize_weights(this_wght)

            this_b, this_vix = wrap.wrapper_beta_from_covmat(
                covmat=this_vcv,
                wght=this_wght,
                exclude_self=exclude_self)

            # save
            B.loc[idx, :] = this_b
            vix.loc[idx] = this_vix

        # rename
        B = B.loc[:, sorted(B.columns)]

        # store
        self.to_hdf({
            ("mfibetas/" + mnemonic): B,
            ("vix/" + mnemonic): vix,
            ("wght/" + mnemonic): wght})

        return B

    @staticmethod
    def trim_covmat(covmat):
        """
        """
        covmat_trm = covmat.copy()

        # init count of nans
        nan_count_total = pd.isnull(covmat_trm).sum().sum()

        # while there are nans in covmat, remove columns with max no. of nans
        while nan_count_total > 0:
            # detect rows where number of nans is less than maximum
            nan_max = pd.isnull(covmat_trm).sum()
            nan_max_idx = max([(p,q) for q,p in enumerate(nan_max)])[1]

            covmat_trm = covmat_trm.drop(covmat_trm.columns[nan_max_idx],
                axis=0)
            covmat_trm = covmat_trm.drop(covmat_trm.columns[nan_max_idx],
                axis=1)

            # new count of nans
            nan_count_total = pd.isnull(covmat_trm).sum().sum()

        return covmat_trm

    @staticmethod
    def smooth_to_monthly(x, wght):
        """
        """
        my_filter = lambda x: x.ewm(alpha=wght).mean()

        return x.resample("M").apply(my_filter).resample("M").last()

if __name__ == "__main__":

    import ipdb
    oibp = OIBPaper(tau_str="1m", ccur="usd", x_curs=["sek", "nok", "dkk"])
    # ipdb.set_trace()
    mfiv = oibp.mfiv
    mficov = oibp.mficov
    mficov = mficov.drop("usd", axis="minor_axis")\
        .drop("usd", axis="major_axis")
    oibp.to_hdf({"mficov": mficov})
    mficov.isnull().sum(axis="items")
    B = oibp.calculate_mfibetas(trim_vcv=True)
