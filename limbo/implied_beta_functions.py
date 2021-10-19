import pandas as pd
import numpy as np
import os
import datetime
from pandas.tseries.offsets import MonthEnd, QuarterEnd, DateOffset
import statsmodels.api as sm
import pickle

from optools import pricing as op
from optools import wrappers as wrap
from optools.import_data import *

from foolbox import portfolio_construction as poco, RegressionModel as regm
from foolbox.finance import into_currency
from foolbox.data_mgmt import set_credentials as setc

# import ipdb

class ImpliedBetaEnvironment():
    """
    """
    def __init__(self, tau_str, opt_meth, path_to_data, path_to_raw=None,
        ccur="usd", exclude_cur=[]):
        """
        """
        self.tau_str = tau_str
        self.tau = float(tau_str[:-1])/12.0
        self.path_to_raw = path_to_raw
        self.path_to_data = path_to_data
        self.path_to_spot = setc.gdrive_path("research_data/fx_and_events/")
        self.opt_meth = opt_meth
        self.ccur = ccur
        self.exclude_cur = exclude_cur

        # construct filename
        self.storage_name = path_to_data+\
            "data_"+"vs_"+ccur+"_"+tau_str+"_"+opt_meth+\
            str(10-len(exclude_cur))+"_curs"+".h5"

        with pd.HDFStore(path_to_data+"mfiv_1m.h5", mode='r') as hangar:
            self.mfiv = hangar["variances"]

    def _fetch_from_hdf(self, what, s_dt=None, e_dt=None):
        """
        """
        with pd.HDFStore(self.storage_name, mode='r') as hangar:
                res = hangar[what]

        if (s_dt is None) and (e_dt is None):
            return res
        else:
            if s_dt is None:
                s_dt = res.index[0]
            if e_dt is None:
                e_dt = res.index[-1]
            return res.loc[s_dt:e_dt]

    def _store_to_hdf(self, what_dict):
        """
        what_dict : dict
            {key : value}
        """
        with pd.HDFStore(self.storage_name, mode='a') as hangar:
            for k, v in what_dict.items():
                hangar.put(k, v)

        return

    def _fetch_raw(self, pickle_name="data_wmr_dev_m.p", what="rx",
        s_dt=None, e_dt=None, cols=None):
        """
        """
        with open(self.path_to_spot+pickle_name, mode='rb') as fname:
            data = pickle.load(fname)

        data = data[what]

        if s_dt is None:
            s_dt = data.index[0]
        if e_dt is None:
            e_dt = data.index[-1]
        if cols is None:
            cols = data.columns

        return data.loc[s_dt:e_dt,cols]

    def get_mfiv(self):
        """
        """
        # interest rates
        ir_name = import_rf_bloomi(
            self.path_to_raw+"ir_bloomi.xlsx", self.tau_str)

        # fetch all files with raw data
        files = list(filter(lambda x: x.endswith("deriv.xlsx"),
            os.listdir(self.path_to_raw)))

        mfiv = pd.DataFrame()

        for filename in files:
            # collect data from .xlsx file
            # filename = files[17]
            # filename = 'usdjpy_fx_deriv.xlsx'
            data_for_est = import_data(
                data_path=self.path_to_raw,
                filename=filename,
                tau_str=self.tau_str,
                ir_name=ir_name)

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
                    data=np.vstack((K,ivs)).T,
                    columns=["K","iv"])

                # transform & integrate
                res = op.mfiv_wrapper(
                    vol_surf,row["f"],row["rf"],self.tau,"spline")
                this_mfiv[idx] = res

            mfiv[filename[:6]] = this_mfiv

        self._store_to_hdf({"variances": mfiv})

        with pd.HDFStore(path_out+"mfiv_"+self.tau_str+".h5", mode='w') \
            as hangar:
                hangar.put("variances", mfiv)

    def get_mfis(self):
        """
        """
        # interest rates
        ir_name = import_rf_bloomi(
            self.path_to_raw+"ir_bloomi.xlsx", self.tau_str)

        # fetch all files with raw data
        files = list(filter(lambda x: x.endswith("deriv.xlsx"),
            os.listdir(self.path_to_raw)))

        mfis = pd.DataFrame()

        for filename in files:
            # ipdb.set_trace()
            # collect data from .xlsx file
            # filename = files[17]
            # filename = 'usdjpy_fx_deriv.xlsx'
            data_for_est = import_data(
                data_path=self.path_to_raw,
                filename=filename,
                tau_str=self.tau_str,
                ir_name=ir_name)

            if data_for_est.empty:
                continue

            this_mfis = pd.Series(index=data_for_est.index)
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
                    data=np.vstack((K,ivs)).T,
                    columns=["K","iv"])

                # transform & integrate
                res = op.mfiskew_wrapper(
                    vol_surf, row["f"], row["rf"], self.tau, row["s"],
                    method="spline")

                this_mfis[idx] = res

            mfis[filename[:6]] = this_mfis

        self._store_to_hdf({"skewness": mfis})

        with pd.HDFStore(self.path_to_data+"mfis_"+self.tau_str+".h5", mode='a') \
            as hangar:
                hangar.put("skewness", mfis)

    def get_covariances(self):
        """
        """
        # exclude_cur = ["dkk","nok","sek"]
        # collect all variances into one dataframe --------------------------
        variances = self.mfiv

        # exclude some ------------------------------------------------------
        filter_fun = lambda x: all([p not in x for p in self.exclude_cur])
        all_pairs = list(filter(filter_fun, variances.columns))
        variances = variances[all_pairs]

        # # fill gaps
        # # TODO: rewrite once Regression model is improved
        # for c in variances.columns:
        #     b, _ = regm.light_ols(
        #         variances[c],
        #         variances.drop(c, axis=1).mean(axis=1),
        #         add_constant=True, ts=False)
        #     y_hat = sm.add_constant(variances.drop(c, axis=1).mean(axis=1)).dot(b)
        #     y_hat.name = c
        #     variances.loc[:,c] = variances[c].fillna(y_hat)
        variances = variances.interpolate("nearest", axis=0, limit=1)

        # estimate covariances ----------------------------------------------
        for idx, row in variances.iterrows():
            # row = variances.loc["2012-04-13"]
            covmat, _ = wrap.wrapper_implied_co_mat(row, self.ccur)
            # if exists not, allocate space
            if idx == variances.index[0]:
                covmat_panel = pd.Panel(
                    items=variances.index,
                    major_axis=covmat.index,
                    minor_axis=covmat.index)

            # save matrices to Panel
            covmat_panel.loc[idx] = covmat

        # drop na
        covmat_panel = covmat_panel.dropna(axis="items", how="all")

        # store covmats and correlations
        self._store_to_hdf({"covariances": covmat_panel})

        hangar = pd.HDFStore(self.storage_name, mode='a')
        hangar.root.covariances._v_attrs.excluded_cur = self.exclude_cur
        hangar.close()

    def get_implied_betas(self, wght_bis=None, exclude_self=False):
        """
        """
        # fetch covariances
        vcv = self._fetch_from_hdf("covariances")

        # record names, for later use (this will be an Index)
        cur_names = vcv.minor_axis

        # trade-based weights (approx., from BIS triennial...) --------------
        if wght_bis is None:
            # ['aud','eur','gbp','nzd','cad','chf','dkk','jpy','nok','sek']
            # wght_bis = pd.Series(np.array([7, 31, 13, 2, 5, 5, 1, 22, 2, 1])/100,
            #     ['aud','eur','gbp','nzd','cad','chf','dkk','jpy','nok','sek'])
            wght_bis = pd.Series(np.array([
                129, 289, 55, 30, (123+43+39+34+13+11+19+16+4.6),
                4.8, 9.9, 10, 4, 7.9
                ]),
                ['jpy','cad','gbp','chf','eur','nok','sek','aud','nzd','dkk'])

            wght_bis = wght_bis[cur_names]

            wght_bis = pd.DataFrame(
                data=np.tile(wght_bis.values[np.newaxis,:], (vcv.shape[0],1)),
                index=vcv.items,
                columns=cur_names)

        wght_bis = wght_bis[cur_names]

        # make sure weights sum up to 1
        wght_bis = wght_bis.divide(np.abs(wght_bis.sum(axis=1)), axis=0)

        # equal weights are... equal! ---------------------------------------
        wght_eq = pd.Series(np.ones(len(cur_names)), index=cur_names)

        # estimate betas ----------------------------------------------------
        # space for betas
        b_impl_bis = pd.DataFrame(data=np.empty(shape=(vcv.shape[:2])),
            index=vcv.items, columns=cur_names)*np.nan
        b_impl_eq = b_impl_bis.copy()

        dol_s2 = pd.DataFrame(index=vcv.items, columns=["bis","eq"])

        # loop over dates
        for idx, row in vcv.iteritems():
            # idx = vcv.items[10]
            # skip this matrix if any NA is present
            if row.isnull().any().any():
                continue

            b_bis, dol_s2_bis = \
                wrap.wrapper_beta_from_covmat(covmat=vcv.loc[idx],
                wght=wght_bis.loc[idx,:], exclude_self=exclude_self)
            b_eq, dol_s2_eq =\
                wrap.wrapper_beta_from_covmat(covmat=vcv.loc[idx],
                wght=wght_eq, exclude_self=exclude_self)
            # ipdb.set_trace()

            # save
            b_impl_bis.loc[idx] = b_bis
            b_impl_eq.loc[idx] = b_eq

            dol_s2.loc[idx, "bis"] = dol_s2_bis
            dol_s2.loc[idx, "eq"] = dol_s2_eq

        # rename
        b_impl_bis = b_impl_bis[sorted(b_impl_bis.columns)]
        b_impl_eq = b_impl_eq[sorted(b_impl_eq.columns)]

        # store
        self._store_to_hdf({
            "eq/b_impl": b_impl_eq.loc["2008-07":],
            "bis/b_impl": b_impl_bis.loc["2008-07":],
            "eq/dol_mfiv": dol_s2["eq"],
            "bis/dol_mfiv": dol_s2["bis"]})

        # store weights
        hangar = pd.HDFStore(self.storage_name, mode='a')
        # hangar.root.bis._v_attrs.weights = wght_bis
        hangar.close()

        self.b_impl_eq = b_impl_eq

    @staticmethod
    def get_ib(vcv, wght=None):
        """
        """
        # record names, for later use (this will be an Index)
        cur_names = vcv.minor_axis

        # equal weights are... equal! ---------------------------------------
        if wght is None:
            wght = vcv.iloc[:,:,0].copy()
            wght /= wght

        # estimate betas ----------------------------------------------------
        # space for betas
        b_impl_usr = wght.copy()*np.nan

        # loop over dates
        for idx, row in wght.iterrows():
            # idx = vcv.items[10]
            # skip this matrix if any NA is present
            if vcv.loc[idx].isnull().any().any():
                continue

            b, _ =\
                wrap.wrapper_beta_from_covmat(covmat=vcv.loc[idx],
                    wght=row,
                    zero_cost=True) # TODO: this is a quick fix

            # save
            b_impl_usr.loc[idx] = b

        b_impl_usr = b_impl_usr[sorted(b_impl_usr.columns)]

        return b_impl_usr


    @staticmethod
    def b_realized_later(y, x, idx, d):
        """ Calculate beta realized `d` days later.
        """
        res = pd.Series(index=idx)*np.nan
        for t in idx:
            dt = t + DateOffset(days=d)
            if dt > (y.index[-1] - DateOffset(days=d-1)):
                break
            # subsample y and x
            this_y = y.loc[(t+DateOffset(days=1)):dt]
            this_x = x.loc[(t+DateOffset(days=1)):dt]
            res.loc[t] = regm.light_ols(this_y, this_x, True)[0][1]

        return res

    def get_actual_betas(self, s_d, s_m):
        """
        """
        # number of days for this tau_str -----------------------------------
        tau_days = int(self.tau_str[:-1])*30

        # data --------------------------------------------------------------
        # we need dollar factor be constructed perfectly in line with what
        #   currencies have implied betas

        # # valid columns are currencies with implied betas
        b_impl_eq = self._fetch_from_hdf("eq/b_impl")
        # cur_names = b_impl_eq.columns
        time_idx = b_impl_eq.index

        # drop some currencies
        s_d = s_d.drop(self.exclude_cur, axis=1)
        s_m = s_m.drop(self.exclude_cur, axis=1)
        cur_names = s_d.columns

        # # weights
        # wght_bis = pd.Series(np.array([7, 31, 13, 2, 5, 5, 1, 22, 2, 1])/100,
        #     ['aud','eur','gbp','nzd','cad','chf','dkk','jpy','nok','sek'])
        # wght_bis = wght_bis[cur_names]
        # # make sure weights sum up to 1
        # wght_bis = wght_bis/wght_bis.sum()

        wght_bis = self._fetch_from_hdf("wght_bis")
        wght_bis = wght_bis.divide(wght_bis.sum(axis=1), axis=0)

        # dollar factor -----------------------------------------------------
        dol_d = pd.DataFrame(columns=["bis", "eq"])
        # daily
        dol_d["eq"] = s_d.mean(axis=1)
        # dol_d["bis"] = s_d.dot(wght_bis)
        dol_d["bis"] = s_d.multiply(
            wght_bis.reindex(index=s_d.index, method="ffill"), axis=0)\
                .sum(axis=1).replace(0.0, np.nan)

        # monthly
        dol_m = pd.DataFrame(columns=["bis", "eq"])
        dol_m["eq"] = s_m.mean(axis=1)

        # dol_m["bis"] = s_m.dot(wght_bis)
        dol_m["bis"] = s_m.multiply(
            wght_bis.loc[s_m.index,:], axis=0).sum(axis=1).replace(
                0.0,np.nan)

        # loop over equally-weighted/bis-weighted
        for w in ["eq", "bis"]:
            # w = "eq"
            # gap beta, one per period (month, quarter, 3 quarters, etc.)
            b_gap = pd.DataFrame(columns=cur_names)
            # rolling beta, estimated on monthly data with window=48 months
            b_roll_m = pd.DataFrame(columns=cur_names)
            # rolling beta, estimated on daily data with window=N days
            b_roll = pd.DataFrame(columns=cur_names)
            # expanding beta
            b_exp_m = pd.DataFrame(columns=cur_names)
            # beta realized N days later
            b_real = pd.DataFrame(columns=cur_names)

            # loop over currencies
            for col in cur_names:
                # col = "aud"
                # monthly ---------------------------------------------------
                # response is currency returns
                y = s_m[col]
                # regressor is dollar factor, this type-weighted
                x = dol_m[w]

                # rolling, estimated on monthly returns
                _, b = regm.DynamicOLS("rolling", y, x, window=48).fit()
                b_roll_m[col] = b

                # expanding, estimated on monthly returns
                _, b = regm.DynamicOLS("expanding", y, x, min_periods=48).fit()
                b_exp_m[col] = b

                # daily -----------------------------------------------------
                # realized N days later
                y = s_d[col]
                x = dol_d[w]
                b_real[col] = self.b_realized_later(
                    y, x, time_idx, tau_days)

                # rolling, N-day window
                _, b = regm.DynamicOLS("rolling", y, x, window=tau_days).fit()
                b_roll[col] = b

                # gap -------------------------------------------------------
                if tau_days == 30:
                    _, b = regm.DynamicOLS("grouped_by", y, x,
                        by=[lambda x: x.year, lambda x: x.month]).fit()
                elif tau_days == 90:
                    _, b = regm.DynamicOLS("grouped_by", y, x,
                        by=[lambda x: x.year, lambda x: x.quarter]).fit()
                else:
                    b = pd.Series(index=time_idx)*np.nan

                b_gap[col] = b

            # now index of b_gap_m is weird -> change to 31st
            if tau_days == 90:
                new_idx = [QuarterEnd().rollforward(
                    datetime.date(p[0], p[-1]*3, 1)) for p in b_gap.index]
            elif tau_days == 30:
                new_idx = [MonthEnd().rollforward(
                    datetime.date(p[0], p[-1], 1)) for p in b_gap.index]
            else:
                new_idx = time_idx

            b_gap.index = new_idx

            # store
            dict_to_store = {
                w+"/b_gap": b_gap,
                w+"/b_roll_m": b_roll_m,
                w+"/b_exp_m": b_exp_m,
                w+"/b_real": b_real,
                w+"/b_roll": b_roll,
                w+"/dol_m": dol_m[w],
                w+"/dol_d": dol_d[w]}

            self._store_to_hdf(dict_to_store)

    @staticmethod
    def get_hf_daily_betas(s_hf, wght_m):
        """
        """
        dol_hf = s_hf.dot(wght_m)
        b_hf_d = pd.DataFrame(columns=s_hf.columns)
        for cur in s_hf.columns:
            _, b = regm.DynamicOLS("grouped_by", y0=s_hf[cur], x0=dol_hf,
                by=[lambda x: x.month,lambda x: x.day]).fit()
            b_hf_d[cur] = b

        new_idx = [datetime.date(2015, p[0], p[1]) for p in b_hf_d.index]
        b_hf_d.index = new_idx

        return b_hf_d

    def get_portfolio_betas(self, pfs, wght_m):
        """
        pfs: dict
            output of poco.rank_sort()
        """
        # pfs=strat
        # vcv = BImpl._fetch_from_hdf("covariances")
        # fetch covariances
        vcv = self._fetch_from_hdf("covariances")
        # vcv = vcv.resample('M', axis='items').last().shift(1, axis="items")

        # fetch portfolios from which one can obtain weights
        keys = sorted([p for p in pfs.keys() if "portfolio" in p])

        # space for betas: columns for portfolios
        B = pd.DataFrame(
            columns=["p"+p[-1] for p in keys],
            index=vcv.items)

        # # delete ----
        # pf = (pfs["portfolio3"]/pfs["portfolio3"]).divide(
        #     pfs["portfolio3"].notnull().sum(axis=1), axis=0).fillna(
        #         -1*(pfs["portfolio1"]/pfs["portfolio1"]).divide(
        #             pfs["portfolio3"].notnull().sum(axis=1), axis=0))
        # pf = pf.fillna(0)
        # # -----------
        # B = list()

        # loop over portfolios
        for k in keys:
            # k = "portfolio1"
            # all portfolios are equally-weighted
            pf = pfs[k].notnull().divide(pfs[k].notnull().sum(axis=1), axis=0)
            pf = pf.reindex(vcv.items, method="bfill")
            # loop over time
            for t in pf.index:
                # t = pf.index[100]
                try:
                    idx = vcv.items.get_loc(t, "bfill")
                    this_vcv = vcv.iloc[idx,:,:]
                    this_b = wrap.wrapper_beta_of_portfolio(
                        covmat=this_vcv,
                        wght_p=pf.loc[t,:],
                        wght_m=wght_m)
                except:
                    this_b = np.nan

                # B += [this_b,]
                B.loc[t,k[0]+k[-1]] = this_b
                # b_impl_carry = pd.Series(index=pf.index, data=B)

        return B

    def get_portfolio_betas_s(self, b_impl, pfs):
        """
        pfs: dict
            output of poco.rank_sort()
        """
        # pfs=strat
        # vcv = BImpl._fetch_from_hdf("covariances")
        # fetch covariances
        # vcv = vcv.resample('M', axis='items').last().shift(1, axis="items")

        # fetch portfolios from which one can obtain weights
        keys = sorted([p for p in pfs.keys() if "portfolio" in p])

        # space for betas: columns for portfolios
        B = pd.DataFrame(
            columns=["p"+p[-1] for p in keys],
            index=b_impl.index)

        # # delete ----
        # pf = (pfs["portfolio3"]/pfs["portfolio3"]).divide(
        #     pfs["portfolio3"].notnull().sum(axis=1), axis=0).fillna(
        #         -1*(pfs["portfolio1"]/pfs["portfolio1"]).divide(
        #             pfs["portfolio3"].notnull().sum(axis=1), axis=0))
        # pf = pf.fillna(0)
        # # -----------
        # B = list()

        # loop over portfolios
        for k in keys:
            # k = "portfolio1"
            # all portfolios are equally-weighted
            pf = pfs[k].notnull().divide(pfs[k].notnull().sum(axis=1), axis=0)
            pf = pf.reindex(b_impl.index, method="bfill")
            # loop over time
            this_b = (b_impl.where(pf.notnull()) * pf).sum(axis=1)

            B.loc[:,k[0]+k[-1]] = this_b

        return B

    def get_fx_strategies(self, pickle_name, exclude_cur=[], n_portf=3,
        wght_for_flb=0.5):
        """
        """
        with open(self.path_to_spot+pickle_name, mode='rb') as fname:
            data = pickle.load(fname)

        # to lowercase
        for k, _ in data.items():
            data[k].columns = [p.lower() for p in data[k].columns]

        # # clean up, align
        # S_d = 1/data["spot_mid"].drop(exclude_cur, axis=1).dropna(how="all")
        # Fm_d = 1/data["fwd_mid"].drop(exclude_cur, axis=1).dropna(how="all")
        # S_d, Fm_d = S_d.align(Fm_d, axis=0, join="inner")

        # # many days of escess returns (not only end-of-month)
        # rxm_d = S_d*np.nan
        # for t in S_d.index[22:]:
        #     # t = S_d.index[1000]
        #     t_plus_dt = t - DateOffset(months=1)
        #     t_plus_int = Fm_d.index.get_loc(t_plus_dt, "bfill")
        #     t_plus_loc = Fm_d.index[t_plus_int]
        #     rxm_d.loc[t,:] = \
        #         np.log(Fm_d.loc[t_plus_loc,:]/S_d.loc[t,:])

        # clean up, align
        s_d = data["spot_ret"].drop(exclude_cur, axis=1).dropna(how="all")
        fdisc_d = data["fwd_disc"].drop(exclude_cur, axis=1).dropna(how="all")
        s_d, fdisc_d = s_d.align(fdisc_d, axis=0, join="inner")

        # monthly spot
        s_m = s_d.resample('M').sum()

        # monthly excess
        rx_m = s_m + fdisc_d.resample('M').last().shift(1)

        # # to Panel
        # rxm_d_reixed = rxm_d.reindex(index=pd.date_range(
        #     rxm_d.index[0],rxm_d.index[-1],freq='D'), method="ffill")
        # fx_dict = dict()
        # for p in range(5,26):
        #     # p = 5
        #     fx_dict[p] = rxm_d_reixed.ix[rxm_d_reixed.index.day == p,:]
        #
        # rxm_panel_by_day = pd.Panel.from_dict(fx_dict, orient="minor")
        #
        # # currencies
        # these_cur = self._fetch_from_hdf("eq/b_impl").columns
        # this_idx = self._fetch_from_hdf("eq/b_impl").index

        # strat
        carry = poco.rank_sort(
            returns=rx_m,
            signals=fdisc_d.resample('M').mean().shift(1),
            n_portfolios=n_portf)

        pf_carry = poco.get_factor_portfolios(carry, hml=False)

        mom = poco.rank_sort(
            returns=rx_m,
            signals=s_m.rolling(6).mean().shift(7),
            n_portfolios=n_portf)

        pf_mom = poco.get_factor_portfolios(mom, hml=False)

        b_impl_m_eq = \
            self.smooth_to_monthly(
                self._fetch_from_hdf("eq/b_impl", s_dt="2008-07"),
                wght_for_flb)
        b_impl_m_bis = \
            self.smooth_to_monthly(
                self._fetch_from_hdf("bis/b_impl", s_dt="2008-07"),
                wght_for_flb)

        flb = poco.rank_sort(
            returns=rx_m.loc[:,b_impl_m_eq.columns],
            signals=b_impl_m_eq.shift(1),
            n_portfolios=n_portf)
        pf_flb_eq = poco.get_factor_portfolios(flb, hml=False)

        flb = poco.rank_sort(
            returns=rx_m.loc[:,b_impl_m_eq.columns],
            signals=b_impl_m_bis.shift(1),
            n_portfolios=n_portf)
        pf_flb_bis = poco.get_factor_portfolios(flb, hml=False)

        # self._store_to_hdf(
        #     {"returns/rxm_panel_by_day": rxm_panel_by_day\
        #         .loc[these_cur,this_idx[0]:,:]})
        self._store_to_hdf({"strat": pf_carry})
        self._store_to_hdf({"mom": pf_mom})
        self._store_to_hdf({"eq/flb": pf_flb_eq, "bis/flb": pf_flb_bis})
        self._store_to_hdf({"s_d": s_d, "rx_m": rx_m, "s_m": s_m})
        self._store_to_hdf({
            "eq/b_impl_m": b_impl_m_eq,
            "bis/b_impl_m": b_impl_m_bis})

    @staticmethod
    def smooth_to_monthly(x, wght):
        """
        """
        my_filter = lambda x: x.ewm(alpha=wght).mean()

        return x.resample("M").apply(my_filter).resample("M").last()


if __name__ == "__main__":

    path = setc.gdrive_path("option_implied_betas_project/")
    path_to_spot = setc.gdrive_path("research_data/fx_and_events/")

    path_to_raw = path+"data/raw/longer/"
    path_to_data = path+"data/estimates/"
    tau_str = "1m"
    opt_meth = "mfiv"
    exclude_cur = ["dkk", "sek", "nok"]

    BImpl = ImpliedBetaEnvironment(
        path_to_raw=path_to_raw,
        path_to_data=path_to_data,
        tau_str=tau_str,
        opt_meth=opt_meth,
        ccur="usd",
        exclude_cur=exclude_cur)

    # BImpl.get_mfiv()
    BImpl.get_mfis()

    # exclude_cur = []
    # ipdb.set_trace()
    BImpl.get_covariances()
    # (BImpl._fetch_from_hdf("covariances").apply(np.linalg.det,axis="items") <\
    #     0).count()

    cv = BImpl._fetch_from_hdf("covariances")
    # cv.loc["2008-07-11",:,:].isnull().any().any()

    # # weight
    # wght_curs = pd.read_excel(path_to_spot+"us_imports_m.xlsx",
    #     sheetname="iso")
    # wght_curs = wght_curs.columns
    # wght_bis = pd.read_excel(path_to_spot+"us_imports_m.xlsx",
    #     index_col=0, skiprows=2, header=None)
    # wght_bis.columns = wght_curs
    # wght_bis.index = wght_bis.index.map(lambda x: x+MonthEnd())
    # wght_bis = wght_bis.rolling(12).mean().shift(1)
    # wght_bis = wght_bis[cv.minor_axis]

    wght_bis = pd.read_excel(path+"data/raw/swi_imps_usd_2000_2017_q.xlsx",
        sheetname="data", index_col=0)
    wght_bis.drop(exclude_cur, axis=1, inplace=True)
    wght_bis = wght_bis.rolling(12).mean().shift(1)
    wght_bis = wght_bis.divide(wght_bis.sum(axis=1), axis=0)
    wght_bis = wght_bis.reindex(index=cv.items, method="ffill").ffill()

    BImpl.get_implied_betas(wght_bis=None, exclude_self=True)
    BImpl._fetch_from_hdf("eq/b_impl").describe()
    BImpl._fetch_from_hdf("eq/b_impl").rolling(252).mean().plot()

    # returns data ----------------------------------------------------------
    # weight of currencies in the strat portfolios
    with open(path_to_spot+"data_wmr_dev_m.p", mode='rb') as fname:
        data_m = pickle.load(fname)

    rx_m = data_m["rx"].drop(exclude_cur, axis=1)

    rx_m = into_currency(rx_m, "chf")

    fdisc_m = data_m["fwd_disc"].drop(exclude_cur, axis=1)

    with open(path_to_spot+"data_wmr_dev_d.p", mode='rb') as fname:
        data_d = pickle.load(fname)

    s_d = data_d["spot_ret"]
    s_m = s_d.resample('M').sum()

    # rv_m = s_d.resample('M').std()
    # rv_d = s_d.rolling(252*5).std()
    # rv_m = rv_d.resample('M').last()
    # hml = poco.get_hml(rx_m, rv_m.shift(1), 5)
    # hml.cumsum().plot()
    #
    # bs = get_dynamic_betas(s_d, s_d.mean(axis=1), "grouped_by",
    #     by=pd.TimeGrouper(freq='M'))
    # bs_m = bs.resample('M').last()
    # hml = poco.get_hml(rx_m, bs.shift(1), 5)
    # hml.cumsum().plot()

    # realized betas
    BImpl.get_actual_betas(s_d, s_m, exclude_cur=exclude_cur)

    # additional stuff ------------------------------------------------------
    carry_pf = poco.rank_sort(rx_m, fdisc_m.shift(1), 3)

    wght_grid = poco.hml_weight_grid(carry_pf)
    wght_grid = wght_grid.reindex(
        index=BImpl._fetch_from_hdf("covariances").items,
        method="bfill")

    BImpl.get_implied_betas()
    BImpl._fetch_from_hdf("eq/b_real").loc["2013"]
    BImpl._fetch_from_hdf("eq/b_impl")


    BImpl.get_actual_betas(s_d, s_m, exclude_cur=exclude_cur)
    strat = BImpl.get_fx_strategies("data_wmr_dev_d.p",
        exclude_cur=exclude_cur)

    # # -----------------------------------------------------------------------
    # %matplotlib
    # b_impl = BImpl._fetch_from_hdf("eq/b_impl").dropna()
    # with open(path_to_spot+"data_wmr_dev_m.p", mode='rb') as fname:
    #     rx = pickle.load(fname)["rx"]
    # rx = rx.loc[b_impl.index[0]:,b_impl.columns]
    #

    # strat = BImpl._fetch_from_hdf("strat")
    # (strat["p3"]-strat["p1"]).cumsum().plot()
    # flb = BImpl._fetch_from_hdf("eq/flb")
    # (flb["p3"]-flb["p1"]).cumsum().plot(color='r')

    # BImpl.get_fx_strategies("data_dev_d.p", exclude_cur=exclude_cur)

    #
    # b_impl_full_bis
    # b_impl_full_eq
    # b_impl_trim_bis.loc["2013-01-08":]
    # b_impl_trim_eq.loc["2013-01-08":]
    #
    # b_impl_d = b_impl_full_bis
    # f, ax = plt.subplots()

#     with pd.HDFStore(path_to_data+"data_vs_usd_1m_mfiv9_curs.h5", mode="r") \
#         as hangar:
#         mfis = hangar["skewness"]
#
# curs = [p for p in mfis.columns if "usd" in p]
# mfis = mfis.loc[:,curs]
# curs = [re.sub("usd", '', p) for p in curs]
#
# %matplotlib inline
# mfis.plot()
# which_usd_first = [p for p in curs if p[:3] == "usd"]
# mfis.loc[:,which_usd_first] *= -1
#
# mfis.columns = [re.sub("usd", '', p) for p in mfis.columns]
#
# with open(path_to_spot+"data_wmr_dev_m.p", mode="rb") as hangar:
#     fx = pickle.load(hangar)
#
# rx_m = fx["rx"]
#
# with open(path_to_spot+"data_wmr_dev_d.p", mode="rb") as hangar:
#     fx_d = pickle.load(hangar)
#
# s_d = fx_d["spot_ret"]
# from scipy import stats
# roll_sk = s_d.resample('M').apply(stats.skew).loc["2008-08":]
#
# drop_curs = ["dkk","nok","sek"]
#
# yeah = poco.get_hml(rx_m.drop(drop_curs, axis=1).loc["2008-09":],
#     (mfis.resample('M').mean()-roll_sk).shift(1), n_portf=3)
#
# yeah.cumsum().plot()
# from foolbox.api import taf
# taf.descriptives(yeah.to_frame())
