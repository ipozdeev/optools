import pandas as pd
import pickle
import matplotlib.pyplot as plt
from foolbox.api import poco, taf
from foolbox.linear_models import get_dynamic_betas

# %matplotlib inline

class ForwardLookingBeta():
    """
    """
    def __init__(self, b_daily, path_to_spot):
        """
        """
        # returns and discounts ---------------------------------------------
        # monthly
        with open(path_to_spot+"data_wmr_dev_m.p", mode='rb') as fname:
            data_m = pickle.load(fname)

        s_m = data_m["spot_ret"].loc[:,b_daily.columns]
        rx_m = data_m["rx"].loc[:,b_daily.columns]
        fdisc_m = data_m["fwd_disc"].loc[:,b_daily.columns]

        # dollar factor
        dol_spot_m = s_m.mean(axis=1)
        dol_rx_m = rx_m.mean(axis=1)

        # daily
        with open(path_to_spot+"data_wmr_dev_d.p", mode='rb') as fname:
            data_d = pickle.load(fname)

        s_d = data_d["spot_ret"].loc[:,b_daily.columns]
        fdisc_d = data_d["fwd_disc"].loc[:,b_daily.columns]

        # dollar factor
        dol_spot_d = s_d.mean(axis=1)

        self.s_m = s_m
        self.rx_m = rx_m
        self.fdisc_d = fdisc_d
        self.fdisc_m = fdisc_m
        self.dol_spot_d = dol_spot_d
        self.dol_spot_m = dol_spot_m
        self.dol_rx_m = dol_rx_m

        self.b_daily = b_daily

    @staticmethod
    def smooth_to_monthly(x, wght):
        """ Resample monthly by taking ewma with given weight at month's end.
        """
        my_filter = lambda x: x.ewm(alpha=wght).mean()[-1]
        res = x.resample("M").apply(my_filter)

        return res

    def construct_strategy(self, asset=None, signal=None, n_portf=3):
        """
        Parameters
        ----------
        signal : pd.DataFrame
            of signals, must be shifted already thus aligned with `asset`
        """
        if asset is None:
            asset = self.rx_m
        if signal is None:
            signal = self.smooth_to_monthly(self.b_daily, 0.5).shift(1)

        # sort
        ps = poco.rank_sort(asset, signal, n_portf)
        flb_pf = poco.get_factor_portfolios(ps, hml=True)

        self.flb_hml = flb_pf["hml"]
        self.flb_pf = flb_pf.drop("hml", axis=1)

        return flb_pf["hml"], flb_pf.drop("hml", axis=1)

    def compare_to_carry(self, asset=None, signal=None, n_portf=3):
        """
        """
        # flb (or something else)
        flb_hml, _ = self.construct_strategy(asset, signal, n_portf)
        flb_hml.name = "flb"

        # carry
        carry_hml, _ = self.construct_strategy(
            asset=self.rx_m,
            signal=self.fdisc_m.shift(1),
            n_portf=n_portf)
        carry_hml.name = "carry"

        # align
        both = pd.concat((flb_hml, carry_hml), axis=1)
        both = both.loc[flb_hml.first_valid_index():,:]

        # ap tests
        tests = taf.ts_ap_tests(carry_hml.to_frame(), flb_hml.to_frame())

        # plot
        fig, ax = plt.subplots(figsize=(8.4,11.7/2))
        both.cumsum().plot(ax=ax)

        return both, (ax, fig), tests

    def sort_on_covariance(self, window_1=22, window_2=66, smooth_wght=0.95,
        n_portf=3):
        """
        """
        prem = self.dol_spot_d
        # prem = self.dol_spot_d.rolling(window_1).sum()

        # signal based on covariance
        cov_sig = get_dynamic_betas(
            Y=self.b_daily,
            x=prem.loc[self.b_daily.index[0]:],
            method="rolling",
            window=window_2)

        # smooth covariance
        cov_sig = self.smooth_to_monthly(cov_sig, smooth_wght)

        # sort
        covb_hml, _ = self.construct_strategy(
            signal=cov_sig.shift(1), n_portf=n_portf)

        return covb_hml


if __name__ == "__main__":
    # settings --------------------------------------------------------------
    from foolbox.api import set_credentials
    from implied_beta_functions import ImpliedBetaEnvironment

    path = set_credentials.gdrive_path("option_implied_betas_project/")
    path_to_spot = set_credentials.gdrive_path("research_data/fx_and_events/")

    tau_str = "1m"
    opt_meth = "mfiv"

    w = "bis"

    n_portf = 2

    # environment -----------------------------------------------------------
    BImpl = ImpliedBetaEnvironment(
        path_to_raw=path+"data/raw/longer/",
        path_to_data=path+"data/estimates/",
        tau_str=tau_str,
        opt_meth=opt_meth)

    b_impl_d = BImpl._fetch_from_hdf(w+"/b_impl")

    b_ols = BImpl._fetch_from_hdf(w+"/b_roll")
    b_ols = b_ols.drop(["dkk"], axis=1, errors="ignore")\
        .loc[b_impl_d.index[0]:]

    # environment -----------------------------------------------------------
    flb = ForwardLookingBeta(b_impl_d, path_to_spot)

    # plot
    temp = flb.construct_strategy(
        signal=b_impl_d.resample('M').last().shift(1),
        n_portf=n_portf)
    temp = flb.construct_strategy(
        signal=b_impl_d.shift(1).shift(1),
        n_portf=n_portf)

    master_flb = temp[0]
    master_flb.cumsum().plot()
    taf.descriptives(master_flb.loc["2008-09":"2017"].to_frame())
    from statsmodels.api import OLS
    mod = OLS(master_flb.loc["2008-09":], master_flb.loc["2008-09":]*0.0+1)
    res = mod.fit(cov_type="HAC", cov_kwds={"maxlags":1})
    res.params / res.bse

    0.07/np.sqrt(master_flb.count())*2

    # compare to carry
    both, _, tests = flb.compare_to_carry()

    # second sort
    covb_hml = flb.sort_on_covariance()
    covb_hml.cumsum().plot()

    # many monthly ----------------------------------------------------------
    with open(path_to_spot+"data_wmr_dev_d.p", mode='rb') as fname:
        data_d = pickle.load(fname)

    s_d = data_d["spot_ret"].loc[:,b_impl_d.columns]
    fdisc_d = data_d["fwd_disc"].loc[:,b_impl_d.columns]

    all_m = poco.many_monthly_rx(
        s_d=data_d["spot_ret"].loc["2007":], f_d=data_d["fwd_disc"].loc["2007":])
    all_m[max(all_m.keys())+1] = rx_m

    flbs = pd.DataFrame(columns=range(len(all_m.keys())), index=b_impl_d.index)

    # run strategies
    for p in range(len(all_m.keys())):
        # p = 0
        ret = all_m[p].loc[b_impl_d.index[0]:]
        # betas
        this_b = b_impl_d.ffill().reindex(index=ret.index, method="ffill")
        # flb strategy
        this_flb = poco.get_hml(ret, this_b.shift(1), n_portf=n_portf)
        this_flb.name = "flb"
        # store
        flbs.loc[:,p] = this_flb.loc["2008-09":]

    # plot
    fig, ax = plt.subplots()
    flbs.fillna(value=0.0).cumsum().plot(ax=ax, color="gray")
    ax.legend_.remove()
    flbs.mean(axis=1).resample('M').mean().cumsum().plot(
        ax=ax, color='k', linewidth=1.5)
    taf.descriptives(flbs.mean(axis=1).resample('M').mean().to_frame())

    # ols betas -------------------------------------------------------------
    dol_d = s_d.mean(axis=1)

    #
    b_ols = get_dynamic_betas(s_d, dol_d, "rolling", window=120)
    b_ols = b_ols.loc[b_impl_d.index[0]:,b_impl_d.columns]

    # check on many monthly betas -------------------------------------------
    beta_to_test = b_impl_d.loc["2013-01-08":]

    fig_flb, ax_flb = plt.subplots(figsize=(8.4,11.7/2))
    fig_carry, ax_carry = plt.subplots(figsize=(8.4,11.7/2))
    fig_b_carry, ax_b_carry = plt.subplots(figsize=(8.4,11.7/2))
    fig_b_carry_enhcd, ax_b_carry_enhcd = plt.subplots(figsize=(8.4,11.7/2))

    ax_flb.set_title("flb")
    ax_carry.set_title("carry")
    ax_b_carry.set_title("b_carry")
    ax_b_carry_enhcd.set_title("ax_b_carry")

    all_res = pd.Panel(
        items=list(range(22)),
        major_axis=["carry","se_carry"],
        minor_axis=["alpha","flb","adj_r_sq"])

    carries = pd.DataFrame(columns=range(22), index=beta_to_test.index)
    flbs = carries.copy()
    b_carries = carries.copy()
    b_carries_enhcd = carries.copy()

    descr = pd.DataFrame(columns=range(22), index=range(9))

    #
    vcv = BImpl._fetch_from_hdf("covariances")

    # loop over possible versions of carry trade
    for p in range(22):
        # p = 0
        # select ------------------------------------------------------------
        # returns
        ret = all_m[p].loc[beta_to_test.index[0]:]
        # betas
        b = beta_to_test.loc[ret.index,:]
        # forward discounts
        fd = fdisc_d.loc[ret.index,:]

        # flb strategy
        this_flb = poco.get_hml(ret, b.shift(1), n_portf=n_portf)
        this_flb.name = "flb"
        # respective carry
        this_carry = poco.get_hml(ret, fd.shift(1), n_portf=n_portf)
        this_carry.name = "carry"

        # carry
        carry_pf = poco.rank_sort(ret, fd.shift(1), n_portf)
        wght_grid = poco.hml_weight_grid(carry_pf).fillna(0)
        b_carry = BImpl.get_ib(vcv, wght_grid.loc[:vcv.items[-1]].shift(-1))
        b_carry_hml = poco.get_hml(ret, b_carry.shift(1), n_portf)

        # within carry
        sig_hi = carry_pf["portfolio3"]/carry_pf["portfolio3"]
        sig_hi = sig_hi.divide(sig_hi.sum(axis=1), axis=0).fillna(0)
        sig_lo = carry_pf["portfolio1"]/carry_pf["portfolio1"]
        sig_lo = sig_lo.divide(sig_lo.sum(axis=1), axis=0).fillna(0)
        sig_mid = carry_pf["portfolio2"]/carry_pf["portfolio2"]
        sig_mid = sig_mid.divide(sig_mid.sum(axis=1), axis=0).fillna(0)

        b_hi = b_carry.multiply(sig_hi,axis=0).sum(axis=1).shift(1)
        b_mid = b_carry.multiply(sig_mid,axis=0).sum(axis=1).shift(1)
        b_lo = b_carry.multiply(sig_lo,axis=0).sum(axis=1).shift(1)

        b_all = pd.concat((b_lo, b_mid, b_hi), axis=1)
        b_all.columns = ["p1","p2","p3"]

        carry_pfs = poco.get_factor_portfolios(carry_pf, hml=False)
        b_carry_enhcd = poco.get_hml(carry_pfs, b_all, 3)

        descr.loc[:,p] = taf.descriptives(this_flb.to_frame())

        # ap test
        ap_res = taf.ts_ap_tests(
            this_carry.to_frame(), this_flb.to_frame(), scale=12)
        all_res.loc[p,:,:] = ap_res

        # plot
        this_flb.cumsum().plot(ax=ax_flb, color="gray")
        this_carry.loc["2013-03":].cumsum().plot(ax=ax_carry, color="gray")
        b_carry_hml.loc["2013-03":].cumsum().plot(ax=ax_b_carry, color="gray")
        b_carry_enhcd.loc["2013-03":].cumsum().plot(
            ax=ax_b_carry_enhcd, color="gray")

        # store
        flbs.loc[:,p] = this_flb
        carries.loc[:,p] = this_carry
        b_carries.loc[:,p] = b_carry_hml
        b_carries_enhcd.loc[:,p] = b_carry_enhcd

    descr.index = taf.descriptives(this_flb.to_frame()).index

    fig, ax = plt.subplots(figsize=(8.4,11.7/2))
    all_res.loc[:,"carry","adj_r_sq"].hist()

    carries.corrwith(flbs)

    all_res.loc[:,"carry","flb"].hist()

    # corr of fwd disc and betas
    fdisc_d.corrwith(b_impl_d)
    (s_d.loc["2013":,"aud"]*100).plot()

    aud = s_d.loc["2013":,"aud"].where(s_d.loc["2013":,"aud"] < -0.01).dropna()
    b_impl_d.loc[:,"aud"].diff().shift(1).loc[aud.index]

    flbs.cumsum().plot()

    # -------------------------
    with open(path_to_spot+"data_wmr_dev_m.p", mode='rb') as fname:
        data_m = pickle.load(fname)

    fdisc_m = data_m["fwd_disc"]
    rx_m = data_m["rx"]

    b_impl_m = b_impl_d.resample("M").last()
    b_impl_m = b_impl_d.rolling(66).mean().resample("M").last()
    b_impl_m.loc[:,"gbp"].plot(ax=ax)

    flb = poco.get_hml(rx_m, b_impl_m.shift(1), 3)
    flb.cumsum().plot()

    # realized
    dol_d = s_d.mean(axis=1)
    b_ols = get_dynamic_betas(s_d, dol_d, method="grouped_by",
        by=[lambda x: x.year, lambda x: x.month])
    b_ols = get_dynamic_betas(s_d, dol_d, method="rolling",
        window=140)
    b_ols = b_ols.resample('M').last()

    # reindex a bit
    b_ols.index = [MonthEnd().rollforward(
        datetime.date(p[0], p[-1], 1)) for p in b_ols.index]

    fig, ax = plt.subplots()
    poco.get_hml(rx_m.loc["2013":], b_ols.shift(1), 3).cumsum().plot(ax=ax)
    rx_m.loc["2013":].mean().mean()*12*100

    rx_m.loc["2013":].mean(axis=1).cumsum().plot()

    carry_m = poco.get_hml(rx_m.loc["2013":], fdisc_m.shift(1), 3)
    carry_m.name = "carry"
    dol_m = rx_m.loc["2013":].mean(axis=1)
    dol_m.name = "dol"
    flb_m = poco.get_hml(rx_m, b_impl_m.shift(1), 3)
    flb_m.name = "flb"
    ols_m = poco.get_hml(rx_m.loc["2013":], b_ols.shift(1), 3)
    ols_m.name = "ols_m"
    fishy_m = poco.get_hml(rx_m.loc["2013":], (b_ols-b_impl_m).shift(1), 3)
    fishy_m.name = "fishy"
    fishy_m.cumsum().plot(ax=ax)

    cucat = pd.concat((carry_m, dol_m, flb_m, ols_m), axis=1)
    taf.descriptives(cucat*100, 12)
    flb_m.cumsum().plot()

    taf.ts_ap_tests(flb_m.to_frame(), dol_m.to_frame())

    (s_d.ewm(0.5).std()*100).loc["2002-06"].plot()


    (b_impl_m-b_ols).dropna().plot()

    # -----------------------------------------------------------------------
    flb_ps = poco.rank_sort(s_d, b_impl_d.shift(1), n_portf)
    flb_hml = poco.get_factor_portfolios(flb_ps, hml=True).loc[:,"hml"]
    flb_hml.cumsum().plot()

    s_m = flb.s_m
    rx_m = flb.rx_m

    flb_ps = poco.rank_sort(rx_m,
        b_impl_d.resample('M').last().shift(1), 3)
    flb_hml = poco.get_factor_portfolios(flb_ps, hml=True).loc[:,"hml"]
    flb_hml.cumsum().plot()

    taf.descriptives(flb_hml.to_frame())

    # -------------------
    b_gap = BImpl._fetch_from_hdf("eq/b_gap", s_dt)
    lol = poco.rank_sort(rx_m, b_impl_d.resample('M').last().shift(1), 3)
    poco.get_factor_portfolios(lol, hml=True).hml.cumsum().plot(ax=ax,
        linewidth=1.5, color=my_palette[1])


    carry_pf = poco.rank_sort(s_m, fdisc_m.loc["2013-01-08":], n_portf)
    wght_grid = poco.hml_weight_grid(carry_pf).fillna(0)
    wght_grid = wght_grid.loc["2013-01-08":"2017-02"]
    b_carry = BImpl.get_ib(vcv, wght_grid)
    b_carry_hml = poco.get_hml(ret, b_carry.shift(1), n_portf)

    fig, ax = plt.subplots(figsize=(8.3, 8.3/1.5))
    flb_hml = BImpl._fetch_from_hdf("b").loc["2013-01-08":,"gbp"]
    flb_hml.plot(ax=ax,
        color=my_palette[0], linewidth=1.5)
