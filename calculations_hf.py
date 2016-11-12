# calculations from high-frequency data
import pandas as pd
import optools_wrappers as wrap
import datetime
import import_data as imp
import matplotlib.pyplot as plt
import logging
# %matplotlib inline

usr = "hsg-spezial"
path = "c:/users/" + usr + "/google drive/" + \
    "personal/option_implied_betas_project/"

# logger settings
logging.basicConfig(filename=path+"log/hf_logger.txt",
    filemode='w',
    format="%(asctime)s || %(levelname)s:%(message)s",
    datefmt="%H:%M:%S")
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

data_path = path+"data/raw/hf/"
filename = "eurchf_fx_deriv_hf.xlsx"

# currencies
base_cur = "eur"
counter_cur = "chf"

# # store data as .hp in data/raw/hf and get time index
# date_idx = imp.import_data_hf(data_path=data_path, filename=filename)

if __name__ == "__main__":
    # sip at .h5
    storage = pd.HDFStore(path+"data/estimates/hf/"+base_cur+counter_cur+\
        ".h5", mode='w')
    hangar = pd.HDFStore(data_path+base_cur+counter_cur+"_panel.h5", mode='r')
    date_idx = pd.read_hdf(hangar, "temp").index

    # day by day
    dates_only = pd.unique(date_idx.map(lambda x: x.date))
    for p in dates_only[1:-1]:
        # p = dates_only[1:-1][0]
        dt_lo = p
        dt_hi = dt_lo + datetime.timedelta(days=1)

        # extract a piece
        condition_panel = "major_axis >= dt_lo and major_axis < dt_hi"
        condition_series = "index >= dt_lo and index < dt_hi"
        day_piece = pd.read_hdf(hangar, "panel", where = condition_panel)
        s = pd.read_hdf(hangar, "s", where = condition_series)
        # day_panel.loc["1W",:,].index
        # day_panel.major_axis

        res = wrap.wrapper_rnd_nonparametric(day_panel=day_piece, s=s,
            maturity="1M", h=0.22)
        storage.put(key='d'+str(p).replace('-',''), value=res)

        # fig, ax = plt.subplots(figsize=(7,7/4*3))
        # res.plot(ax=ax)
        # plt.show()

    hangar.close()
    storage.close()











#
# df_stacked = pd.DataFrame(columns=["iv", "f", "K", "tau"])
#
# # 1-day loop over maturities
# for tau_str, df in deriv_panel.loc[:,:"2016-10-05 23:45:00",:].iteritems():
#     # convert string maturity e.g. "3W" into float e.g. 3/52
#     # if ends with 'W' -> weekly; 'M' -> monthly; 'Y' -> yearly
#     if tau_str.endswith('W'):
#         tau = float(tau_str[:-1])/52
#     elif tau_str.endswith('M'):
#         tau = float(tau_str[:-1])/12
#     else:
#         tau = float(tau_str[:-1])
#
#     # loop over time stamps within each maturity
#     for time_idx, row in df.iterrows():
#         # get deltas and ivs
#         deltas, ivs = op.get_wings(
#             row["rr25d"],row["rr10d"],row["bf25d"],row["bf10d"],row["atm"],
#             row["rf_counter"], tau)
#         # transform deltas to strikes
#         strikes = op.strike_from_delta(delta=deltas,
#             X=s.loc[time_idx], rf=row["rf_base"], y=row["rf_counter"],
#             tau=tau, sigma=ivs, is_call=True)
#         # store everything
#         tmp_df = pd.DataFrame.from_dict(
#             {
#                 "iv" : ivs,
#                 "f" : np.ones(5)*row["f"],
#                 "K" : strikes,
#                 "tau" : np.ones(5)*tau
#             })
#         # merge with df_stacked
#         df_stacked = pd.concat((df_stacked, tmp_df), axis=0, ignore_index=True)
#
# if __name__ == "__main__":
#     # y is iv
#     y = df_stacked["iv"]
#     X = df_stacked.drop(["iv",],axis=1)
#
#     # prepare values at which predictions to be made
#     # strikes are equally spaced [min(K), max(K)], memento prewhiten!
#     dK = 1e-05
#     K_pred = np.arange(min(df_stacked["K"]), max(df_stacked["K"]), dK)
#     # forward is mean forward price over that day, memento prewhiten!
#     f_pred = np.ones(len(K_pred))*df_stacked["f"].mean()
#     # maturity is 1 month, memento prewhiten!
#     tau_pred = np.ones(len(K_pred))*1/12
#     # all together
#     X_pred = np.stack((K_pred, f_pred, tau_pred), axis=1)
#
#     #
#     mod = regm.KernelRegression(y0=y, X0=X)
#     # mod.bleach(z_score=True, add_constant=False, dropna=True)
#     h_opt = mod.cross_validation(k=10)
#     print(mod.mu)
#     print(mod.sigma)
#     y_hat = mod.fit(X_pred=X_pred, h=h_opt*2)
#     domain = K_pred
#     fig, ax = plt.subplots(figsize=(7,7/4*3))
#     ax.plot(K_pred, y_hat, "k-")
#     fig.savefig(path+"pics/hf/smile.png")
#
#     # from iv to price
#     rf = deriv_panel.loc["1M",:,"rf_base"].mean()
#     C = op.bs_price(f_pred, K_pred, rf, 1/12, y_hat)
#
#     # derivative
#     d2C = np.exp(rf*1/12)*(C[2:] - 2*C[1:-1] + C[:-2])/dK**2
#     fig, ax = plt.subplots(figsize=(7,7/4*3))
#     ax.plot(K_pred[1:-1], d2C, "k-", label='non-parametric')
#     fig.savefig(path+"pics/hf/rnd_compare.png")
#
#     d2C[abs(d2C) < 1e-05] = 0
#     (d2C < 0).any()
#     np.trapz(d2C, K_pred[1:-1])
#     np.trapz(d2C*K_pred[1:-1], K_pred[1:-1])

    # # get parametric density from calculations_from_parameters and plot
    # ax.plot(domain, p, color='r', label='parametric')
    # ax.set_xlim((0.95, 1.2))
    # ax.grid(True)
    # ax.legend(loc='upper left')
    # ax.set_title("2016-10-05")
    # fig
    #
    # domain = np.linspace(0.6, 1.4, 250)
