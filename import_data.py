# here be dragons
import optools as opt
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt

# parameters
freq_mult = 1/12

# import data
user = "hsg-spezial"
data_path = "c:/users/"+user+\
    "/google drive/personal/research_proposal/option_implied_betas/data/"
# date_parser = lambda x: dt.datetime.strptime(x, "%Y-%m-%d")
# float_parser = lambda x: float(x)
# read in: contracts
deriv = pd.read_excel(
    io=data_path+"eurchf_fx_deriv.xlsx",
    sheetname="1M",
    skiprows=8,
    header=None)
# disassemble and concatenate
deriv = pd.concat(
    objs=[pd.DataFrame(
        data=deriv.ix[:,p*2+1].values, index=deriv.ix[:,p*2].values)
            for p in range(5)],
    axis=1,
    ignore_index=True)
# rename
deriv.columns = ["rr10d", "rr25d", "bf10d", "bf25d", "atm"]
deriv = deriv/100

# read in: S,F
sf = pd.read_excel(
    io=data_path+"eurchf_fx_deriv.xlsx",
    sheetname="SF",
    skiprows=0,
    header=0)
sf = pd.concat(objs=[pd.DataFrame(data=sf.ix[:,1].values,index=sf.ix[:,0].values), pd.DataFrame(data=sf.ix[:,4].values,index=sf.ix[:,2].values)],axis=1,ignore_index=True)
sf.columns = ["s","f"]

# read in: rf
rf = pd.read_excel(
    io=data_path+"eurchf_fx_deriv.xlsx",
    sheetname="RF",
    skiprows=0,
    header=0)
rf_eur = pd.DataFrame(data=rf.ix[:,1].values,index=rf.ix[:,0].values)
rf_chf = pd.DataFrame(data=rf.ix[:,3].values,index=rf.ix[:,2].values)
rf = pd.concat([rf_eur.dropna(), rf_chf.dropna()], axis=1, ignore_index=True)
rf.columns = ["eur", "chf"]
rf = rf/100*freq_mult

# align everything
data = deriv.join(sf, how="inner").join(rf, how="inner")
data.dropna(inplace=True)

# # estimate one date
# gen_date = np.random.choice(data.index.values, size=1)
# # fetch wings
# deltas, ivs = opt.get_wings(
#     data.loc[gen_date, "rr25d"].values[0],
#     data.loc[gen_date, "rr10d"].values[0],
#     data.loc[gen_date, "bf25d"].values[0],
#     data.loc[gen_date, "bf10d"].values[0],
#     data.loc[gen_date, "atm"].values[0],
#     data.loc[gen_date, "eur"].values[0],
#     1)
#
# # to strikes
# K = opt.strike_from_delta(
#     deltas,
#     data.loc[gen_date, "s"].values[0],
#     data.loc[gen_date, "chf"].values[0],
#     data.loc[gen_date, "eur"].values[0],
#     1,
#     ivs,
#     True)
#
# # weighting matrix: inverse squared vegas
# W = opt.bs_vega(
#     data.loc[gen_date, "f"].values[0],
#     K,
#     data.loc[gen_date, "chf"].values[0],
#     data.loc[gen_date, "eur"].values[0],
#     1,
#     ivs)
# W = np.diag(1/(W*W))
#
# # set constraints
# C = np.array([
#     [0, 0],
#     [0, 0],
#     [-1, 1],
#     [4/3, -3/4]])
#
# # estimate rnd!
# res = opt.estimate_rnd(
#     ivs,
#     data.loc[gen_date, "f"].values[0],
#     K,
#     data.loc[gen_date, "chf"].values[0],
#     True,
#     W,
#     constraints={
#         "type" : "ineq",
#         "fun" : lambda x: x.dot(C)
#     })
#
# # plot density
# ln_mix = opt.lognormal_mixture(res[1][:2], res[1][3:], res[0])
# x = np.arange(0.5, 1.7, 0.005)
# p = ln_mix.pdf(x)
#
# plt.plot(x,p)
# plt.show()
