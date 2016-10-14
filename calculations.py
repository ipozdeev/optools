# here be calculations
import optools as op
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import logging

usr = "hsg-spezial"
path = "c:/users/" + usr + "/google drive/" + \
    "personal/option_implied_betas_project"

# logger settings
logging.basicConfig(filename=path+"/log/optools_logger.txt",
    filemode='w',
    format="%(asctime)s || %(levelname)s:%(message)s",
    datefmt="%H:%M:%S")

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# # load existing data
# filename = "c:/users/hsg-spezial/google drive/" + \
#     "personal/research_proposal/option_implied_betas/est_res/" + \
#     "eurchf_11_16_d_perc.csv"
# ps = pd.read_csv(filename, index_col=0, parse_dates=True)
#
# # check if integrates to 1
# np.trapz(ps.ix[1000,], [float(p) for p in ps.columns])
#
# # plot 10th and 90th percentiles
# ps.plot()
# plt.show()

# #
# ps.loc["2012-04":"2013-09"]

# # debug
# quant.loc["2013-07-22":"2013-08-12"].plot()
# plt.show()
#
# data_for_est.loc["2013-07-25":"2013-08-09"].pct_change().plot()
# par.loc["2013-07-25":"2013-08-09"]
#
# save_this = data_for_est
# data = data_for_est.loc["2013-07-30":"2013-08-06"]

if __name__ == "__main__":

    data = pd.read_csv(path+"/data/data_for_est.csv", index_col=0,
        parse_dates=True)

    # constraints: ratio of sigmas <4/3
    C = np.array([
        [0, 0],
        [0, 0],
        [-1, 4/3],
        [4/3, -1]])

    # constraints
    constraints = {
        "type" : "ineq",
        "fun" : lambda x: x.dot(C)}

    # support of rnd
    domain = np.arange(0.8, 1.5, 0.005)
    # percentiles to calculate
    perc = np.array([0.1, 0.5, 0.9])

    # fetch specific maturity
    tau = 1/12
    tau_str = "1m"
    data_for_est = data[list(data.columns[:7]) + \
        ["f"+tau_str, "chf"+tau_str, "eur"+tau_str]]

    # rename
    data_for_est.columns = list(data.columns[:7]) + ["f", "rf", "y"]

    # from forward points to forward
    data_for_est.loc[:,"f"] = data_for_est["s"]+data_for_est["f"]/10000

    dens, par, perc = \
        op.estimation_wrapper(data_for_est.loc["2013-07-30":"2013-08-06"],
            tau, constraints, domain, perc)

    # # save
    # par.to_csv(path + "/calc/res/eurchf_11_16_d_par.csv")
    # dens.to_csv(path + "/calc/res/eurchf_11_16_d_dens.csv")
    # quant.to_csv(path + "/calc/res/eurchf_11_16_d_perc.csv")
