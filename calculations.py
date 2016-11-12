import numpy as np
import pandas as pd
import os

usr = "hsg-spezial"
path = "c:/users/" + usr + "/google drive/" + \
    "personal/option_implied_betas_project/"

import config
config.init()

# settings --------------------------------------------------------------------
# contraints: ratio of sigmas <4/3 (only works with opt_meth = "SLSQP")
C = np.array([
    [0, 0],
    [0, 0],
    [-1, 4/3],
    [4/3, -1]])
constraints = {
    "type" : "ineq",
    "fun" : lambda x: x.dot(C)}
# maturities
tau_in_months = 1
tau = tau_in_months/12.0
tau_str = str(tau_in_months)+"m"
# optimization method
opt_meth = "SLSQP"

config.init()
config.cfg_dict["tau"] = tau
config.cfg_dict["opt_meth"] = opt_meth
config.cfg_dict["constraints"] = constraints
# end of settings -------------------------------------------------------------

import optools as op
import import_data as imp
# import ipdb
import logging

# logger settings
logging.basicConfig(filename=path+"log/optools_logger.txt",
    filemode='w',
    format="%(asctime)s || %(levelname)s:%(message)s",
    datefmt="%H:%M:%S")

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

if __name__ == "__main__":

    # fetch all files with raw data
    all_files = list(filter(lambda x: x.endswith(".xlsx"),
        os.listdir(path + "data/raw/")))

    files = all_files
    # # fetch existing estimates to build intersection
    # x_files = [p[:6] for p in os.listdir(path+"data/estimates/par/")]
    # files = [p for p in all_files if p[:6] not in x_files]

    for filename in files:

        # collect data from .xlsx file
        # logger.info("collecting data for %s" % filename[:6])
        data_for_est = imp.import_data(
            data_path=path+"data/raw/",
            filename=filename,
            tau_str=tau_str)

        # main routine
        par = \
            op.estimation_wrapper(data_for_est,
                tau, parallel=True)

        # save
        par.to_csv(path + "data/estimates/par/" + filename[:6] + \
            "_" + tau_str + "_" + opt_meth[:4].lower() + "_par.csv")
