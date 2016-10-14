# here be dragons
import optools as opt
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt

# import data
usr = "hsg-spezial"
data_path = "c:/users/"+usr+\
    "/google drive/personal/option_implied_betas_project/data/"

# read in: contracts
deriv = pd.read_excel(
    io=(data_path+"eurchf_fx_deriv.xlsx"),
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

# from percentage to fractions of 1
deriv = deriv/100

# read in: S,F
sf = pd.read_excel(
    io=data_path+"eurchf_fx_deriv.xlsx",
    sheetname="S,F",
    skiprows=5,
    header=None)

sf = pd.concat(
    objs=[pd.DataFrame(
        sf.ix[:,p*2+1].values, index=sf.ix[:,p*2].values)
            for p in range(3)],
    axis=1,
    ignore_index=True)

sf.columns = ["s","f1m","f3m"]

# read in: rf
rf = pd.read_excel(
    io=data_path+"eurchf_fx_deriv.xlsx",
    sheetname="RF",
    skiprows=5,
    header=None)

rf = pd.concat(
    objs=[pd.DataFrame(
        rf.ix[:,p*2+1].dropna().values, index=rf.ix[:,p*2].dropna().values)
            for p in range(4)],
    axis=1,
    ignore_index=True)

rf.columns = ["eur1m", "eur3m", "chf1m", "chf3m"]
rf = rf/100

# merge everything
data = deriv.join(sf, how="inner").join(rf, how="inner")

# drop na
data.dropna(inplace=True)
