# here be dragons
import optools as opt
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt

# import data
user = "hsg-spezial"
data_path = "c:/users/"+user+\
    "/google drive/personal/research_proposal/option_implied_betas/data/"

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
    sheetname="SF",
    skiprows=0,
    header=0)
sf = pd.concat(
    objs=[pd.DataFrame(data=sf.ix[:,1].values,index=sf.ix[:,0].values),
        pd.DataFrame(data=sf.ix[:,4].values,index=sf.ix[:,2].values)],
    axis=1,
    ignore_index=True)

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
rf = rf/100

# merge everything
data = deriv.join(sf, how="inner").join(rf, how="inner")

# drop na
data.dropna(inplace=True)
