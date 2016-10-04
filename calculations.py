# here be calculations
import optools as opt
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt

# load existing data
filename = "c:/users/hsg-spezial/google drive/personal/research_proposal/option_implied_betas/est_res/eurchf_11_16_d.csv"

ps = pd.read_csv(filename, index_col=0, parse_dates=True)
np.trapz(ps.ix[1,], [float(p) for p in ps.columns])
