# here be dragons
import pandas as pd

def import_data(data_path, filename, tau_str):
    """ Read in info from `filename`

    File has to contain sheets named "1M" "3M" which store quotes of option
    contracts
    """
    base_cur = filename[:3]
    counter_cur = filename[3:6]

    # read in: contracts ------------------------------------------------------
    deriv = pd.read_excel(
        io=data_path+filename,
        sheetname=tau_str.upper(),
        skiprows=8,
        header=None)

    # fetch pairs of (dates, values), remove nan
    deriv = [deriv.ix[:,(p*2):(p*2+1)].dropna() for p in range(5)]

    # transform each pair into DataFrame indexed by first column (dates)
    for p in range(5):
        deriv[p].index = deriv[p].pop(p*2)

    # concatenate pairs
    deriv = pd.concat(deriv, axis=1, ignore_index=True)

    # rename
    deriv.columns = ["rr10d", "rr25d", "bf10d", "bf25d", "atm"]

    # volatility from percentage to fractions of 1
    deriv = deriv/100

    # read in: S,F ------------------------------------------------------------
    sf = pd.read_excel(
        io=data_path+filename,
        sheetname="S,F",
        skiprows=5,
        header=None)

    sf = [sf.ix[:,(p*2):(p*2+1)].dropna() for p in range(3)]

    for p in range(3):
        sf[p].index = sf[p].pop(p*2)

    sf = pd.concat(sf, axis=1, ignore_index=True)

    sf.columns = ["s","f1m","f3m"]

    # choose required maturity
    sf = sf[["s", "f"+tau_str]]

    # read in: rf -------------------------------------------------------------
    # converters for dates (experimental)
    converters = {}
    for p in range(4):
        converters[p*2] = lambda x: pd.to_datetime(x)
    rf = pd.read_excel(
        io=data_path+filename,
        sheetname="RF",
        skiprows=5,
        header=None,
        converters=converters)

    rf = [rf.ix[:,(p*2):(p*2+1)].dropna() for p in range(4)]

    for p in range(4):
        rf[p].index = rf[p].pop(p*2)

    rf = pd.concat(rf, axis=1, ignore_index=True)

    rf.columns = [base_cur+"1m", base_cur+"3m",
        counter_cur+"1m", counter_cur+"3m"]

    # select required maturity
    rf = rf[[base_cur+tau_str, counter_cur+tau_str]]

    rf = rf/100

    # merge everything --------------------------------------------------------
    data = deriv.join(sf, how="inner").join(rf, how="inner")

    # select relevant maturity ------------------------------------------------
    data = data[list(deriv.columns) + ["s",] + \
        ["f"+tau_str, counter_cur+tau_str, base_cur+tau_str]]

    # rename
    data.columns = list(deriv.columns) + ["s",] + ["f", "rf", "y"]

    # from forward points to forward ------------------------------------------
    data.loc[:,"f"] = data["s"]+data["f"]/10000

    # drop na -----------------------------------------------------------------
    data.dropna(inplace=True)

    return data
