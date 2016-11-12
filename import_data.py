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

def import_data_hf(data_path, filename):
    """ Read in info from `filename`, high-frequency version

    """
    # data_path = path+"data/raw/hf/"
    # filename = "eurchf_fx_deriv_hf.xlsx"
    base_cur = filename[:3]
    counter_cur = filename[3:6]

    # read in: contracts ------------------------------------------------------
    tau_str = ["1W", "2W", "3W", "1M", "2M", "3M", "4M", "6M", "9M",
        "1Y", "18M", "2Y"]
    # create Panel: items are horizons, major_axis is datetime, minor_axis is
    #   contracts
    deriv_panel = pd.Panel(items=tau_str,
        minor_axis=["rr10d", "rr25d", "bf10d", "bf25d", "atm"])
    # loop over maturities
    for tau in tau_str:
        # tau = "1W"
        # read in spreadsheet
        deriv = pd.read_excel(
            io=data_path+filename,
            sheetname=tau,
            skiprows=9,
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
        deriv.index.name = "datetime"

        # volatility from percentage to fractions of 1
        deriv = deriv/100

        # store in Panel: reindex the latter, then append to it
        deriv_panel = deriv_panel.reindex(
            major_axis=deriv_panel.major_axis.union(deriv.index))
        deriv_panel.loc[tau,deriv.index,:] = deriv

    # deriv_panel.dropna(axis="items", how="any", inplace=True)
    # deriv_panel.loc["1M",:,:]

    # read in: S,F ------------------------------------------------------------
    sf = pd.read_excel(
        io=data_path+filename,
        sheetname="S,F",
        skiprows=9,
        header=None)

    sf = [sf.ix[:,(p*2):(p*2+1)].dropna() for p in range(len(tau_str)+1)]

    for p in range(len(tau_str)+1):
        sf[p].index = sf[p].pop(p*2)

    sf = pd.concat(sf, axis=1, ignore_index=True)

    # rename
    sf.columns = ["s",] + tau_str
    sf.index.name = "datetime"

    # get from forward poitns to forward prices
    sf.loc[:,"1W":] /= 10000
    sf.loc[:,"1W":] = sf.loc[:,"1W":].add(sf["s"], axis="index")

    # # drop na
    # sf.dropna(axis="index", how="any")

    # read in: rf -------------------------------------------------------------
    # converters for dates (experimental)
    # converters = {}
    # for p in range(len(tau_str)):
    #     converters[p*2] = lambda x: pd.to_datetime(x)
    # base currency
    rf_base = pd.read_excel(
        io=data_path+filename,
        sheetname="RF_BASE",
        skiprows=9,
        header=None)

    rf_base = [rf_base.ix[:,(p*2):(p*2+1)].dropna() \
        for p in range(len(tau_str))]

    for p in range(len(tau_str)):
        rf_base[p].index = rf_base[p].pop(p*2)

    rf_base = pd.concat(rf_base, axis=1, ignore_index=True)

    # rename
    rf_base.columns = tau_str
    rf_base.index.name = "datetime"

    # to fractions if 1
    rf_base = rf_base/100

    # make it constant over the whole day
    rf_base = rf_base.reindex(index=deriv_panel.major_axis,
        method="ffill")

    # counter currency
    rf_counter = pd.read_excel(
        io=data_path+filename,
        sheetname="RF_COUNTER",
        skiprows=9,
        header=None)

    rf_counter = [rf_counter.ix[:,(p*2):(p*2+1)].dropna() \
        for p in range(len(tau_str))]

    for p in range(len(tau_str)):
        rf_counter[p].index = rf_counter[p].pop(p*2)

    rf_counter = pd.concat(rf_counter, axis=1, ignore_index=True)

    # rename
    rf_counter.columns = tau_str
    rf_counter.index.name = "datetime"

    # to fractions of 1
    rf_counter = rf_counter/100

    # make it constant over the whole day
    rf_counter = rf_counter.reindex(index=deriv_panel.major_axis,
        method="ffill")

    # merge everything ------------------------------------------------------
    deriv_panel.loc[:,:,"f"] = sf.loc[:,"1W":]
    s = sf.loc[:,"s"].reindex(deriv_panel.major_axis)
    deriv_panel.loc[:,:,"rf_base"] = rf_base
    deriv_panel.loc[:,:,"rf_counter"] = rf_counter

    # # drop na -------------------------------------------------------------
    # deriv_panel.dropna(axis="major_axis", how="any", inplace=True)

    # store in HDF
    hangar = pd.HDFStore(data_path+base_cur+counter_cur+"_panel.h5", mode='w')
    hangar.put(key="panel", value=deriv_panel, format="table")
    hangar.put(key="temp", value=pd.Series(index=deriv_panel.major_axis))
    hangar.put(key="s", value=s, format="table")
    hangar.close()

    return deriv_panel.major_axis
