# foolbox
import numpy as np
from scipy import signal as sig
import pandas as pd
from pandas.tseries.offsets import *

# my functions
from foolbox import tables_and_figures as taf

# for plotting
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
my_palette = ["#bf4905", "#2c53a5"]
my_cmap = LinearSegmentedColormap.from_list("my_cmap", my_palette)

# ---------------------------------------------------------------------------
def explain_carry(ax, p_carry, p_explanatory, scale=12, cmap=my_cmap):
    """
    """
    # concatenate both together
    together = pd.concat((p_explanatory, p_carry), axis=1, join="inner")
    together.dropna(how="any", inplace=True)

    # prepend with zero for visual appeal
    together = pd.concat((
        pd.DataFrame(
            data=np.array([[0.0, 0.0]]),
            columns=together.columns,
            index=[min(together.index) - DateOffset(months=1) + MonthEnd(),]),
        together), axis=0)

    # quick ap test
    ap_res = taf.ts_ap_tests(p_carry.to_frame(), p_explanatory.to_frame(),
        scale=scale)
    # r-sq (adjusted)
    R2 = ap_res.loc["carry", "adj_r_sq"]

    # plot
    # fig, ax = plt.subplots(figsize=(8,8*3/4))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation="horizontal",
      ha="center")
    together.cumsum().plot(ax=ax, linewidth=1.5, cmap=cmap)
    ax.grid(axis="both", which="major", alpha=0.75)
    ax.legend_.remove()
    # # add text
    # plt.text(
    #     x=min(together.index) + DateOffset(months=6),
    #     y=(ax.get_ylim()[1]-ax.get_ylim()[0])*3/5,
    #     s="$R^2$={:3.2f}".format(R2),
    #     fontsize=14, color=my_palette[0])

    plt.show()

    return(ap_res, ax)

def many_subplots(df1, df2):
    """ Draw many time-series plots from intertwined columns of df1 and df2.
    """
    # number of subplots
    N = len(df1.columns)

    # dra figure
    fig, ax = plt.subplots(N, sharex=True, sharey=True, figsize=(8.27,11.69))

    # set limits to the very last one (lowest) subplot
    ax[N-1].set_xlim((df1.first_valid_index() - DateOffset(months=3),
        df1.last_valid_index() + DateOffset(months=1)))
    ax[N-1].set_ylim([-2, 4])

    # set tick location and text appearance via Locator and DateFormatter
    # major ticks are 1 years apart, minor - 3 months
    ax[N-1].xaxis.set_major_locator(mdates.YearLocator(1))
    ax[N-1].xaxis.set_ticks(pd.date_range(
        "2010-12-31", "2016-01-01", freq="12M")+DateOffset(days=1),
        minor=False)
    ax[N-1].xaxis.set_minor_locator(mdates.MonthLocator(bymonth=range(1,13,6)))
    ax[N-1].tick_params(axis='x', which='major', labelsize=10)

    # plot 8 pieces
    for p in range(N):
        df1.ix[:,p].plot(ax=ax[p], x_compat=True, color=my_palette[1])
        df2.ix[:,p].plot(ax=ax[p], x_compat=True, color=my_palette[0],
            linestyle='-')
        ax[p].yaxis.set_ticks([-1,0,1,2,3])
        ax[p].yaxis.set_major_locator(ticker.FixedLocator([0,]))
        ax[p].yaxis.set_minor_locator(ticker.FixedLocator([-1,1,2,3]))
        ax[p].tick_params(axis='y', which='both', labelsize=10)
        ax[p].set_yticklabels([-1,1,2,3], minor=True)
        # ax[p].grid(True)
        ax[p].grid(axis='y', which="major", alpha=1)
        ax[p].grid(axis='x', which="both", alpha=0.25)
        ax[p].set_title(df1.columns[p], y=0.97, fontsize=10)

    # tick settings
    plt.setp(ax[N-1].xaxis.get_majorticklabels(), rotation="horizontal",
        ha="center")

    # save
    plt.show()

    return(fig)
