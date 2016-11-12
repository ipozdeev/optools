# plots of correlation
import numpy as np
import pandas as pd
import h5py
# import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from cycler import cycler
from pandas import DateOffset
import matplotlib.dates as mdates
from matplotlib import animation, ticker
import misc
# %matplotlib inline

usr = "hsg-spezial"
path = "c:/users/" + usr + "/google drive/" + \
    "personal/option_implied_betas_project/"

my_palette = ["#d60000", "#d66300", "#00a31b", "#1092b2",
    "#1030b2", "#8f04f2", "#ad00a4", "#540409"]
my_cycler = cycler("color", my_palette)
my_cmap = LinearSegmentedColormap.from_list("my_cmap", my_palette)

if __name__ == "__main__":

    tau_str = "1m"
    opt_meth = "diff"

    # load hdf dataset
    hangar = h5py.File(path + \
        "data/estimates/betas_and_returns_"+tau_str+"_"+opt_meth+".h5",
        mode='r')

    # # animated correlation --------------------------------------------------
    # # from hdf to pd.Panel
    # cormat = misc.hdf_to_pandas(hangar["correlations"])
    # cormat = cormat.loc[:,sorted(cormat.major_axis),sorted(cormat.major_axis)]
    # hangar.close()
    # # smooth panel along time dimension
    # cormat_smoothed = misc.panel_rolling(data=cormat, fun=np.nanmean,
    #     window=5, min_periods=3)
    #
    # # dimensions: needed for allocation of storage space later
    # T,N,_ = cormat_smoothed.shape
    #
    # # color palette (diverging, centered on white)
    # cmap = sns.diverging_palette(240, 10, s=85, l=40,
    #     n=11, center='light', as_cmap=True)
    #
    # # start figure
    # fig, ax = plt.subplots(figsize=(5, 5*3/4))
    # # enlarge tick labels
    # ax.tick_params(labelsize=9)
    #
    # def animate(t):
    #     plt.clf()
    #     data = cormat_smoothed.loc[t]
    #     heatm = sns.heatmap(data, vmax=1., vmin=0.,
    #         square=True, linewidths=.5, cmap="hot_r")
    #     heatm.set_title(str(t.date()), fontsize=12)
    #
    # anim = animation.FuncAnimation(fig, animate,
    #     frames=list(cormat_smoothed.items), repeat=True, interval=50)
    # anim.save(path+"pics/correlation/anim_5_day_"+opt_meth+".gif", dpi=80,
    #     writer="imagemagick")
    # # anim.save(path+"pics/correlation/anim_5_day.mp4", dpi=150,
    # #     writer="ffmpeg")

    # # betas, time-series plot -----------------------------------------------
    # b_impl = misc.hdf_to_pandas(hangar["betas_eq"]).rolling(
    #     window=5, min_periods=3).mean()
    # hangar.close()
    #
    # plt.rc('axes', prop_cycle=my_cycler)
    # fig, ax = plt.subplots(2, sharex=True, sharey=True, figsize=(7,7/4*3))
    # b_impl.ix[:,0::2].plot(ax=ax[0], x_compat=True, cmap=my_cmap)
    # b_impl.ix[:,1::2].plot(ax=ax[1], x_compat=True, cmap=my_cmap)
    # # set limits
    # ax[1].set_xlim((b_impl.first_valid_index() - DateOffset(months=3),
    #     b_impl.last_valid_index() + DateOffset(months=1)))
    # ax[1].set_ylim([-2, 4])
    #
    # # set tick location and text appearance via Locator and DateFormatter
    # # major ticks are 1 years apart, minor - 3 months
    # ax[1].xaxis.set_major_locator(mdates.YearLocator(1))
    # ax[1].xaxis.set_ticks(pd.date_range(
    #     "2010-12-31", "2016-01-01", freq="12M")+DateOffset(days=1),
    #     minor=False)
    # ax[1].xaxis.set_minor_locator(mdates.MonthLocator(bymonth=range(1,13,6)))
    #
    # # tick settings
    # plt.setp(ax[1].xaxis.get_majorticklabels(), rotation="vertical",
    #     ha="center")
    #
    # # grid
    # ax[1].grid(which="minor")
    # ax[0].grid(which="minor")
    #
    # # set legend located in the upper-left corner, with small fontsize
    # ax[1].legend(loc=2, fontsize="small")
    # ax[0].legend(loc=2, fontsize="small")
    #
    # # save
    # plt.show()
    # fig.savefig(path+"pics/betas_eq_"+tau_str+"_"+opt_meth+".pdf")
    # #

    # betas, 8 time-series plot ---------------------------------------------
    b_impl = misc.hdf_to_pandas(hangar["b_impl_eq"]).rolling(
        window=5, min_periods=3).mean()
    b_roll = misc.hdf_to_pandas(hangar["b_roll"])
    hangar.close()

    b_impl = b_impl.sort(axis=1)
    N = len(b_impl.columns)

    fig, ax = plt.subplots(8, sharex=True, sharey=True,
        figsize=(8.27,11.69))

    # set limits
    ax[N-1].set_xlim((b_impl.first_valid_index() - DateOffset(months=3),
        b_impl.last_valid_index() + DateOffset(months=1)))
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
        b_impl.ix[:,p].plot(ax=ax[p], x_compat=True, color='k')
        b_roll.ix[:,p].plot(ax=ax[p], x_compat=True, color=my_palette[0],
            linestyle='--')
        ax[p].yaxis.set_ticks([-1,0,1,2,3])
        ax[p].yaxis.set_major_locator(ticker.FixedLocator([0,]))
        ax[p].yaxis.set_minor_locator(ticker.FixedLocator([-1,1,2,3]))
        ax[p].tick_params(axis='y', which='both', labelsize=10)
        ax[p].set_yticklabels([-1,1,2,3], minor=True)
        # ax[p].grid(True)
        ax[p].grid(axis='y', which="major", alpha=1)
        ax[p].grid(axis='x', which="both", alpha=0.25)
        ax[p].set_title(b_impl.columns[p], y=0.97, fontsize=10)

    # tick settings
    plt.setp(ax[N-1].xaxis.get_majorticklabels(), rotation="horizontal",
        ha="center")

    # save
    plt.show()
    fig.savefig(path+"pics/betas_eq_"+tau_str+"_"+opt_meth+"_subplots"+".pdf",
        bbox_inches="tight")

    # 
