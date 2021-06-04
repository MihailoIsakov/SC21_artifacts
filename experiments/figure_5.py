import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import joypy

from utils import data_loader


# Figure size in inches, font 
sns.set_style('dark')

# Font 
fontpath = os.path.expanduser('/usr/share/fonts/opentype/linux-libertine/LinLibertine_R.otf')
prop = matplotlib.font_manager.FontProperties(fname=fontpath)
matplotlib.rcParams['font.family'] = prop.get_name()
matplotlib.rc('font', family=prop.get_name())
plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)
plt.rc('axes', labelsize=8)
plt.rc('axes', titlesize=10)
plt.rc('ytick.major', pad=0)
plt.rc('xtick.major', pad=0)
plt.rc('ytick.major', pad=0)
plt.rc('axes', labelpad=2)
plt.rc('axes', titlepad=2)
plt.rc('axes', grid=True)


def get_duplicate_data():
    df, features = data_loader.get_dataset("data/darshan_theta_2017_2020.csv", "POSIX")

    df = df[df.duplicated(features, keep=False)]
    df['prediction'] = -1
    df['time_diff'] = -1
    df.reset_index(inplace=True)
    df.drop(columns=['index', 'level_0'], inplace=True)

    for f, duplicate_set in df.groupby(features):
        group_size = duplicate_set.shape[0]
        sum_throughput = duplicate_set.POSIX_AGG_PERF_BY_SLOWEST_LOG10.sum()
        sum_time       = duplicate_set.START_TIME.sum()
        for idx, row in duplicate_set.iterrows(): 
            df.iloc[idx, -2] = (sum_throughput - row.POSIX_AGG_PERF_BY_SLOWEST_LOG10) / (group_size - 1)
            df.iloc[idx, -1] = np.abs((sum_time - row.START_TIME) / (group_size - 1) - row.START_TIME)

    return df


def duplicates():
    df = get_duplicate_data()
    df['error'] = df["POSIX_AGG_PERF_BY_SLOWEST_LOG10"] - df["prediction"]

    df = df[np.abs(df.error) < np.log10(1.5)]
    df.time_diff = np.log10(df.time_diff + 0.01)
    # df.error = np.abs(df.error)

    cuts = [-np.inf] + list(range(9))
    groups = [df[(df.time_diff >= low) & (df.time_diff < high)].error for low, high in zip(cuts[:-1], cuts[1:])]

    # fit a student t distribution 
    from scipy.stats import t
    param = t.fit(groups[0])
    norm_gen_data = t.rvs(param[0], param[1], param[2], 10000)

    groups = list(reversed([norm_gen_data] + groups))
    labels = list(reversed(["t-distribution fit", "0s to 1s"] + ["$10^{}s$ to $10^{}s$".format(low, high) for low, high in zip(cuts[1:-1], cuts[2:])]))

    fig, axes = joypy.joyplot(groups,
            colormap=matplotlib.cm.coolwarm_r, overlap=0.3, linewidth=1., 
            ylim='own', range_style='own', tails=0.2, bins=100, 
            labels=labels, figsize=(2.5, 3))

    for idx, ax in enumerate(axes): 
        try: 
            ax.set_yticklabels([labels[idx]], fontsize=8, rotation=120)
        except: pass
        ax.set_xlim([-0.2, 0.2])
        ax.set_xticks(np.log10([1/1.5, 1/1.2, 1, 1.2, 1.5]))
        ax.set_xticklabels(["$.67\\times$", "$.83\\times$", "$1\\times$", "$1.2\\times$", "$1.5\\times$"], rotation=90, fontsize=8)

    plt.xlabel("Error", rotation=180)
    plt.ylabel("Time ranges")

    plt.savefig("figures/figure_5.pdf", dpi=600, bbox_inches='tight', pad_inches=0)


if __name__ == "__main__":
    duplicates()
