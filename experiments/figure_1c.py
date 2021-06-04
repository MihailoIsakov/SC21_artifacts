import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

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


def main():
    df = get_duplicate_data()
    df['error'] = df["POSIX_AGG_PERF_BY_SLOWEST_LOG10"] - df["prediction"]

    df = df[np.abs(df.error) < np.log10(1.5)]
    df.time_diff = np.log10(df.time_diff + 0.01)
    df.error = np.abs(df.error)

    joint_kws=dict(linewidths=0, bins=50)
    marginal_kws=dict(bins=25)

    plot = sns.jointplot(
        data=df[(df.error < 0.25) & (df.error > -0.25)], 
        x="time_diff", y='error',
        kind='hist', 
        joint_kws=joint_kws, marginal_kws=marginal_kws,
        cmap='Spectral_r',
        height=1.65, 
        ratio=7)

    plot.ax_joint.set_xlabel("Relative time")
    plot.ax_joint.set_xticks(np.log10([0.01, 1, 60, 3600, 24*3600, 24*3600*365]))
    plot.ax_joint.set_xticklabels(["0s", "1s", "1m", "1h", "1d", "1y"])

    plt.ylabel("Mean predictor error", fontsize=8)
    plot.ax_joint.set_yticks(np.log10([1, 1.2, 1.5]))
    plot.ax_joint.set_yticklabels(["$1\\times$", "$1.2\\times$", "$1.5\\times$"])

    plot.ax_joint.set_xlabel("Relative time")
    plot.ax_joint.set_ylabel("Absolute error")

    plot.savefig("figures/figure_1c.pdf", dpi=600, bbox_inches='tight')


if __name__ == "__main__":
    main()
