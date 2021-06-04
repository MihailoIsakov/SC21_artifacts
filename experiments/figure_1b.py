"""
Compares the I/O throughput of all jobs with their prediction error.
Colors jobs from different applications differently
"""
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from scipy.stats import *
from collections import Counter
import matplotlib.ticker as ticker

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



def calculate_duplicate_errors():
    """
    Get all duplicates, take their mean, and predict the throughput
    Returns the real (target) throughputs and relative prediction errors.

    Returns: 
        a list of target throughputs and a list of relative errors
    """
    df, features = data_loader.get_dataset("data/darshan_theta_2017_2020.csv", "POSIX")

    duplicated = df.duplicated(features, keep=False)
    apps       = df[duplicated]["apps_short"]

    regressor = xgb.XGBRegressor(n_estimators=4000, depth=8)
    regressor.fit(df[duplicated][features], df[duplicated]['POSIX_AGG_PERF_BY_SLOWEST_LOG10'])
    y_pred = regressor.predict(df[duplicated][features])

    return df[duplicated]['POSIX_AGG_PERF_BY_SLOWEST_LOG10'], y_pred, apps


def sample_apps_equally(df, sample):
    """
    Given multiple applications, makes sure each has equal representation
    """
    apps = set(df.app)

    dfs = []
    for app in apps:
        dfs.append(df[df.app == app].sample(sample))

    return pd.concat(dfs)


def plot_violins(df, count):
    """
    Given I/O throughputs and predictions, plots violins of errors for different applications.
    """
    top_apps = [c[0] for c in Counter(df.app).most_common()[:count]]
    df = df[df.app.isin(top_apps)]

    dx = 'app' 
    dy = 'error'
    pal = "tab10"
    ort = "v"

    plt.figure(figsize=(1.65, 1.65))

    import ptitprince as pt
    ax=pt.half_violinplot(x=dx, y=dy, data=df, palette=pal, bw=.1, cut=0.,
        scale="width", width=1., inner=None, orient=ort, linewidth=0.8, offset=0.2)
    ax=sns.stripplot(x=dx, y=dy, data=sample_apps_equally(df, 1000), palette=pal, edgecolor="white",
        size=1, jitter=1, zorder=1, orient=ort, alpha=0.5)
    ax=sns.boxplot(x=dx, y=dy, data=df, color="black", width=.2, zorder=10,
        showcaps=True, boxprops={'facecolor': 'none', "zorder": 10}, 
        showfliers=True, whiskerprops={'linewidth': 1, "zorder": 10}, 
        saturation=1, orient=ort, fliersize=0, linewidth=1)

    yticks = [1/2, 1/1.5, 1/1.2, 1, 1.2, 1.5, 2]
    ax.set_yticks(np.log10(yticks))
    ax.set_yticklabels([r"{:.2f} $\times$".format(y) for y in yticks])
    ax.set_xticklabels(["Writer", "pw.x", "HACC", "IOR", "QB"], rotation=30)
    ax.set_xlabel("Application")
    ax.set_ylabel("Error")
    ax.set_axisbelow(True)

    plt.xlim(-0.8, 4.3)
    plt.ylim(np.log10(1/2), np.log10(2))

    plt.savefig("figures/figure_1b.pdf", dpi=600, bbox_inches='tight')


def main():
    throughputs, predictions, apps = calculate_duplicate_errors()
    errors = throughputs - predictions

    df = pd.DataFrame({'throughput': throughputs, 'error': errors, 'app': apps})

    plot_violins(df, 5)


if __name__ == "__main__":
    main()
