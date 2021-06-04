"""
Compares the I/O throughput of all jobs with their prediction error.
Colors jobs from different applications differently
"""
import os
import numpy as np
import pandas as pd
import sklearn
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from scipy.stats import *

from utils import data_loader, test_set_utils


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
    df.reset_index(inplace=True)
    df.drop(columns=['index', 'level_0'], inplace=True)

    for f, duplicate_set in df.groupby(features):
        group_size = duplicate_set.shape[0]
        sum_throughput = duplicate_set.POSIX_AGG_PERF_BY_SLOWEST_LOG10.sum()
        for idx, row in duplicate_set.iterrows(): 
            df.iloc[idx, -1] = (sum_throughput - row.POSIX_AGG_PERF_BY_SLOWEST_LOG10) / (group_size - 1)

    return df


def huber_approx_obj(y_pred, y_test):
    """
    Huber loss, adapted from https://stackoverflow.com/questions/45006341/xgboost-how-to-use-mae-as-objective-function
    """
    d = y_pred - y_test 
    h = 5  # h is delta in the graphic
    scale = 1 + (d / h) ** 2
    scale_sqrt = np.sqrt(scale)
    grad = d / scale_sqrt
    hess = 1 / scale / scale_sqrt
    return grad, hess


def get_full_data():
    df, features = data_loader.get_dataset("data/darshan_theta_2017_2020.csv", "POSIX")

    df.reset_index(inplace=True)
    df.drop(columns=['index', 'level_0'], inplace=True)

    X_train, X_test, y_train, y_test = test_set_utils.random_split(df, "POSIX_AGG_PERF_BY_SLOWEST_LOG10", keep_columns=features, test_size=0.3)

    regressor = xgb.XGBRegressor(obj=huber_approx_obj, n_estimators=2**11, max_depth=7, colsample_bytree=0.8, subsample=1)
    regressor.fit(X_train, y_train, eval_metric=huber_approx_obj)
    y_pred_test = regressor.predict(X_test)

    df = pd.DataFrame({
        'POSIX_AGG_PERF_BY_SLOWEST_LOG10': y_test,
        'prediction': y_pred_test
    })

    return df


def duplicates():
    df = get_duplicate_data()
    df['error'] = df["POSIX_AGG_PERF_BY_SLOWEST_LOG10"] - df["prediction"]

    joint_kws=dict(gridsize=50, linewidths=0, mincnt=1, vmax=800)
    marginal_kws=dict(bins=50)

    sns.jointplot(
        data=df[(df.error < 0.25) & (df.error > -0.25)], 
        x="POSIX_AGG_PERF_BY_SLOWEST_LOG10", y='error',
        kind='hex', 
        joint_kws=joint_kws, marginal_kws=marginal_kws,
        cmap='Spectral_r',
        height=1.65)

    plt.xlabel("I/O throughput")
    plt.xticks([0, 3, 6], ["MiB/s", "GiB/s", "TiB/s"])

    plt.ylabel("Mean predictor error", fontsize=8)
    plt.yticks(np.log10([1/1.5, 1/1.2, 1, 1.2, 1.5]), ["$.67\\times$", "$.83\\times$", "$1\\times$", "$1.2\\times$", "$1.5\\times$"])
    plt.savefig("figures/figure_3a.pdf", dpi=600, bbox_inches='tight', pad_inches=0)


def full():
    df = get_full_data()
    df['error'] = df["POSIX_AGG_PERF_BY_SLOWEST_LOG10"] - df["prediction"]

    joint_kws=dict(gridsize=50, linewidths=0, mincnt=1, vmax=800)
    marginal_kws=dict(bins=50)

    sns.jointplot(
        data=df[(df.error < 0.25) & (df.error > -0.25)], 
        x="POSIX_AGG_PERF_BY_SLOWEST_LOG10", y='error',
        kind='hex', 
        joint_kws=joint_kws, marginal_kws=marginal_kws,
        cmap='Spectral_r',
        height=1.65)

    plt.xlabel("I/O throughput")
    plt.xticks([0, 3, 6], ["MiB/s", "GiB/s", "TiB/s"])

    plt.ylabel("XGBoost predictor error", fontsize=8)
    plt.yticks(np.log10([1/1.5, 1/1.2, 1, 1.2, 1.5]), ["$.67\\times$", "$.83\\times$", "$1\\times$", "$1.2\\times$", "$1.5\\times$"])

    plt.savefig("figures/figure_3b.pdf", dpi=600, bbox_inches='tight', pad_inches=0)


if __name__ == "__main__":
    duplicates()
    full()
