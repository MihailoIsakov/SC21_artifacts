import os
import numpy as np
import pandas as pd
import xgboost as xgb
import sklearn
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
plt.rc('axes', labelpad=2)
plt.rc('axes', titlepad=2)
plt.rc('axes', grid=True)


def prediction_results(X_train, y_train, X_test, y_test):
    """
    Train GBMs to predict y from X.
    Use obj_function during training, and test_error_function for the test evaluation.
    """
    regressor = xgb.XGBRegressor()
    regressor.fit(X_train, y_train)
    y_pred_train = regressor.predict(X_train)
    y_pred_test  = regressor.predict(X_test)

    return y_pred_train, y_pred_test


def predict(split_time):
    df, columns = data_loader.get_dataset(
        'data/darshan_theta_2017_2020.csv', 
        'POSIX', 
        min_job_volume=0
    )

    df_before = df[df.START_TIME <= split_time]
    df_after  = df[df.START_TIME >  split_time]

    X_train, X_test_before, y_train, y_test_before = \
        sklearn.model_selection.train_test_split(df_before[columns + ["START_TIME"]], df_before.POSIX_AGG_PERF_BY_SLOWEST_LOG10, test_size=0.3)

    timestamps_before, timestamps_after = X_test_before.START_TIME.to_numpy(), df_after.START_TIME.to_numpy()
    X_train, X_test_before = X_train[columns], X_test_before[columns]

    X_test_after, y_test_after = df_after[columns], df_after.POSIX_AGG_PERF_BY_SLOWEST_LOG10

    X_train, X_test_before, X_test_after, y_train, y_test_before, y_test_after = \
        X_train.to_numpy(), X_test_before.to_numpy(), X_test_after.to_numpy(), y_train.to_numpy(), y_test_before.to_numpy(), y_test_after.to_numpy()

    len_before = len(y_test_before)

    y_pred_train, y_pred_test = prediction_results(X_train, y_train, np.concatenate((X_test_before, X_test_after)), np.concatenate((y_test_before, y_test_after)))
    y_pred_test_before, y_pred_test_after = y_pred_test[:len_before], y_pred_test[len_before:]

    return timestamps_before, y_test_before, y_pred_test_before, timestamps_after, y_test_after, y_pred_test_after


def main(split_time):
    timestamps_before, y_test_before, y_pred_test_before, timestamps_after, y_test_after, y_pred_test_after = predict(split_time)

    df = pd.DataFrame({
        'timestamp': np.concatenate([timestamps_before, timestamps_after]),
        'week':      pd.to_datetime(np.concatenate([timestamps_before, timestamps_after]), unit='s').to_period('W'),
        'error': np.abs(np.concatenate([y_test_before-y_pred_test_before, y_test_after-y_pred_test_after])),
        'test': np.array([False] * timestamps_before.shape[0] + [True] * timestamps_after.shape[0])
    })


    plt.figure(figsize=(1.65, 1.65))

    temp_df = df[df.error < 1]
    plt.hexbin(temp_df.timestamp.astype(int), temp_df.error, gridsize=50, bins='log', cmap="Spectral_r", mincnt=0, linewidths=0.01, alpha=0.9)
    plt.colorbar()

    x = df[df.test==False].groupby('week').timestamp.mean()
    y = df[df.test==False].groupby('week').error.median()
    plt.plot(x, y, color='green', linewidth=1)

    x = df[df.test==True].groupby('week').timestamp.mean()
    y = df[df.test==True].groupby('week').error.median()
    ax = plt.plot(x, y, color='red', linewidth=1)

    plt.ylim(0, 1)
    plt.yticks(np.log10([1, 2, 3, 5, 7, 9]), ["1$\\times$", "2$\\times$", "3$\\times$", "5$\\times$", "7$\\times$", "9$\\times$"])
    plt.ylabel("Error")

    plt.xticks([1483228800, 1514764800, 1546300800, 1577836800], ["2017/1/1", "2018/1/1", "2019/1/1", "2020/1/1"], rotation=30)
    plt.tick_params(axis='x', direction='inout', length=10)

    plt.savefig("figures/figure_1d.pdf", dpi=600, bbox_inches='tight')


if __name__ == "__main__":
    jan_2020  = 1577836800

    main(jan_2020)




