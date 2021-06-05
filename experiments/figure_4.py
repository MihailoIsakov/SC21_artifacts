"""
Loads one of these datasets:
    1) POSIX
    2) POSIX + categorical features
    3) POSIX + runtime
    4) POSIX + nodes & node_hours
    5) MPIIO
    6) MPIIO + runtime
    7) MPIIO + nodes & node_hours

Splits the dataset into:
    1) Random
    2) DBSCAN - we need to use DBSCAN Fold for this
    3) 2020

"""
import os
import pandas as pd
import numpy as np
import sklearn
import xgboost as xgb
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import ptitprince as pt

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
plt.rc('pdf', fonttype=42)


def load_dataset():
    df, features = data_loader.get_dataset(
        'data/darshan_theta_2017_2020.csv', 
        'POSIX', 
        min_job_volume=0
    )

    return df, features


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


def xgboost_prediction(X_train, y_train, X_test, **kwargs):
    """
    Train GBMs to predict y from X.
    Use obj_function during training, and test_error_function for the test evaluation.
    """
    regressor = xgb.XGBRegressor(obj=huber_approx_obj, **kwargs)
    regressor.fit(X_train, y_train, eval_metric=huber_approx_obj)
    y_pred_train = regressor.predict(X_train)
    y_pred_test  = regressor.predict(X_test)

    return y_pred_train, y_pred_test


def compare_datasets():
    def get_results():
        df, features = load_dataset()
        # Prune bad and extra jobs 
        df = df.sample(100000)

        X_train, X_test = sklearn.model_selection.train_test_split(df, test_size=0.2)
        y_train, y_test = X_train["POSIX_AGG_PERF_BY_SLOWEST_LOG10"], X_test["POSIX_AGG_PERF_BY_SLOWEST_LOG10"]

        # Make features 
        fb = [f for f in features if 'RUNTIME_LOG10' != f] # baseline
        ft = fb + ["START_TIME", "END_TIME"]                                     # timestamps 

        train_pred_baseline,  test_pred_baseline  = xgboost_prediction(X_train[fb], y_train, X_test[fb], n_estimators=512, max_depth=15, colsample_bytree=0.8)
        train_pred_timestamp, test_pred_timestamp = xgboost_prediction(X_train[ft], y_train, X_test[ft], n_estimators=512, max_depth=15, colsample_bytree=0.8)

        return X_train, X_test, y_train, y_test, train_pred_baseline, test_pred_baseline, train_pred_timestamp, test_pred_timestamp 

    X_train, X_test, y_train, y_test, \
        train_pred_baseline, test_pred_baseline, \
        train_pred_timestamp, test_pred_timestamp = get_results()

    df_test = pd.DataFrame({
        'error': np.abs(np.concatenate([
            y_test - test_pred_baseline, 
            y_test - test_pred_timestamp,
        ])),
        'type' : ['baseline'] * y_test.shape[0] + ['timestamp'] * y_test.shape[0]
    })

    def plot_violins(df):
        plt.figure(figsize=(1.65 * 2, 1.4))

        dx  = 'type'
        dy  = 'error'
        pal = "tab10"
        ort = 'v'

        df = df[df.error < np.log10(3)]

        ax=pt.half_violinplot(x=dx, y=dy, data=df, palette=pal, bw=.1, cut=0.,
            scale="width", width=1., inner=None, orient=ort, linewidth=0.8, offset=0.2)

        ax=sns.stripplot(x=dx, y=dy, data=df.sample(2000), palette=pal, 
            edgecolor="white", size=1, jitter=1, zorder=1, orient=ort, alpha=0.5)
        
        ax=sns.boxplot(x=dx, y=dy, data=df, color="black", width=.2, zorder=10,
            showcaps=True, boxprops={'facecolor': 'none', "zorder": 10}, 
            showfliers=True, whiskerprops={'linewidth': 1, "zorder": 10}, 
            saturation=1, orient=ort, fliersize=0, linewidth=1)

        yticks = [1/3, 1/2, 1/1.5, 1/1.2, 1, 1.2, 1.5, 2, 3]
        ax.set_yticks(np.log10(yticks))
        ax.set_yticklabels([r"{:.2f} $\times$".format(y) for y in yticks])
        plt.ylim(-0.03, np.log10(3))
        plt.ylabel("Absolute Error")

        plt.xlim(-0.8, 3.3)
        plt.xticks([0, 1, 2, 3],
            ["POSIX Baseline\nMedian = $1.173\\times$", 
             "POSIX + start time\nMedian = $1.117\\times$"], rotation=30)
        plt.xlabel("")

        print("Baseline median absolute error: {}".format(10**df[df.type=='baseline'].error.median()))
        print("Timestamps median absolute error: {}".format(10**df[df.type=='timestamp'].error.median()))

        plt.savefig("figures/figure_4.pdf", dpi=600, bbox_inches='tight')

    def plot_change():
        plt.figure()
        sns.scatterplot(x=np.array(df[df.type == 'baseline'].error), y=np.array(df[df.type == 'timestamp'].error), hue=np.array(y_test))
        plt.show()

    plot_violins(df_test)


if __name__ == "__main__":
    compare_datasets()
