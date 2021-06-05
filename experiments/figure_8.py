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
    df, features = data_loader.get_dataset('data/darshan_theta_2017_2020.csv', 'POSIX', min_job_volume=0)

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


def xgboost_prediction(X_train, y_train, X_test, X_2020, **kwargs):
    """
    Train GBMs to predict y from X.
    Use obj_function during training, and test_error_function for the test evaluation.
    """
    regressor = xgb.XGBRegressor(obj=huber_approx_obj, **kwargs)
    regressor.fit(X_train, y_train, eval_metric=huber_approx_obj)
    y_pred_train = regressor.predict(X_train)
    y_pred_test  = regressor.predict(X_test)
    y_pred_2020  = regressor.predict(X_2020)

    return y_pred_train, y_pred_test, y_pred_2020


def normalize_dataset(X, feature_norm):
    abs_features = [c for c in X.columns if "LOG10" in c and c != feature_norm and c != "POSIX_AGG_PERF_BY_SLOWEST_LOG10"]
    # We subtract the log10 value of NPROCS from other log values, since this 
    # is equivalent to dividing before the logarithm
    X[abs_features] = X[abs_features].values - np.array(X[feature_norm]).reshape(-1, 1)

    return X 


def compare_methods():
    def get_results(feature_norm, **conf):
        target = "POSIX_AGG_PERF_BY_SLOWEST_LOG10"
        df, features = load_dataset()
        features = [f for f in features if f != 'RUNTIME_LOG10' and f != target] # baseline

        df_2020 = df[df.START_TIME >= 1577836800]
        df_train, df_test = sklearn.model_selection.train_test_split(df[df.START_TIME < 1577836800], test_size=0.2)
        del df

        X_train_bsln, X_test_bsln, X_2020_bsln = df_train[features], df_test[features], df_2020[features]
        y_train, y_test, y_2020 = df_train[target], df_test[target], df_2020[target]

        X_train_norm, X_test_norm, X_2020_norm =    \
            normalize_dataset(X_train_bsln.copy(), feature_norm), \
            normalize_dataset(X_test_bsln.copy(), feature_norm),  \
            normalize_dataset(X_2020_bsln.copy(), feature_norm)

        # Train models with and without normalized features 
        y_pred_train_norm, y_pred_test_norm, y_pred_2020_norm = \
            xgboost_prediction(X_train_norm, y_train, X_test_norm, X_2020_norm, **conf)
        y_pred_train_bsln, y_pred_test_bsln, y_pred_2020_bsln = \
            xgboost_prediction(X_train_bsln, y_train, X_test_bsln, X_2020_bsln, **conf)

        results = pd.DataFrame({
            'error' : np.abs(np.concatenate([
                y_train - y_pred_train_bsln,
                y_train - y_pred_train_norm,
                y_test - y_pred_test_bsln,
                y_test - y_pred_test_norm,
                y_2020 - y_pred_2020_bsln,
                y_2020 - y_pred_2020_norm,
            ])),
            'set' : ['train'] * y_train.shape[0] * 2 + ['test'] * y_test.shape[0] * 2 + ['2020'] * y_2020.shape[0] * 2,
            'type': ['baseline'] * y_train.shape[0] + ['normalized'] * y_train.shape[0] + ['baseline'] * y_test.shape[0] + ['normalized'] * y_test.shape[0] + ['baseline'] * y_2020.shape[0] + ['normalized'] * y_2020.shape[0],
        })

        return results

    conf = {"n_estimators": 512, "max_depth": 7, "colsample_bytree": 0.8}
    result = get_results("POSIX_TOTAL_BYTES_LOG10", **conf)

    def plot_violins(df):
        df = df[df.set != "train"]
        df = df[df.error < np.log10(10)]

        dx  = 'type'
        dy  = 'error'
        pal = "tab10"
        ort = 'v'
        
        #
        # First graph
        #
        plt.figure(figsize=(1.65 * 2, 1.4))
        plt.subplots_adjust(wspace=0.05, left=0, right=1)

        plt.subplot(121)

        ax=pt.half_violinplot(x=dx, y=dy, data=df[df.set =='test'], palette=pal, bw=.1, cut=0.,
            scale="width", width=1., inner=None, orient=ort, linewidth=0.8, offset=0.2)

        ax=sns.stripplot(x=dx, y=dy, data=df[df.set =='test'].sample(2000), palette=pal, 
            edgecolor="white", size=1, jitter=1, zorder=1, orient=ort, alpha=0.5)
        
        ax=sns.boxplot(x=dx, y=dy, data=df[df.set =='test'], color="black", width=.2, zorder=10,
            showcaps=True, boxprops={'facecolor': 'none', "zorder": 10}, 
            showfliers=True, whiskerprops={'linewidth': 1, "zorder": 10}, 
            saturation=1, orient=ort, fliersize=0, linewidth=1)

        plt.xlim(-0.8, 1.3)
        ax.set_xticklabels([
            "Baseline\nMedian={:.2f}$\\times$".format(10**df[(df.set=='test')&(df.type=='baseline')].error.median()), 
            "Normalized\nMedian={:.2f}$\\times$".format(10**df[(df.set=='test')&(df.type=='normalized')].error.median())
            ])
        ax.set_xlabel("")

        yticks = [1, 1.2, 1.5, 2, 5, 10]
        ax.set_yticks(np.log10(yticks))
        ax.set_yticklabels([r"{:.2f} $\times$".format(y) for y in yticks])
        plt.ylabel("Absolute Error")

        ax.set_title("Test set")

        #
        # Second graph
        #
        plt.subplot(122)

        ax = pt.half_violinplot(x=dx, y=dy, data=df[df.set == '2020'], palette=pal, bw=.1, cut=0.,
            scale="width", width=1., inner=None, orient=ort, linewidth=0.8, offset=0.2)

        ax=sns.stripplot(x=dx, y=dy, data=df[df.set == '2020'].sample(2000), palette=pal, 
            edgecolor="white", size=1, jitter=1, zorder=1, orient=ort, alpha=0.5)
        
        ax=sns.boxplot(x=dx, y=dy, data=df[df.set == '2020'], color="black", width=.2, zorder=10,
            showcaps=True, boxprops={'facecolor': 'none', "zorder": 10}, 
            showfliers=True, whiskerprops={'linewidth': 1, "zorder": 10}, 
            saturation=1, orient=ort, fliersize=0, linewidth=1)

        ax.set_xticklabels([
            "Baseline\nMedian={:.2f}$\\times$".format(10**df[(df.set=='2020')&(df.type=='baseline')].error.median()), 
            "Normalized\nMedian={:.2f}$\\times$".format(10**df[(df.set=='2020')&(df.type=='normalized')].error.median())
            ])
        ax.set_xlabel("")
        plt.xlim(-0.8, 1.3)

        ax.set_yticks(np.log10(yticks))
        ax.set_yticklabels(["" for y in yticks])
        plt.ylabel("")

        ax.set_title("2020 set")

        plt.savefig("figures/figure_8.pdf", dpi=600, bbox_inches='tight')

    plot_violins(result)


if __name__ == "__main__":
    compare_methods()
