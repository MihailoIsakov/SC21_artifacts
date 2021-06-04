import os
import pandas as pd
import numpy as np
import sklearn
import xgboost as xgb
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


from utils import pipelines, data_loader


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


def grid_search(max_log2_trees, max_log2_depth):
    """
    Run a grid search over two parameters: tree depth and number of trees. For 
    each configuration, train an XGBoost regressor and evaluate its performance
    on the test set. Plot a matrix of the results.
    """
    df, features = data_loader.get_dataset(
        'data/darshan_theta_2017_2020.csv', 
        'POSIX', 
        min_job_volume=0
    )

    # Don't include runtime in the set of input features
    features = [f for f in features if f != 'RUNTIME_LOG10'] 

    df_train, df_test = sklearn.model_selection.train_test_split(df, test_size=0.2)

    X_train, X_test = df_train[features], df_test[features]
    # POSIX_AGG_PERF_BY_SLOWEST_LOG10 is log10 of Darshan's I/O throughput estimate
    y_train, y_test = df_train["POSIX_AGG_PERF_BY_SLOWEST_LOG10" ], df_test["POSIX_AGG_PERF_BY_SLOWEST_LOG10" ]

    results = {"depth": [], "trees": [], "error": []}

    def evaluate_configuration(depth, trees):
        regressor = xgb.XGBRegressor(obj=huber_approx_obj, n_estimators=trees, max_log2_depth=depth)
        regressor.fit(X_train, y_train, eval_metric=huber_approx_obj)
        y_pred_test  = regressor.predict(X_test)

        error = np.median(10**np.abs(y_test - y_pred_test))
        return error
        

    for trees in [2**x for x in range(1, max_log2_trees+1)]:
        for depth in range(1, max_log2_depth+1):
            error = evaluate_configuration(depth, trees)
            print(f"Trees: {trees}, depth: {depth}, error: {error}")

            results['depth'].append(depth)
            results['trees'].append(trees)
            results['error'].append(error)

    return pd.DataFrame(results)


def plot_figure(results):
    plt.figure(figsize=(1.65, 1.65))

    hm = sns.heatmap(data=results, cmap='Spectral', annot=True, 
            linewidths=.5, fmt=".2f", vmax=1.18, annot_kws={'fontsize': 4}) 

    for t in hm.texts: 
        t.set_text(t.get_text())

    plt.xlabel("Number of estimators")
    plt.ylabel("Estimator depth")

    plt.savefig("figures/figure_1a.pdf", dpi=600, bbox_inches='tight')


def main(max_log2_trees, max_log2_depth):
    results = grid_search(max_log2_trees=max_log2_trees, max_log2_depth=max_log2_depth)
    # results = results[(results.trees > 4) & (results.depth > 3)]
    results = results.pivot("depth", "trees", "error")

    plot_figure(results)


if __name__ == "__main__":
    # main(max_log2_trees=10, max_log2_depth=18)
    main(max_log2_trees=4, max_log2_depth=4)



