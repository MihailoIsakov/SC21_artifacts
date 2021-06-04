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


def load_dataset(module, remove_runtime):
    df, features = data_loader.get_dataset('data/darshan_theta_2017_2020.csv', module, min_job_volume=0)

    if module == "POSIX": 
        features.remove("POSIX_FDSYNCS_LOG10")

    if remove_runtime:
        features.remove("RUNTIME_LOG10")

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


def xgboost_prediction_error(X_train, y_train, X_test, y_test):
    """
    Train GBMs to predict y from X.
    Use obj_function during training, and test_error_function for the test evaluation.
    """
    conf = {"n_estimators": 512, "max_depth": 15, "colsample_bytree": 0.8}
    regressor = xgb.XGBRegressor(obj=huber_approx_obj, **conf)
    regressor.fit(X_train, y_train, eval_metric=huber_approx_obj)
    y_pred_train = regressor.predict(X_train)
    y_pred_test  = regressor.predict(X_test)

    return 10**np.abs(y_train - y_pred_train), 10**np.abs(y_test - y_pred_test)


def train_on_split(df, features):
    """
    Creates a random split and trains on it
    """
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(df[features], df["POSIX_AGG_PERF_BY_SLOWEST_LOG10"], test_size=0.2)

    training_error, test_error = xgboost_prediction_error(X_train, y_train, X_test, y_test)

    return training_error, test_error


def compare_datasets():
    def get_runtime_results():
        df, features = load_dataset(module="POSIX", remove_runtime=True)
        df_rt, features_rt = load_dataset(module="POSIX", remove_runtime=False)

        posix_train, posix_test = train_on_split(df, features)
        runtime_train, runtime_test = train_on_split(df_rt, features_rt)

        return posix_train, posix_test, runtime_train, runtime_test 

    def get_mpiio_results():
        _,        features_posix = load_dataset(module="POSIX", remove_runtime=True)
        _,        features_mpiio = load_dataset(module="MPIIO", remove_runtime=True)
        df_both,  features_both  = load_dataset(module="both",  remove_runtime=True)

        mpiio_train, mpiio_test = train_on_split(df_both, features_mpiio)
        both_train, both_test  = train_on_split(df_both, features_both)

        return mpiio_train, mpiio_test, both_train, both_test  

    def get_cobalt_results(multiple_allocations='ignore'):
        df, features = load_dataset(module="POSIX", remove_runtime=True)
        cobalt = pd.read_csv("data/cobalt_theta_2017_2020.csv")  

        features_cobalt = features + ["NODES_USED_LOG10", "USED_CORE_HOURS_LOG10"]

        if multiple_allocations == 'ignore':
            alloc_sizes = df.groupby(["JOBID"]).size()
            df = df[df.JOBID.isin(alloc_sizes[alloc_sizes == 1].index)]   
        df = pd.merge(df, cobalt, left_on=["JOBID"], right_on=["COBALT_JOBID"])

        df["NODES_USED_LOG10"]      = np.log10(df.NODES_USED)
        df["USED_CORE_HOURS_LOG10"] = np.log10(df.USED_CORE_HOURS)

        cobalt_train, cobalt_test = train_on_split(df, features_cobalt)

        return cobalt_train, cobalt_test

    posix_train, posix_test, runtime_train, runtime_test = get_runtime_results()
    mpiio_train, mpiio_test, both_train, both_test = get_mpiio_results()
    cobalt_train, cobalt_test = get_cobalt_results()


    # posix_train, posix_test, runtime_train, runtime_test, mpiio_train, mpiio_test, both_train, both_test, cobalt_train, cobalt_test

    results = pd.DataFrame({
        'error' : np.concatenate([
            posix_train, posix_test, 
            runtime_train, runtime_test, 
            mpiio_train, mpiio_test, 
            both_train, both_test, 
            cobalt_train, cobalt_test
        ]),
        'set' : ['train'] * posix_train.shape[0] +   ['test'] * posix_test.shape[0] + 
                ['train'] * runtime_train.shape[0] + ['test'] * runtime_test.shape[0] + 
                ['train'] * mpiio_train.shape[0] +   ['test'] * mpiio_test.shape[0] + 
                ['train'] * both_train.shape[0] +    ['test'] * both_test.shape[0] + 
                ['train'] * cobalt_train.shape[0] +  ['test'] * cobalt_test.shape[0],
        'type': ['posix'] * (posix_train.shape[0] + posix_test.shape[0]) + 
                ['runtime'] * (runtime_train.shape[0] + runtime_test.shape[0]) + 
                ['mpiio'] * (mpiio_train.shape[0] + mpiio_test.shape[0]) + 
                ['both'] *(both_train.shape[0] + both_test.shape[0]) + 
                ['cobalt'] *(cobalt_train.shape[0] + cobalt_test.shape[0])
    })

    # Problems with log axes make me have to modify the data 
    results.error = np.log10(results.error)
    
    #
    # Plotting
    #
    dx  = 'type'
    dy  = 'error'
    pal = "tab10"
    ort = 'v'
    
    df = results[results.error < np.log10(2)]

    plt.figure(figsize=(1.65 * 2, 2))

    def sample_type_equally(df, sample):
        """
        Given multiple types, makes sure each has equal representation
        """
        types = set(df.type)

        dfs = []
        for type in types:
            dfs.append(df[df.type == type].sample(sample))

        return pd.concat(dfs)
    #
    # Top figure 
    #
    ax = plt.subplot(211)

    pt.half_violinplot(x=dx, y=dy, data=df[df.set =='train'], palette=pal, bw=.1, cut=0.,
     scale="width", width=1., inner=None, orient=ort, linewidth=0.8, offset=0.2)

    sns.stripplot(x=dx, y=dy, data=sample_type_equally(df[df.set =='train'], 500), palette=pal, 
     edgecolor="white", size=1, jitter=1, zorder=1, orient=ort, alpha=0.5)
    
    sns.boxplot(x=dx, y=dy, data=df[df.set =='train'], color="black", width=.2, zorder=10,
        showcaps=True, boxprops={'facecolor': 'none', "zorder": 10}, 
        showfliers=True, whiskerprops={'linewidth': 1, "zorder": 10}, 
        saturation=1, orient=ort, fliersize=0, linewidth=1)

    yticks = [1, 1.2, 1.5, 2]
    ax.set_yticks(np.log10(yticks))
    ax.set_yticklabels([r"{:.2f} $\times$".format(y) for y in yticks])
    plt.ylabel("Absolute Error")

    plt.xlim(-0.8, 4.3)
    plt.xticks([], [])
    plt.xlabel("")
    ax.set_title("Training set")

    #
    # Second figure
    #
    plt.subplot(212)

    ax=pt.half_violinplot(x=dx, y=dy, data=df[df.set =='test'], palette=pal, bw=.1, cut=0.,
        scale="width", width=1., inner=None, orient=ort, linewidth=0.8, offset=0.2)

    ax=sns.stripplot(x=dx, y=dy, data=sample_type_equally(df[df.set =='test'], 500), palette=pal, 
        edgecolor="white", size=1, jitter=1, zorder=1, orient=ort, alpha=0.5)
    
    ax=sns.boxplot(x=dx, y=dy, data=df[df.set =='test'], color="black", width=.2, zorder=10,
        showcaps=True, boxprops={'facecolor': 'none', "zorder": 10}, 
        showfliers=True, whiskerprops={'linewidth': 1, "zorder": 10}, 
        saturation=1, orient=ort, fliersize=0, linewidth=1)

    yticks = [1, 1.2, 1.5, 2]
    ax.set_yticks(np.log10(yticks))
    ax.set_yticklabels([r"{:.2f} $\times$".format(y) for y in yticks])
    plt.ylabel("Absolute Error")

    plt.xlim(-0.8, 4.3)
    # plt.xticks(range(5), ["POSIX", "POSIX+runtime", "MPI-IO", "POSIX+MPI-IO", "POSIX+Cobalt"], rotation=30)
    plt.xticks(np.arange(5)-0.5, [
        "POSIX\nTest set median={:.2f}$\\times$".format(10**results[(results.set=='test') & (results.type=='posix')].median().item()),
        "POSIX+runtime\nTest set median={:.2f}$\\times$".format(10**results[(results.set=='test') & (results.type=='runtime')].median().item()),
        "MPI-IO\nTest set median={:.2f}$\\times$".format(10**results[(results.set=='test') & (results.type=='mpiio')].median().item()),
        "POSIX+MPI-IO\nTest set median={:.2f}$\\times$".format(10**results[(results.set=='test') & (results.type=='both')].median().item()),
        "POSIX+Cobalt\nTest set median={:.2f}$\\times$".format(10**results[(results.set=='test') & (results.type=='cobalt')].median().item())
    ], rotation=30, ha='right')
    plt.xlabel("")
    ax.set_title("Test set")

    plt.savefig("figures/figure_2.pdf", dpi=600, bbox_inches='tight')


if __name__ == "__main__":
    compare_datasets()
