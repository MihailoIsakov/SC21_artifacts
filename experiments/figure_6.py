"""
Finds the minimum distance all pairs of jobs from a random training / test split, and compares them 
to the average distance. Evaluates the performace of k-nearest neighbors predictors relative to distance.

Tests both using all features, and just percentage ones.
"""
import os 
import numpy as np
import sklearn
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

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


def load_dataset():
    df, features = data_loader.get_dataset('data/darshan_theta_2017_2020.csv', 'POSIX', min_job_volume=0)

    df = df[df.POSIX_TOTAL_BYTES >= 10*1024**2]
    df = df.sample(100000, random_state=0)

    return df, features


df, features = load_dataset()


def calculate_distances_feature_split(split_timestamp_start, split_timestamp_end=None): 
    global df

    X_train, X_test, y_train, y_test = test_set_utils.feature_split(
        df                    = df,                                 # noqa:E251
        y_column              = "POSIX_AGG_PERF_BY_SLOWEST_LOG10",  # noqa:E251
        feature               = "START_TIME",                       # noqa:E251
        feature_threshold_min = split_timestamp_start,              # noqa:E251
        feature_threshold_max = split_timestamp_end,                # noqa:E251
        keep_columns          = features                            # noqa:E251
    )
    X_train, X_test, y_train, y_test = np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)

    argmin_distances, min_distances = sklearn.metrics.pairwise_distances_argmin_min(X_train, X_test, metric='manhattan', axis=0)

    errors = 10**(np.abs(y_train[argmin_distances] - y_test))

    return min_distances, errors


def calculate_distances_random_split(): 
    global df

    X_train, X_test, y_train, y_test = test_set_utils.random_split(df, "POSIX_AGG_PERF_BY_SLOWEST_LOG10", features, test_size=0.2)
    X_train, X_test, y_train, y_test = np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)

    argmin_distances, min_distances = sklearn.metrics.pairwise_distances_argmin_min(X_train, X_test, metric='manhattan', axis=0)

    errors = 10**(np.abs(y_train[argmin_distances] - y_test))

    return min_distances, errors


def test_set_type_comparison(gridsize=25, cmap="Spectral_r"):
    """
    Compares random split, HACC, E3SM, and data split test sets in terms of Manhattan distance 
    and difference in I/O throughput.
    """
    min_distances_feature, errors_feature   = calculate_distances_feature_split(1577836801)
    min_distances_random, errors_random     = calculate_distances_random_split()

    # Select subset of random jobs 
    select = min_distances_random >= 10**-5
    min_distances_random, errors_random =  min_distances_random[select], errors_random[select]

    # Select subset of 2020 jobs 
    select = min_distances_feature <= 15
    min_distances_feature, errors_feature=  min_distances_feature[select], errors_feature[select]

    plt.figure(figsize=(3.3, 1.5))
    plt.subplots_adjust(wspace=0.05)

    ax1 = plt.subplot(121)
    plt.hexbin(np.log10(min_distances_random), np.log10(errors_random+1e-5), bins='log', gridsize=gridsize, cmap=cmap, mincnt=1, linewidths=0)
    plt.ylabel("I/O throughput difference")
    plt.xticks([-3, -1, 1], ["0.001", "0.1", "10"])
    plt.yticks([0, 1, 2, 3, 4], ["1×", "10×", "100×", "1000×", "10000×"])
    plt.xlabel("Manhattan distance")
    plt.title("Random test set")

    ax2 = plt.subplot(122)
    plt.hexbin(min_distances_feature, np.log10(errors_feature+1e-5), bins='log', gridsize=gridsize, cmap=cmap, mincnt=1, linewidths=0)
    plt.yticks([0, 1, 2, 3, 4], [""]*5)
    plt.xticks([0, 5, 10, 15], [0, 5, 10, 15])
    plt.xlabel("Manhattan distance")
    plt.title("2020 jobs")

    cbar = plt.colorbar(ax=[ax1, ax2], aspect=10)
    cbar.ax.tick_params(size=0)

    # Save figure
    plt.savefig("figures/figure_6.pdf", dpi=600, bbox_inches='tight', pad_inches=0)


if __name__ == "__main__":
    test_set_type_comparison()


