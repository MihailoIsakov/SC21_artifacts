"""
Functions for building the test set given the dataset DataFrame, 
(and possibly the clusterer)
"""
from collections import Counter
import numpy as np
import sklearn
from sklearn.model_selection import LeaveOneGroupOut, GroupKFold
from sklearn.utils.validation import check_array


def feature_split(df, y_column, feature, feature_threshold_min, feature_threshold_max=None, keep_columns=None):
    """
    Splits the dataset into rows whose feature is below feature_threshold_min, and 
    rows whose feature is above. If feature_threshold_max is not None, the second 
    split is in the range (feature_threshold_min, feature_threshold_max].

    Args: 
        df: dataset DataFrame 
        y_column: name of column we want to use as target 
        keep_columns: either None, or a list of columns we want to keep in the input sets 

    Returns:
        training and test input and target data
    """
    if keep_columns is None:
        keep_columns = set(df.columns).difference([y_column])

    below_min = df[feature] <= feature_threshold_min
    above_min = df[feature] > feature_threshold_min
    below_max = df[feature] <= feature_threshold_max

    if feature_threshold_max is not None: 
        above_min = above_min & below_max

    X_train = df[below_min][keep_columns]
    X_test  = df[above_min][keep_columns]
    y_train = df[below_min][y_column]
    y_test  = df[above_min][y_column]

    return X_train, X_test, y_train, y_test 


def random_split(df, y_column, keep_columns=None, test_size=0.2):
    """
    Original test set function that doesn't show generalization well.

    Args: 
        df: dataset DataFrame 
        y_column: name of column we want to use as target 
        keep_columns: either None, or a list of columns we want to keep in the input sets 

    Returns:
        training and test input and target data
    """
    if keep_columns is None:
        keep_columns = set(df.columns).difference([y_column])

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(df[keep_columns], df[y_column], test_size=test_size)

    return X_train, X_test, y_train, y_test 


def per_app_split(df, app_name, y_column, keep_columns=None):
    """
    Selects all jobs of the given short application name, and uses them in the test set.

    Args: 
        df: dataset DataFrame 
        app_name: name of application we want to use as test set
        y_column: name of column we want to use as target 
        keep_columns: either None, or a list of columns we want to keep in the input sets 

    Returns:
        training and test input and target data
    """
    if keep_columns is None:
        keep_columns = set(df.columns).difference([y_column])

    X_train = df[df.apps_short != app_name][keep_columns]
    y_train = df[df.apps_short != app_name][y_column]
                                          
    X_test  = df[df.apps_short == app_name][keep_columns]
    y_test  = df[df.apps_short == app_name][y_column]

    return X_train, X_test, y_train, y_test 


class AppFold(LeaveOneGroupOut):
    """ 
    AppFold cross-validator 

    Provides train/test indices to split data in train/test sets. Each
    set of jobs belonging to one of the top n applications is used once
    as the test set (singleton) while the remaining samples form the 
    training set. Jobs that do not belong to the n most numerous 
    applications are never included in the test set.
    """
    def __init__(self, n_splits):
        self.n_splits = n_splits

    def get_n_splits(self, X=None, y=None):
        return self.n_splits

    def split(self, X, y=None):
        """
        Creates the groups based on the DataFrame X and the n_splits parameter,
        then just calls the base class implementation.
        """
        top_apps = Counter(X.apps_short).most_common()[:self.n_splits]
        top_apps = [ta[0] for ta in top_apps]

        for idx, app in enumerate(top_apps):
            train_index = np.flatnonzero(X.apps_short != app)
            test_index  = np.flatnonzero(X.apps_short == app)

            yield train_index, test_index


class FeatureFold(LeaveOneGroupOut):
    """ 
    FeatureFold cross-validator 

    Provides train/test indices to split data in train/test sets. 
    Samples with the feature within the training min and max bound 
    are put in the training set. Samples with the feature within the 
    test min and max bound are put in the test set. The rest of the 
    samples are ignored
    """
    def __init__(self, feature, train_min, train_max, test_min, test_max):
        self.n_splits  = 1
        self.feature   = feature
        self.train_min = train_min
        self.train_max = train_max
        self.test_min  = test_min
        self.test_max  = test_max

    def get_n_splits(self, X=None, y=None):
        return self.n_splits

    def split(self, X, y=None):
        """
        Creates the groups based on the DataFrame X and the n_splits parameter,
        then just calls the base class implementation.
        """
        train_index = np.flatnonzero(np.logical_and(X[self.feature] >= self.train_min, X[self.feature] < self.train_max))
        test_index  = np.flatnonzero(np.logical_and(X[self.feature] >= self.test_min,  X[self.feature] < self.test_max))

        yield train_index, test_index


