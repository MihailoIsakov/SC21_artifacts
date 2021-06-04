"""
The data loader module serves to load the dataset, and should not do any preprocessing of  
the data. We currently support loading a set of CSV files (with the same headers), but if 
your data is e.g., in the JSON format, this is where you should process it. The final 
output is supposed to be a Pandas DataFrame. 
"""
import logging
import glob
import joblib
import pandas as pd

from . import pipelines


memory = joblib.Memory(location='.cache', verbose=0)


def load(paths, delimiter=","):
    """
    Load the CSV file at the provided path, and return a pandas DataFrame.
    """
    if not isinstance(paths, list): 
        paths = [paths]

    df = pd.DataFrame()
    for path in paths: 
        new_df = pd.read_csv(path, delimiter=delimiter)
        df = pd.concat([df, new_df])

    df = df.reset_index()

    return df


@memory.cache
def get_dataset(data_path, module, min_cluster_size=250, min_job_volume=10*1024**2):
    df = load(glob.glob(data_path))  

    df = pipelines.pipeline(df, module, min_job_volume=min_job_volume)

    feature_map = {
        "POSIX": pipelines.POSIX_FEATURES,
        "MPIIO": pipelines.MPIIO_FEATURES,
        "both":  pipelines.BOTH_FEATURES
    }

    features = feature_map[module]

    return df, features


