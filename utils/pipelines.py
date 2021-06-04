"""
The presentation pipeline applies only transformations that affect what and how 
data is visualized, e.g., it may create artificial features if such features 
are useful to the user, or it may discard some features, etc.

The clustering pipeline preprocesses the data before it's fed to the clusterer.
This typically includes selecting a subset of features to use, scaling certain
features by other features, taking the log values of features, normalizing 
features, saniziting samples and features, etc. 

In the end, all of the features belong to the presentation pipeline output, while 
only POSIX_FEATURES and MPIIO_FEATURES belong to the clustering pipeline output.
"""
import re
import logging 
import numpy as np
import pandas as pd 


POSIX_FEATURES = [
    'POSIX_OPENS_LOG10'             , 'POSIX_SEEKS_LOG10'             , 'POSIX_STATS_LOG10'            ,
    'POSIX_MMAPS_LOG10'             , 'POSIX_FSYNCS_LOG10'            , 'POSIX_FDSYNCS_LOG10'          ,
    'POSIX_MODE_LOG10'              , 'POSIX_MEM_ALIGNMENT_LOG10'     , 'POSIX_FILE_ALIGNMENT_LOG10'   ,
    'RUNTIME_LOG10'                 , 'NPROCS_LOG10'                  , 'POSIX_TOTAL_ACCESSES_LOG10'   ,
    'POSIX_TOTAL_BYTES_LOG10'       , 'POSIX_TOTAL_FILES_LOG10'       , 'POSIX_BYTES_READ_PERC'        ,
    'POSIX_READS_PERC'              , 'POSIX_RW_SWITCHES_PERC'        , 'POSIX_SEQ_READS_PERC'         ,
    'POSIX_SEQ_WRITES_PERC'         , 'POSIX_CONSEC_READS_PERC'       , 'POSIX_CONSEC_WRITES_PERC'     ,
    'POSIX_FILE_NOT_ALIGNED_PERC'   , 'POSIX_MEM_NOT_ALIGNED_PERC'    , 'POSIX_SIZE_READ_0_100_PERC'   ,
    'POSIX_SIZE_READ_100_1K_PERC'   , 'POSIX_SIZE_READ_1K_10K_PERC'   , 'POSIX_SIZE_READ_10K_100K_PERC',
    'POSIX_SIZE_READ_100K_1M_PERC'  , 'POSIX_SIZE_READ_1M_4M_PERC'    , 'POSIX_SIZE_READ_4M_10M_PERC'  ,
    'POSIX_SIZE_READ_10M_100M_PERC' , 'POSIX_SIZE_READ_100M_1G_PERC'  , 'POSIX_SIZE_READ_1G_PLUS_PERC' ,
    'POSIX_SIZE_WRITE_0_100_PERC'   , 'POSIX_SIZE_WRITE_100_1K_PERC'  , 'POSIX_SIZE_WRITE_1K_10K_PERC' ,
    'POSIX_SIZE_WRITE_10K_100K_PERC', 'POSIX_SIZE_WRITE_100K_1M_PERC' , 'POSIX_SIZE_WRITE_1M_4M_PERC'  ,
    'POSIX_SIZE_WRITE_4M_10M_PERC'  , 'POSIX_SIZE_WRITE_10M_100M_PERC', 'POSIX_SIZE_WRITE_100M_1G_PERC',
    'POSIX_SIZE_WRITE_1G_PLUS_PERC'
]


MPIIO_FEATURES = [
    'RUNTIME_LOG10'                 , 'NPROCS_LOG10'                  , 'MPIIO_RAW_BYTES_LOG10'        ,
    'MPIIO_BYTES_READ_PERC'         , 'MPIIO_RAW_ACCESSES_LOG10'      , 'MPIIO_INDEP_ACCESSES_PERC'    ,
    'MPIIO_COLL_ACCESSES_PERC'      , 'MPIIO_SPLIT_ACCESSES_PERC'     , 'MPIIO_NB_ACCESSES_PERC'       ,
    'MPIIO_INDEP_READS_PERC'        , 'MPIIO_COLL_READS_PERC'         , 'MPIIO_SPLIT_READS_PERC'       ,
    'MPIIO_NB_READS_PERC'           , 'MPIIO_RAW_INDEP_OPENS_LOG10'   , 'MPIIO_RAW_COLL_OPENS_LOG10'   ,
    'MPIIO_RAW_SYNCS_LOG10'         , 'MPIIO_RAW_HINTS_LOG10'         , 'MPIIO_RAW_VIEWS_LOG10'        ,
    'MPIIO_RAW_MODE_LOG10'          , 'MPIIO_RAW_RW_SWITCHES_LOG10'   , 'MPIIO_SIZE_READ_0_100_PERC'   ,
    'MPIIO_SIZE_READ_100_1K_PERC'   , 'MPIIO_SIZE_READ_1K_10K_PERC'   , 'MPIIO_SIZE_READ_10K_100K_PERC',
    'MPIIO_SIZE_READ_100K_1M_PERC'  , 'MPIIO_SIZE_READ_1M_4M_PERC'    , 'MPIIO_SIZE_READ_4M_10M_PERC'  ,
    'MPIIO_SIZE_READ_10M_100M_PERC' , 'MPIIO_SIZE_READ_100M_1G_PERC'  , 'MPIIO_SIZE_READ_1G_PLUS_PERC' ,
    'MPIIO_SIZE_WRITE_0_100_PERC'   , 'MPIIO_SIZE_WRITE_100_1K_PERC'  , 'MPIIO_SIZE_WRITE_1K_10K_PERC' ,
    'MPIIO_SIZE_WRITE_10K_100K_PERC', 'MPIIO_SIZE_WRITE_100K_1M_PERC' , 'MPIIO_SIZE_WRITE_1M_4M_PERC'  ,
    'MPIIO_SIZE_WRITE_4M_10M_PERC'  , 'MPIIO_SIZE_WRITE_10M_100M_PERC', 'MPIIO_SIZE_WRITE_100M_1G_PERC',
    'MPIIO_SIZE_WRITE_1G_PLUS_PERC'
]

BOTH_FEATURES = list(set(MPIIO_FEATURES).union(set(POSIX_FEATURES))) 


def get_number_columns(df):
    """
    Since some columns contain string metadata, and others contain values,
    this function returns the columns that contain values.
    """
    return df.columns[np.logical_or(df.dtypes == np.float64, df.dtypes == np.int64)]


def remove_NaN_features(df):
    """
    Removes features don't have values at all.
    """
    for column in get_number_columns(df): 
        if np.all(np.isnan(df[column])):
            df = df.drop(columns=column)
            logging.info("Removing NaN feature {}".format(column))

    return df


def remove_NaN_jobs(df, columns):
    """
    Removes any rows that have NaN values.
    """
    bad_rows = pd.isnull(df[columns]).any(axis=1)
    logging.info("Removing {} jobs that have NaN values".format(np.sum(bad_rows)))
    return df.loc[~bad_rows] 


def remove_subzero_features_and_jobs(df, min_zeros_to_drop=10000):
    """
    Remove columns with too many sub-zero values and jobs with negative values.
    """ 
    # First, drop bad columns
    drop_columns = []
    for idx, c in enumerate(df.columns):
        if df.dtypes[idx] == np.int64 or df.dtypes[idx] == np.float64:
            subzeros = np.sum(df[c] < 0)

            if subzeros > 0:
                logging.info("{} jobs had a negative value in column {}".format(subzeros, c))

            if subzeros > min_zeros_to_drop:
                drop_columns.append(c)
                logging.info("Dropping column {}".format(c))

    df = df.drop(columns=drop_columns)

    # Next, drop jobs that have negative values
    jobs_without_zeros = np.sum(df[get_number_columns(df)] < 0, axis=1) == 0
    pd.options.display.max_rows = 999
    logging.info("Number of zero values / feature")
    logging.info(np.sum(df[get_number_columns(df)] < 0)[np.sum(df[get_number_columns(df)] < 0) > 0])
    logging.info("Removing {} jobs".format(np.sum(~jobs_without_zeros)))
    df = df.loc[jobs_without_zeros]

    return df


def extract_users(df):
    """
    From the filenames, extracts users and adds a user column.
    """
    df['users'] = [re.match(r"([a-zA-Z0-9\+]*)_([a-zA-Z0-9_\-.\+]+)_id.*", re.findall(r"[a-zA-Z0-9_.\+-]+\.darshan", p)[0], re.MULTILINE).groups()[0] for p in df.FILENAME]
    return df


def extract_apps(df):
    """
    From the filenames, extracts applications and adds an application column.
    """
    df['apps']       = [re.match(r"([a-zA-Z0-9\+]*)_([a-zA-Z0-9_\-.\+]+)_id.*", re.findall(r"[a-zA-Z0-9_.\+-]+\.darshan", p)[0], re.MULTILINE).groups()[1] for p in df.FILENAME]
    df['apps_short'] = [re.match(r"([a-zA-Z0-9]+).*", x) for x in df.apps]
    df['apps_short'] = [x.groups(1)[0] if x is not None else "" for x in df.apps_short]
    return df


def replace_timestamps(df): 
    """
    Replace timestamps with appropriate intervals.
    """
    for label in df.columns:
        if "START_TIMESTAMP" in label: 
            if label.replace("START_TIMESTAMP", "END_TIMESTAMP") in df.columns:
                start_label = label
                end_label   = label.replace("START_TIMESTAMP", "END_TIMESTAMP")
                delta_label = label.replace("START_TIMESTAMP", "DELTA")

                df[delta_label] = df[end_label] - df[start_label]
                df = df.drop(columns=[start_label, end_label])

                # Log how many jobs had negative end timestamps or start timestamps > end timestamps
                if np.sum(df[delta_label] < 0) > 0:
                    logging.info("Column {} had {} negative delta periods".format(delta_label, np.sum(df[delta_label] < 0)))
            else:
                logging.error("Found column {} but could not find matching column {}".format(label, label.replace("START_TIMESTAMP", "END_TIMESTAMP")))

    # Check if we didn't remove any END_TIMESTAMP columns
    for label in df.columns: 
        if "END_TIMESTAMP" in label: 
            logging.error("Found column {} that did not have a maching start column".format(label))

    return df


def convert_features_to_percentages(df):
    """
    Certain features like POSIX_SEQ_READS make more sense when normalized by a more general feature such as POSIX_READS
    For all features that measure either the number of a certain type of access, or the number of bytes, we normalize by
    the total number POSIX accesses and total number of POSIX bytes accessed.
    """
    df = df.copy()

    # These artificial features are calculated by Darshan, but we'll redo the calculations
    df.drop(columns=["POSIX_TOTAL_BYTES", "MPIIO_TOTAL_BYTES", "STDIO_TOTAL_BYTES", 
        "POSIX_TOTAL_TOTAL_BYTES", "MPIIO_TOTAL_TOTAL_BYTES", 
        "POSIX_TOTAL_FILE_COUNT", "MPIIO_TOTAL_FILE_COUNT"], inplace=True)

    # Remove total from columns 
    df.rename(columns={c: c.replace("TOTAL_", "") for c in df.columns}, inplace=True)

    # Specific to the ANL CSVs
    df["RUNTIME"] = df["RUN TIME"]
    df.drop(columns=["RUN TIME"], inplace=True)
    
    df['POSIX_TOTAL_ACCESSES'] = df.POSIX_WRITES     + df.POSIX_READS
    df['POSIX_TOTAL_BYTES']    = df.POSIX_BYTES_READ + df.POSIX_BYTES_WRITTEN
    df['POSIX_TOTAL_FILES']    = df.POSIX_SHARED_FILE_COUNT + df.POSIX_UNIQUE_FILE_COUNT

    #
    # Byte count percentage features
    #
    try:
        df['POSIX_BYTES_READ_PERC'      ] = df.POSIX_BYTES_READ       / df.POSIX_TOTAL_BYTES
        df['POSIX_UNIQUE_BYTES_PERC'    ] = df.POSIX_UNIQUE_BYTES     / df.POSIX_TOTAL_BYTES
        df['POSIX_SHARED_BYTES_PERC'    ] = df.POSIX_SHARED_BYTES     / df.POSIX_TOTAL_BYTES
        df['POSIX_READ_ONLY_BYTES_PERC' ] = df.POSIX_READ_ONLY_BYTES  / df.POSIX_TOTAL_BYTES
        df['POSIX_READ_WRITE_BYTES_PERC'] = df.POSIX_READ_WRITE_BYTES / df.POSIX_TOTAL_BYTES
        df['POSIX_WRITE_ONLY_BYTES_PERC'] = df.POSIX_WRITE_ONLY_BYTES / df.POSIX_TOTAL_BYTES
    except:
        logging.error("Failed to normalize one of the features in [POSIX_BYTES_READ, POSIX_BYTES_WRITTEN, unique_bytes, shared_bytes, read_only_bytes, read_write_bytes, write_only_bytes") 

    #
    # File count percentage features
    #
    try: 
        df['POSIX_UNIQUE_FILES_PERC']     = df.POSIX_UNIQUE_FILE_COUNT     / df.POSIX_TOTAL_FILES
        df['POSIX_SHARED_FILES_PERC']     = df.POSIX_SHARED_FILE_COUNT     / df.POSIX_TOTAL_FILES
        df['POSIX_READ_ONLY_FILES_PERC']  = df.POSIX_READ_ONLY_FILE_COUNT  / df.POSIX_TOTAL_FILES
        df['POSIX_READ_WRITE_FILES_PERC'] = df.POSIX_READ_WRITE_FILE_COUNT / df.POSIX_TOTAL_FILES
        df['POSIX_WRITE_ONLY_FILES_PERC'] = df.POSIX_WRITE_ONLY_FILE_COUNT / df.POSIX_TOTAL_FILES
    except:
        logging.error("Failed to normalize one of the *_files features")


    try:
        df['POSIX_READS_PERC']            = df.POSIX_READS            / df.POSIX_TOTAL_ACCESSES 
        df['POSIX_WRITES_PERC']           = df.POSIX_WRITES           / df.POSIX_TOTAL_ACCESSES 
        df['POSIX_RW_SWITCHES_PERC']      = df.POSIX_RW_SWITCHES      / df.POSIX_TOTAL_ACCESSES 
        df['POSIX_SEQ_READS_PERC']        = df.POSIX_SEQ_READS        / df.POSIX_TOTAL_ACCESSES 
        df['POSIX_SEQ_WRITES_PERC']       = df.POSIX_SEQ_WRITES       / df.POSIX_TOTAL_ACCESSES
        df['POSIX_CONSEC_READS_PERC']     = df.POSIX_CONSEC_READS     / df.POSIX_TOTAL_ACCESSES
        df['POSIX_CONSEC_WRITES_PERC']    = df.POSIX_CONSEC_WRITES    / df.POSIX_TOTAL_ACCESSES
        df['POSIX_FILE_NOT_ALIGNED_PERC'] = df.POSIX_FILE_NOT_ALIGNED / df.POSIX_TOTAL_ACCESSES
        df['POSIX_MEM_NOT_ALIGNED_PERC']  = df.POSIX_MEM_NOT_ALIGNED  / df.POSIX_TOTAL_ACCESSES
        # df = df.drop(columns=["POSIX_READS", "POSIX_WRITES", "POSIX_RW_SWITCHES", "POSIX_SEQ_WRITES", "POSIX_SEQ_READS", "POSIX_CONSEC_READS", "POSIX_CONSEC_WRITES", "POSIX_FILE_NOT_ALIGNED", "POSIX_MEM_NOT_ALIGNED"])
    except:
        logging.error("Failed to normalize one of the features in [POSIX_READS, POSIX_WRITES, POSIX_SEQ_WRITES, POSIX_SEQ_READS, POSIX_CONSEC_READS, POSIX_CONSEC_WRITES, POSIX_FILE_NOT_ALIGNED_PERC, POSIX_MEM_NOT_ALIGNED_PERC]") 


    try:
        # NaN comparisons make this assert difficult
        # if np.any(df.POSIX_SIZE_READ_0_100   + df.POSIX_SIZE_READ_100_1K + df.POSIX_SIZE_READ_1K_10K + df.POSIX_SIZE_READ_10K_100K +
                  # df.POSIX_SIZE_READ_100K_1M + df.POSIX_SIZE_READ_1M_4M  + df.POSIX_SIZE_READ_4M_10M + df.POSIX_SIZE_READ_10M_100M +
                  # df.POSIX_SIZE_READ_100M_1G + df.POSIX_SIZE_READ_1G_PLUS +
                  # df.POSIX_SIZE_WRITE_0_100   + df.POSIX_SIZE_WRITE_100_1K + df.POSIX_SIZE_WRITE_1K_10K + df.POSIX_SIZE_WRITE_10K_100K +
                  # df.POSIX_SIZE_WRITE_100K_1M + df.POSIX_SIZE_WRITE_1M_4M  + df.POSIX_SIZE_WRITE_4M_10M + df.POSIX_SIZE_WRITE_10M_100M +
                  # df.POSIX_SIZE_WRITE_100M_1G + df.POSIX_SIZE_WRITE_1G_PLUS != df.POSIX_TOTAL_ACCESSES):
            # logging.warning("POSIX_SIZE_WRITE* + POSIX_SIZE_READ* columns do not add up to POSIX_TOTAL_ACCESSES")


        df['POSIX_SIZE_READ_0_100_PERC'    ] = df.POSIX_SIZE_READ_0_100     / df.POSIX_TOTAL_ACCESSES
        df['POSIX_SIZE_READ_100_1K_PERC'   ] = df.POSIX_SIZE_READ_100_1K    / df.POSIX_TOTAL_ACCESSES 
        df['POSIX_SIZE_READ_1K_10K_PERC'   ] = df.POSIX_SIZE_READ_1K_10K    / df.POSIX_TOTAL_ACCESSES 
        df['POSIX_SIZE_READ_10K_100K_PERC' ] = df.POSIX_SIZE_READ_10K_100K  / df.POSIX_TOTAL_ACCESSES 
        df['POSIX_SIZE_READ_100K_1M_PERC'  ] = df.POSIX_SIZE_READ_100K_1M   / df.POSIX_TOTAL_ACCESSES 
        df['POSIX_SIZE_READ_1M_4M_PERC'    ] = df.POSIX_SIZE_READ_1M_4M     / df.POSIX_TOTAL_ACCESSES 
        df['POSIX_SIZE_READ_4M_10M_PERC'   ] = df.POSIX_SIZE_READ_4M_10M    / df.POSIX_TOTAL_ACCESSES 
        df['POSIX_SIZE_READ_10M_100M_PERC' ] = df.POSIX_SIZE_READ_10M_100M  / df.POSIX_TOTAL_ACCESSES 
        df['POSIX_SIZE_READ_100M_1G_PERC'  ] = df.POSIX_SIZE_READ_100M_1G   / df.POSIX_TOTAL_ACCESSES 
        df['POSIX_SIZE_READ_1G_PLUS_PERC'  ] = df.POSIX_SIZE_READ_1G_PLUS   / df.POSIX_TOTAL_ACCESSES 

        df['POSIX_SIZE_WRITE_0_100_PERC'   ] = df.POSIX_SIZE_WRITE_0_100    / df.POSIX_TOTAL_ACCESSES
        df['POSIX_SIZE_WRITE_100_1K_PERC'  ] = df.POSIX_SIZE_WRITE_100_1K   / df.POSIX_TOTAL_ACCESSES
        df['POSIX_SIZE_WRITE_1K_10K_PERC'  ] = df.POSIX_SIZE_WRITE_1K_10K   / df.POSIX_TOTAL_ACCESSES
        df['POSIX_SIZE_WRITE_10K_100K_PERC'] = df.POSIX_SIZE_WRITE_10K_100K / df.POSIX_TOTAL_ACCESSES
        df['POSIX_SIZE_WRITE_100K_1M_PERC' ] = df.POSIX_SIZE_WRITE_100K_1M  / df.POSIX_TOTAL_ACCESSES
        df['POSIX_SIZE_WRITE_1M_4M_PERC'   ] = df.POSIX_SIZE_WRITE_1M_4M    / df.POSIX_TOTAL_ACCESSES
        df['POSIX_SIZE_WRITE_4M_10M_PERC'  ] = df.POSIX_SIZE_WRITE_4M_10M   / df.POSIX_TOTAL_ACCESSES
        df['POSIX_SIZE_WRITE_10M_100M_PERC'] = df.POSIX_SIZE_WRITE_10M_100M / df.POSIX_TOTAL_ACCESSES
        df['POSIX_SIZE_WRITE_100M_1G_PERC' ] = df.POSIX_SIZE_WRITE_100M_1G  / df.POSIX_TOTAL_ACCESSES
        df['POSIX_SIZE_WRITE_1G_PLUS_PERC' ] = df.POSIX_SIZE_WRITE_1G_PLUS  / df.POSIX_TOTAL_ACCESSES

        # drop_columns = ["POSIX_SIZE_READ_0_100",   "POSIX_SIZE_READ_100_1K", "POSIX_SIZE_READ_1K_10K", "POSIX_SIZE_READ_10K_100K",
                        # "POSIX_SIZE_READ_100K_1M", "POSIX_SIZE_READ_1M_4M", "POSIX_SIZE_READ_4M_10M", "POSIX_SIZE_READ_10M_100M",
                        # "POSIX_SIZE_READ_100M_1G", "POSIX_SIZE_READ_1G_PLUS",
                        # "POSIX_SIZE_WRITE_0_100",   "POSIX_SIZE_WRITE_100_1K", "POSIX_SIZE_WRITE_1K_10K", "POSIX_SIZE_WRITE_10K_100K",
                        # "POSIX_SIZE_WRITE_100K_1M", "POSIX_SIZE_WRITE_1M_4M", "POSIX_SIZE_WRITE_4M_10M", "POSIX_SIZE_WRITE_10M_100M",
                        # "POSIX_SIZE_WRITE_100M_1G", "POSIX_SIZE_WRITE_1G_PLUS"]

        # df = df.drop(columns=drop_columns)
    except:
        logging.warning("Failed to normalize POSIX_SIZE_*") 
        

    try:
        df['POSIX_ACCESS1_COUNT_PERC'] = df.POSIX_ACCESS1_COUNT / df.POSIX_TOTAL_ACCESSES
        df['POSIX_ACCESS2_COUNT_PERC'] = df.POSIX_ACCESS2_COUNT / df.POSIX_TOTAL_ACCESSES
        df['POSIX_ACCESS3_COUNT_PERC'] = df.POSIX_ACCESS3_COUNT / df.POSIX_TOTAL_ACCESSES
        df['POSIX_ACCESS4_COUNT_PERC'] = df.POSIX_ACCESS4_COUNT / df.POSIX_TOTAL_ACCESSES

        # logging.info("Normalized access values:")
        # logging.info("Access 1 %: max={}, mean={}, median={}".format(np.max(df.POSIX_ACCESS1_COUNT_PERC), np.mean(df.POSIX_ACCESS1_COUNT_PERC), np.median(df.POSIX_ACCESS1_COUNT_PERC)))
        # logging.info("Access 2 %: max={}, mean={}, median={}".format(np.max(df.POSIX_ACCESS2_COUNT_PERC), np.mean(df.POSIX_ACCESS2_COUNT_PERC), np.median(df.POSIX_ACCESS2_COUNT_PERC)))
        # logging.info("Access 3 %: max={}, mean={}, median={}".format(np.max(df.POSIX_ACCESS3_COUNT_PERC), np.mean(df.POSIX_ACCESS3_COUNT_PERC), np.median(df.POSIX_ACCESS3_COUNT_PERC)))
        # logging.info("Access 4 %: max={}, mean={}, median={}".format(np.max(df.POSIX_ACCESS4_COUNT_PERC), np.mean(df.POSIX_ACCESS4_COUNT_PERC), np.median(df.POSIX_ACCESS4_COUNT_PERC)))

        # df = df.drop(columns=['POSIX_ACCESS1_COUNT', 'POSIX_ACCESS2_COUNT', 'POSIX_ACCESS3_COUNT', 'POSIX_ACCESS4_COUNT'])
    except: 
        logging.warning("Failed to normalize POSIX_ACCESS[1-4]_COUNT") 


    try: 
        df['POSIX_F_READ_DELTA_PERC'       ] = df['POSIX_F_READ_DELTA'       ] / df.RUNTIME
        df['POSIX_F_WRITE_DELTA_PERC'      ] = df['POSIX_F_WRITE_DELTA'      ] / df.RUNTIME
        df['POSIX_F_CLOSE_DELTA_PERC'      ] = df['POSIX_F_CLOSE_DELTA'      ] / df.RUNTIME
        df['POSIX_F_OPEN_DELTA_PERC'       ] = df['POSIX_F_OPEN_DELTA'       ] / df.RUNTIME

        df["POSIX_F_READ_TIME_PERC"        ] = df["POSIX_F_READ_TIME"        ] / df.RUNTIME
        df["POSIX_F_WRITE_TIME_PERC"       ] = df["POSIX_F_WRITE_TIME"       ] / df.RUNTIME
        df["POSIX_F_META_TIME_PERC"        ] = df["POSIX_F_META_TIME"        ] / df.RUNTIME
        df["POSIX_F_MAX_READ_TIME_PERC"    ] = df["POSIX_F_MAX_READ_TIME"    ] / df.RUNTIME
        df["POSIX_F_MAX_WRITE_TIME_PERC"   ] = df["POSIX_F_MAX_WRITE_TIME"   ] / df.RUNTIME

        # keep = set(['POSIX_F_READ_DELTA_PERC', 'POSIX_F_WRITE_DELTA_PERC', 'POSIX_F_CLOSE_DELTA_PERC', 'POSIX_F_OPEN_DELTA_PERC', 
                    # # "POSIX_F_READ_TIME_PERC", "POSIX_F_WRITE_TIME_PERC", "POSIX_F_META_TIME_PERC", 
                    # "POSIX_F_READ_TIME",        "POSIX_F_WRITE_TIME",      "POSIX_F_META_TIME", 
                    # "POSIX_F_MAX_READ_TIME_PERC", "POSIX_F_MAX_WRITE_TIME_PERC"
                    # ])

        # drop = set([x for x in df.columns if "_TIME" in x or "DELTA" in x]).difference(keep)
        # df = df.drop(columns=drop)
    except:
        logging.warning("Failed to normalize DELTA features: ")

    #
    # Similar to convert_POSIX_features_to_percentages, except on MPIIO
    #
    # Question: since indep, coll, split and nb accesses add up to the histogram accesses, 
    # how should we normalize the four? 
    #

    #
    # Bytes features: one absolute, one relative
    #
    df["MPIIO_RAW_BYTES"]       = df["MPIIO_BYTES_READ" ] + df["MPIIO_BYTES_WRITTEN"]
    df["MPIIO_BYTES_READ_PERC"] = df["MPIIO_BYTES_READ" ] / df["MPIIO_RAW_BYTES"    ]

    #
    # Four types of accesses: one absolute number of accesses, four relative, and for relative R/W breakdowns
    #
    df["MPIIO_RAW_ACCESSES"] = df["MPIIO_INDEP_READS"] + df["MPIIO_INDEP_WRITES"] \
                             + df["MPIIO_COLL_READS" ] + df["MPIIO_COLL_WRITES" ] \
                             + df["MPIIO_SPLIT_READS"] + df["MPIIO_SPLIT_WRITES"] \
                             + df["MPIIO_NB_READS"   ] + df["MPIIO_NB_WRITES"   ] 

    df["MPIIO_INDEP_ACCESSES_PERC"] = (df["MPIIO_INDEP_READS"] + df["MPIIO_INDEP_WRITES"]) / df["MPIIO_RAW_ACCESSES"]
    df["MPIIO_COLL_ACCESSES_PERC" ] = (df["MPIIO_COLL_READS" ] + df["MPIIO_COLL_WRITES" ]) / df["MPIIO_RAW_ACCESSES"]
    df["MPIIO_SPLIT_ACCESSES_PERC"] = (df["MPIIO_SPLIT_READS"] + df["MPIIO_SPLIT_WRITES"]) / df["MPIIO_RAW_ACCESSES"]
    df["MPIIO_NB_ACCESSES_PERC"   ] = (df["MPIIO_NB_READS"   ] + df["MPIIO_NB_WRITES"   ]) / df["MPIIO_RAW_ACCESSES"]

    df["MPIIO_INDEP_READS_PERC"] = df["MPIIO_INDEP_READS"] / (df["MPIIO_INDEP_READS"] + df["MPIIO_INDEP_WRITES"])
    df["MPIIO_COLL_READS_PERC" ] = df["MPIIO_COLL_READS" ] / (df["MPIIO_COLL_READS" ] + df["MPIIO_COLL_WRITES" ])
    df["MPIIO_SPLIT_READS_PERC"] = df["MPIIO_SPLIT_READS"] / (df["MPIIO_SPLIT_READS"] + df["MPIIO_SPLIT_WRITES"])
    df["MPIIO_NB_READS_PERC"   ] = df["MPIIO_NB_READS"   ] / (df["MPIIO_NB_READS"   ] + df["MPIIO_NB_WRITES"   ])

    #
    # General features that can't be made relative
    #
    df['MPIIO_RAW_INDEP_OPENS'   ] = df['MPIIO_INDEP_OPENS']
    df['MPIIO_RAW_COLL_OPENS'    ] = df['MPIIO_COLL_OPENS' ]
    df['MPIIO_RAW_SYNCS'         ] = df['MPIIO_SYNCS'      ]
    df['MPIIO_RAW_HINTS'         ] = df['MPIIO_HINTS'      ]
    df['MPIIO_RAW_VIEWS'         ] = df['MPIIO_VIEWS'      ]
    df['MPIIO_RAW_MODE'          ] = df['MPIIO_MODE'       ]
    df['MPIIO_RAW_RW_SWITCHES'   ] = df['MPIIO_RW_SWITCHES']

    df.drop(columns=['MPIIO_INDEP_OPENS', 'MPIIO_COLL_OPENS', 'MPIIO_SYNCS', 
                     'MPIIO_HINTS',       'MPIIO_VIEWS',      'MPIIO_MODE'])

    #
    # Relative histogram features
    #
    df.rename(columns={c: c.replace("READ_AGG_", "READ_") for c in df.columns if "READ_AGG" in c}, inplace=True)
    df.rename(columns={c: c.replace("WRITE_AGG_", "WRITE_") for c in df.columns if "WRITE_AGG" in c}, inplace=True)

    df['MPIIO_SIZE_READ_0_100_PERC'    ] = df['MPIIO_SIZE_READ_0_100'    ] / df["MPIIO_RAW_ACCESSES"]
    df['MPIIO_SIZE_READ_100_1K_PERC'   ] = df['MPIIO_SIZE_READ_100_1K'   ] / df["MPIIO_RAW_ACCESSES"]
    df['MPIIO_SIZE_READ_1K_10K_PERC'   ] = df['MPIIO_SIZE_READ_1K_10K'   ] / df["MPIIO_RAW_ACCESSES"]
    df['MPIIO_SIZE_READ_10K_100K_PERC' ] = df['MPIIO_SIZE_READ_10K_100K' ] / df["MPIIO_RAW_ACCESSES"]
    df['MPIIO_SIZE_READ_100K_1M_PERC'  ] = df['MPIIO_SIZE_READ_100K_1M'  ] / df["MPIIO_RAW_ACCESSES"]
    df['MPIIO_SIZE_READ_1M_4M_PERC'    ] = df['MPIIO_SIZE_READ_1M_4M'    ] / df["MPIIO_RAW_ACCESSES"]
    df['MPIIO_SIZE_READ_4M_10M_PERC'   ] = df['MPIIO_SIZE_READ_4M_10M'   ] / df["MPIIO_RAW_ACCESSES"]
    df['MPIIO_SIZE_READ_10M_100M_PERC' ] = df['MPIIO_SIZE_READ_10M_100M' ] / df["MPIIO_RAW_ACCESSES"]
    df['MPIIO_SIZE_READ_100M_1G_PERC'  ] = df['MPIIO_SIZE_READ_100M_1G'  ] / df["MPIIO_RAW_ACCESSES"]
    df['MPIIO_SIZE_READ_1G_PLUS_PERC'  ] = df['MPIIO_SIZE_READ_1G_PLUS'  ] / df["MPIIO_RAW_ACCESSES"]
    df['MPIIO_SIZE_WRITE_0_100_PERC'   ] = df['MPIIO_SIZE_WRITE_0_100'   ] / df["MPIIO_RAW_ACCESSES"]
    df['MPIIO_SIZE_WRITE_100_1K_PERC'  ] = df['MPIIO_SIZE_WRITE_100_1K'  ] / df["MPIIO_RAW_ACCESSES"]
    df['MPIIO_SIZE_WRITE_1K_10K_PERC'  ] = df['MPIIO_SIZE_WRITE_1K_10K'  ] / df["MPIIO_RAW_ACCESSES"]
    df['MPIIO_SIZE_WRITE_10K_100K_PERC'] = df['MPIIO_SIZE_WRITE_10K_100K'] / df["MPIIO_RAW_ACCESSES"]
    df['MPIIO_SIZE_WRITE_100K_1M_PERC' ] = df['MPIIO_SIZE_WRITE_100K_1M' ] / df["MPIIO_RAW_ACCESSES"]
    df['MPIIO_SIZE_WRITE_1M_4M_PERC'   ] = df['MPIIO_SIZE_WRITE_1M_4M'   ] / df["MPIIO_RAW_ACCESSES"]
    df['MPIIO_SIZE_WRITE_4M_10M_PERC'  ] = df['MPIIO_SIZE_WRITE_4M_10M'  ] / df["MPIIO_RAW_ACCESSES"]
    df['MPIIO_SIZE_WRITE_10M_100M_PERC'] = df['MPIIO_SIZE_WRITE_10M_100M'] / df["MPIIO_RAW_ACCESSES"]
    df['MPIIO_SIZE_WRITE_100M_1G_PERC' ] = df['MPIIO_SIZE_WRITE_100M_1G' ] / df["MPIIO_RAW_ACCESSES"]
    df['MPIIO_SIZE_WRITE_1G_PLUS_PERC' ] = df['MPIIO_SIZE_WRITE_1G_PLUS' ] / df["MPIIO_RAW_ACCESSES"]

    #
    # In case some of the percentages were normalized using 0 values, set percentages to 0
    #
    df.fillna(0, inplace=True)
    # df[[c for c in df.columns if "PERC" in c]].fillna(0, inplace=True)

    return df


def log_scale_features(df, features, add_small_value=0.1, set_NaNs_to=-3):
    for feature in features:
        new_feature = feature + "_LOG10"
        df[new_feature] = np.log10(df[feature] + add_small_value).fillna(value=set_NaNs_to)
        df.loc[df[new_feature] == -np.inf, new_feature] = set_NaNs_to

    return df


def pipeline(df, main_module, min_job_volume=0):
    """
    Presentation pipeline described at the beginning of this file.
    """
    df = df.copy()

    df = df[df.POSIX_BYTES_READ + df.POSIX_BYTES_WRITTEN > min_job_volume]

    logging.info("Extracting apps and users")
    df = extract_apps(df)
    df = extract_users(df)

    logging.info("Replacing timestamps")
    df = replace_timestamps(df)
    logging.info("Converting features to percentages")
    df = convert_features_to_percentages(df)
    
    # Remove jobs missing MPI-IO 
    if main_module == "MPIIO":
        df = df.loc[df.MPIIO_RAW_ACCESSES != 0]

    # Apply log10 to non-percentage features used in clustering
    if main_module == "POSIX":
        log_features = [f.replace("_LOG10", "") for f in POSIX_FEATURES if "perc" not in f.lower()]
        log_features += ["POSIX_AGG_PERF_BY_SLOWEST"]
    elif main_module == "MPIIO":
        log_features = [f.replace("_LOG10", "") for f in MPIIO_FEATURES if "perc" not in f.lower()]
        log_features += ["POSIX_AGG_PERF_BY_SLOWEST"]
    elif main_module == "both":
        log_features = [f.replace("_LOG10", "") for f in BOTH_FEATURES  if "perc" not in f.lower()]
        log_features += ["POSIX_AGG_PERF_BY_SLOWEST"]
    else: 
        raise NotImplementedError("Don't know of main module {}".format(main_module))
    
    logging.info("Log-scaling features")
    df = log_scale_features(df, log_features)
    logging.info("Preprocessing pipeline complete")

    return df

