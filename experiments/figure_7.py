import os
import numpy as np
import pandas as pd 
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns 


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


# Load data downloaded from https://gauge.ascslab-tools.org/ 
df = pd.read_csv("data/gauge_data.csv")
df["empty"] = ""

# Preprocessing
df["POSIX_RAW_total_bytes"]         = np.log10(df["POSIX_RAW_total_bytes"] / 1024 ** 2)
df["POSIX_RAW_total_accesses"]      = np.log10(df["POSIX_RAW_total_accesses"])
df["POSIX_RAW_total_files"]         = np.log10(df["POSIX_RAW_total_files"])
df["RAW_runtime"]                   = np.log10(df["RAW_runtime"]) 
df["RAW_nprocs"]                    = np.log10(df["RAW_nprocs"])
df["POSIX_RAW_agg_perf_by_slowest"] = np.log10(df["POSIX_RAW_agg_perf_by_slowest"])

#
# First graph
#
columns = [
    "RAW_nprocs",
    "POSIX_RAW_total_accesses",
    "POSIX_RAW_total_files",
    "POSIX_RAW_total_bytes",
    "RAW_runtime",
    "POSIX_RAW_agg_perf_by_slowest",
]

fig = plt.figure(figsize=(1.65, 1.3))

ax = plt.gca()
pd.plotting.parallel_coordinates(df[columns + ["empty"]], "empty", ax=ax, linewidth=.3, color='r')

plt.xticks(range(6), [
    "# of processes", 
    "# of accesses", 
    "# of files", 
    "I/O Volume [MiB]", 
    "Runtime [s]", 
    "I/O throughput [MiB/s]"
    ], rotation=30, ha='right')

plt.ylim(1, 5.3)
plt.yticks(range(1, 6), [10**x for x in range(1, 6)])
plt.grid()

plt.savefig("figures/figure_7b.pdf", dpi=600, bbox_inches='tight')


#
# Second graph
#
columns = [
    "POSIX_READS_PERC", 
    "POSIX_BYTES_READ_PERC", 
    "POSIX_read_only_files_perc",
    "POSIX_write_only_files_perc",
    "POSIX_unique_files_perc"
]

fig = plt.figure(figsize=(1.65, 1.3))
ax = plt.gca()
pd.plotting.parallel_coordinates(df[columns + ["empty"]], "empty", ax=ax, linewidth=.5, color='b')

plt.xticks(range(5), [
    "R/W accesses", 
    "R/W bytes", 
    "Read-only files",
    "Write-only files",
    "Unique files", 
    ], rotation=30, ha='right')

plt.yticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"])
plt.grid()

plt.savefig("figures/figure_7a.pdf", dpi=600, bbox_inches='tight')
