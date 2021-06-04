# Getting the data

Since the data is too large to be stored in the repo, we have to fetch it first. To do so, run:
```
cd data/
wget ...
```

# Installation
We assume that the Linux machine already has `virtualenv` installed.
If not, consult [this link](https://virtualenv.pypa.io/en/latest/installation.html).

First, setup the virtual environment and install dependencies with
```
virtualenv venv -p python3.6
source venv/bin/activate 
pip install -r requirements 
```

# Running the experiments
All of the experiments can be ran with just e.g.,:
```
python -m experiments.figure_1a
```
Just change which figure you want to generate by replacing `figure_1a` with the name of any other figure.
Do note how the path is specified ("." instead of the separator "/" and no ".py" file extension).

After running the scripts, an output PDF should be produced in the `figures/` directory.


# Replication notes:
Some of the produced figures may differ slightly from those in the paper:

* Figure 1d in the paper was produced on the NERSC Cori dataset, but as we can only share the ALCF Theta dataset, we are reproducing it with Theta data. The figure does not significantly change, and the interpretation is the same - error is low when predicting on data drawn from the same time period as the training set (up to January 1st, 2020), and error grows after January 2020, since training set does not cover 2020.
* Figure 4 was also produced using the NERSC Cori dataset, since Cori has LMT logs. Because Theta doesn't have LMT logs, we are able to reproduce only the 1st and 3rd distribution.

# TODO
* Check whether Figure 3 still works 
