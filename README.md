

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


