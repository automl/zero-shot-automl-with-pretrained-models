# AutoFolio

AutoFolio is an algorithm selection tool,
i.e., selecting a well-performing algorithm for a given instance [Rice 1976].
In contrast to other algorithm selection tools,
users of AutoFolio are bothered with the decision which algorithm selection approach to use
and how to set its hyper-parameters.
AutoFolio uses one of the state-of-the-art algorithm configuration tools, namely SMAC [Hutter et al LION'16]
to automatically determine a well-performing algorithm selection approach
and its hyper-parameters for a given algorithm selection data.
Therefore, AutoFolio has a robust performance across different algorithm selection tasks.

## Version

This package is a re-implementation of the original AutoFolio.
It follows the same approach as the original AutoFolio
but it has some crucial differences:

* instead of SMAC v2, we use the pure Python implementation of SMAC (v3)
* less implemented algorithm selection approaches -- focus on promising approaches to waste not unnecessary time during configuration
* support of solution quality scenarios

## License

This program is free software: you can redistribute it and/or modify it under the terms of the 2-clause BSD license (please see the LICENSE file).
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
You should have received a copy of the 2-clause BSD license along with this program (see LICENSE file). If not, see https://opensource.org/licenses/BSD-2-Clause.

## Installation

### Requirements

NOTE: AutoFolio requires the future SMAC 0.9; currently only available in the [development branch of SMAC](https://github.com/automl/SMAC3/tree/development)

AutoFolio runs with '''Python 3.5'''.

To install (nearly) all requirements, please run:

`cat requirements.txt | xargs -n 1 -L 1 pip install`

Many of its dependencies can be fulfilled by using [Anaconda >3.4](https://www.continuum.io/).
If you use Anaconda as your Python environment, you have to install three packages before you can install SMAC (as one of AutoFolio's requirements):

`conda install gxx_linux-64 gcc_linux-64 swig`

To use pre-solving schedules, [clingo](http://potassco.sourceforge.net/) is required. We provide binary compiled under Ubuntu 14.04 which may not work under another OS. Please put a working `clingo` binary with Python support into the folder `aspeed/`.
 
## Usage

We provide under `scripts` a command-line interface for AutoFolio.
To get an overview over all options of AutoFolio, simply run:

`python3 scripts/autofolio --help`

We provide some examples in `examples/`

### Input Formats 

AutoFolio reads two input formats: CSV and [ASlib](www.aslib.net).
The CSV format is easier for new users but has some limitations to express all kind of input data.
The ASlib format has a higher expressiveness -- please see [www.aslib.net](www.aslib.net) for all details on this input format.

For the CSV format, simply two files are required.
One file with the performance data of each algorithm on each instance (each row an instance, and each column an algorithm).
And another file with the instance features for each instance (each row an instance and each column an feature).
All other meta-data (such as runtime cutoff) has to be specified by command line options (see `python3 scripts/autofolio --help`).

### Configuration file

A YAML configuration file can be given to control some of the internal AutoFolio
behavior. It is given with the `--config` option. 

The recognized options and their types are as follows.

* `wallclock_limit`. The amount of time (in seconds) for optimizing 
  hyperparameters. Type: integer. Default: 300 seconds --- should be increased!
  
#### Feature groups
  
* `allowed_feature_groups`. A list of the feature groups to consider for 
  prediction. This must match those specified in the ASlib scenario. Type: list
  of strings. Default: all feature sets are allowed.

#### Preprocessing

* `pca`. Whether to include PCA as a choice for preprocessing. Type: Boolean. Default: True.

* `impute`. Whether missing value imputation is a choice for preprocessing. Type: Boolean. Default: True.

* `scale`. Whether z-score scaling is a choice for preprocessing. Type: Boolean. Default: True.

#### Presolving

* `presolve`. Whether to use a presolver. Type: Boolean. Default: True.

#### Algorithm selection model classes

* `random_forest_classifier`. Whether the random forest classifier is a model class choice. Type: Boolean. Default: True.

* `xgboost_classifier`. Whether the XGBoost classifier is a model class choice. Type: Boolean. Default: True.

* `random_forest_regressor`. Whether the random forest regressor is a model class choice. Type: Boolean. Default: True.

### Cross-Validation Mode

The default mode of AutoFolio is running a 10-fold cross-validation to estimate the performance of AutoFolio.

### "Outer" Cross-Validation Mode

"Outer" cross-validation again uses a 10-fold cross-validation scheme to
evaluate AutoFolio; in this case, though, the subset for testing is not at all
seen by AutoFolio during training. Internally, the nine training folds are
further use in an "inner" cross-validation to avoid overfitting.

The `--outer-cv` flag indicates to use this mode. For example:

```
python3 scripts/autofolio -s examples/asp-aslib/data/ --outer-cv

```
#### Saving the outer cross-validation choices

The learned model and solver choices for each instance can be saved using the
`--out-template` option. If given, the fit model and solver choices will be
saved to this location. The string is considered a template. "${fold}" will be 
replaced with the outer cv fold, and "${type}" will be replaced with the 
appropriate file extension, "pkl" for the models and "csv" for the solver 
choices. See string.Template for more details about valid tempaltes.

**N.B.** In many shells (such as bash), it is necessary to put the template in 
single quotes to avoid shell replacement in the template. (Double quotes will
not typically work.)

```
python3 scripts/autofolio -s examples/asp-aslib/data/ --outer-cv --out-template 'asp.fold-${fold}.${type}'

```
#### Parallelizing the outer cross-validation

Optionally, only a single "outer" cv fold can be run. Presumably, this is used
to parallelize the outer cv calls across a cluster. The `--outer-cv-fold` option
specifies which fold is used. Typically, this option would be combined with
`--out-template`, and the results would be combined in post-processing.

**N.B.** This number should range from 1 to 10 (not 0 to 9).

```
python3 scripts/autofolio -s examples/asp-aslib/data/ --outer-cv --outer-cv-fold 1 --out-template 'asp.fold-${fold}.${type}'
```


### Prediction Mode

If you want to use AutoFolio to predict for instances not represented in the given data,
you need to train AutoFolio save its internal state to disk (use `python3 scripts/autofolio --save [filename]`).
To predict on a new instance,
please run

`python3 scripts/autofolio --load [filename] --feature_vec "[space-separated feature vector]"`

Please note that the quotes around the feature vector are important.

### Self-Tuning Mode

To use algorithm configuration to optimize the performance of AutoFolio please use the option `--tune`. 

## Reference

[JAIR Journal Article](http://aad.informatik.uni-freiburg.de/papers/15-JAIR-Autofolio.pdf)

@ARTICLE{lindauer-jair15a,
  author    = {M. Lindauer and H. Hoos and F. Hutter and T. Schaub},
  title     = {AutoFolio: An automatically configured Algorithm Selector},
  volume    = {53},
  journal   = {Journal of Artificial Intelligence Research},
  year      = {2015},
  pages     = {745-778}
}

## Contact

Marius Lindauer: lindauer@cs.uni-freiburg.de
