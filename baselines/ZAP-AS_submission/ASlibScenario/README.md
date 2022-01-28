# ASlibScenario

`aslib_scenario` is a package in Python to load and store data of a scenario of the Algorithm Selection Library (ASlib).

## License

This program is free software: you can redistribute it and/or modify it under the terms of the 2-clause BSD license (please see the LICENSE file).
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
You should have received a copy of the 2-clause BSD license along with this program (see LICENSE file). If not, see https://opensource.org/licenses/BSD-2-Clause.

## Installation

### Requirements

ASlibScenario runs with '''Python 3.5'''.
Many of its dependencies can be fulfilled by using [Anaconda 3.4](https://www.continuum.io/).  

To install all requirements, please run:

`pip install -r requirements.txt`

## Usage

### Input Formats 

ASlibScenario reads two input formats: CSV and [ASlib](www.aslib.net).
The CSV format is easier for new users but has some limitations to express all kind of input data.
The ASlib format has a higher expressiveness -- please see [www.aslib.net](www.aslib.net) for all details on this input format.

For the CSV format, simply two files are required.
One file with the performance data of each algorithm on each instance (each row an instance, and each column an algorithm).
And another file with the instance features for each instance (each row an instance and each column an feature).

See [AutoFolio](https://github.com/mlindauer/AutoFolio) for examples how to use this package.


## Important Implementation Notes

### Handling of Repetitions

If a scenario is provided in the ASlib format, you can specify repeated measurements (e.g., of algorithm runs).
This package will aggregate repeated measurements into a single number:

    * Algorithm Run Data: using the median performance across all repetitions; the runstatus is determined by the most frequent status
    * Feature Values: using mean across all repetitions
    * Feature Group Runstatus: most frequent status
    * Feature Cost: using median across all repetitions
    * CV: only saving the first repetition

### Handling of Non-OK Runs

If the runstatus of an algorithm run is not "ok"
and the objective is to optimize "runtime",
we impute the corresponding runs with a PAR10 value (10 x running time cutoff).

### Handling of Maximization Scenarios

If the objective is to __maximize__ solution quality,
we multiply the algorithm performance data with -1,
since we assume that we always minimize. 

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
