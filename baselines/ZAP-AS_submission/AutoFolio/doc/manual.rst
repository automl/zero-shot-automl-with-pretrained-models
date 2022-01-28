Manual
======
.. role:: bash(code)
    :language: bash


In the following we will show how to use **AutFolio**.

.. _quick:

Quick Start
-----------
| If you have not installed *AutFolio* yet take a look at the `installation instructions <installation.html>`_ and make sure that all the requirements are fulfilled.
| In the examples folder, you can find examples that illustrate how to reads scenario files that allow you to automatically configure an algorithm, as well as examples that show how to directly use *AutFolio* in Python.

We will demonstrate the usage of *AutFolio* on a simple toy example, see `examples/toy_example_csv`.

To run the example, change into the root-directory of *AutoFolio* and type the following commands:

.. code-block:: bash

    cd examples/toy_example_csv/
    python ../../scripts/autofolio --perf perf.csv --feature_csv feats.csv 
     

AutoFolio will run a 10-fold cross validation on the given data.
The `perf.csv` file is a csv file where each column corresponds to an algorithm
and each row to an instance. Each entry is the performance of an algorithm on a given instance.
The `feats.csv` is a csv file where each column corresponds an instance feature
and each row to an instance.
Per default, AutoFolio assumes that we want to minimize the performance as solution cost metric (in contrast to a runtime metric).

In the end, AutoFolio prints the aggregated performance across the 10-folds.

.. code-block:: bash

	INFO:AutoFolio:>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
	INFO:AutoFolio:CV Stats
	INFO:Stats:Number of instances: 10
	INFO:Stats:Average Solution Quality: 6.0000

Looking into `perf.csv`, we can see that AutoFolio performs quite poorly on this example.
The better of the two algorithms in `perf.csv` has a mean performance of 3.9.
The issue is that two default of the hyperparameters of the random forest (`rf:min_samples_leaf` and `rf:bootstrap`)
are a bad choice. 

So far, AutoFolio used only its default parameters.
To automatically optimize its parameters use the arguments `-t, --tune`, e.g.,

.. code-block:: bash

    cd examples/toy_example_csv/
    python ../../scripts/autofolio --perf perf.csv --feature_csv feats.csv -t
    
In most cases, AutoFolio should be able to figure out that the previously mentioned parameters have to be changed
such that AutoFolio can get a better performance.

.. code-block:: bash

	INFO:AutoFolio:>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
	INFO:AutoFolio:CV Stats
	INFO:Stats:Number of instances: 10
	INFO:Stats:Average Solution Quality: 2.0000

Input
-----

AutoFolio can read two input formats:
i) A simple interface for beginners based on two csv files (i.e., one for the performance of each algorithm on each instance
and one for the instance features on each instance).
ii) An advanced interface based on the ASlib format specification.

CSV Input Format
----------------

The csv input format consists of two files:

	1. A performance file, where each column corresponds to an algorithm and each row to an instance.
	2. A feature file, where each column corresponds to an instance feature and each row to an instance.
	
See `examples/toy_examples_csv` for a trivial example
and `examples/ttp` for a complex example.

Furthermore, you can specify 3 how to interprete the performance file:

.. code-block:: bash

  --objective {runtime,solution_quality}
                        Are the objective values in the performance data
                        runtimes or an arbitrary solution quality (or cost)
                        value (default: solution_quality)
  --runtime_cutoff RUNTIME_CUTOFF
                        cutoff time for each algorithm run for the performance
                        data (default: None)
  --maximize            Set this parameter to indicate maximization of the
                        performance metric (default: minimization) (default:
                        False)

ASlib Input format
------------------

The ASlib Input format is more complex
but also is more flexible and allows to express more complex scenarios.
See <http://www.aslib.net> for examples and a formal specification.
`examples/asp-aslib` also provides a complex scenario as an example in this format.

To use this format, please use

.. code-block:: bash

  -s SCENARIO, --scenario SCENARIO
                        directory with ASlib scenario files (required if not
                        using --load or csv input files (default: None)

Modes
-----

Cross-Validation Mode
---------------------

The default mode of AutoFolio is running a 10-fold cross validation to estimate the performance of AutFolio.

Prediction Mode
---------------

If you want to use AutoFolio to predict for instances not represented in the given data,
you need to train AutoFolio save its internal state to disk (use `python3 scripts/autofolio --save [filename]`).
To predict on a new instance,
please run

.. code-block:: bash

	python3 scripts/autofolio --load [filename] --feature_vec [space-separated feature vector]

Self-Tuning Mode
----------------

To use algorithm configuration to optimize the performance of AutoFolio please use the option `--tune`. 




