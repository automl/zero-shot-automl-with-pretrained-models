# Traveling Thief Problem Example

This is an example to use AutoFolio on the Traveling Thief Problem using the csv format.
In `data/features.csv`, you find the instance features for each problem instance;
and in `data/perf.csv`, you find the performance of each algorithm on each problem instance.

By calling:

`python3 ../../scripts/autofolio --performance_csv data/perf.csv --feature_csv data/features.csv --maximize`

AutoFolio will perform a 10-fold cross validation on selecting the algorithm with the largest performance value (`--maximize`) for each given instance. We expect a performance of roughly 0.99.



