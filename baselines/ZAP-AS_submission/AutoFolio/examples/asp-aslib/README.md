# Traveling Thief Problem Example

This is an example to use AutoFolio on the Answer Set Programming using the ASlib format.
Here, we use the data of the original `ASP-POTASSCO` ASlib scenario.
For all full description of the format, please see www.aslib.net.

By calling:

`python3 ../../scripts/autofolio --scenario data/`

AutoFolio will perform a 10-fold cross validation on selecting the algorithm with the smallest runtime (see `data/description.txt`) for each given instance. We expect a performance of roughly 135.

To get a better performance, please use the option `--tune`.



