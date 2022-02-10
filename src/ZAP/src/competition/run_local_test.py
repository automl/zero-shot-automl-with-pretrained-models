################################################################################
# Name:         Run Local Test Tool
# Author:       Zhengying Liu
# Created on:   20 Sep 2018
# Update time:  5 May 2019
# Usage: 		    python run_local_test.py -dataset_dir=<dataset_dir> -code_dir=<code_dir>

VERISION = "v20190505"
DESCRIPTION = """This script allows participants to run local test of their method within the
downloaded starting kit folder (and avoid using submission quota on CodaLab). To
do this, run:
```
python run_local_test.py -dataset_dir=./sample_data/miniciao -code_dir=./sample_code_submission/
```
in the starting kit directory. If you want to test the performance of a
different algorithm on a different dataset, please specify them using respective
arguments.

If you want to use default folders (i.e. those in above command line), simply
run
```
python run_local_test.py
```
"""

# ALL INFORMATION, SOFTWARE, DOCUMENTATION, AND DATA ARE PROVIDED "AS-IS".
# ISABELLE GUYON, CHALEARN, AND/OR OTHER ORGANIZERS OR CODE AUTHORS DISCLAIM
# ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR ANY PARTICULAR PURPOSE, AND THE
# WARRANTY OF NON-INFRINGEMENT OF ANY THIRD PARTY'S INTELLECTUAL PROPERTY RIGHTS.
# IN NO EVENT SHALL ISABELLE GUYON AND/OR OTHER ORGANIZERS BE LIABLE FOR ANY SPECIAL,
# INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER ARISING OUT OF OR IN
# CONNECTION WITH THE USE OR PERFORMANCE OF SOFTWARE, DOCUMENTS, MATERIALS,
# PUBLICATIONS, OR INFORMATION MADE AVAILABLE FOR THE CHALLENGE.
################################################################################

# Verbosity level of logging.
# Can be: NOTSET, DEBUG, INFO, WARNING, ERROR, CRITICAL
verbosity_level = "INFO"

import logging
import os
import sys
sys.path.append(os.getcwd())

import shutil  # for deleting a whole directory
import time
import webbrowser
from multiprocessing import Process
from pathlib import Path

import tensorflow as tf
from src.competition.ingestion_program.ingestion import ingestion_fn
from src.competition.scoring_program.score import score_fn

logging.basicConfig(
    level=getattr(logging, verbosity_level),
    format="%(asctime)s %(levelname)s %(filename)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def _HERE(*args):
    h = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(h, *args)


def get_path_to_ingestion_program(starting_kit_dir):
    return os.path.join(starting_kit_dir, "ingestion_program", "ingestion.py")


def get_path_to_scoring_program(starting_kit_dir):
    return os.path.join(starting_kit_dir, "scoring_program", "score.py")


def remove_dir(output_dir):
    """Remove the directory `output_dir`.

  This aims to clean existing output of last run of local test.
  """
    if os.path.isdir(output_dir):
        logging.info("Cleaning existing output directory of last run: {}".format(output_dir))
        shutil.rmtree(output_dir)


def get_basename(path):
    if len(path) == 0:
        return ""
    if path[-1] == os.sep:
        path = path[:-1]
    return path.split(os.sep)[-1]


def run_baseline(
    dataset_dir,
    code_dir,
    experiment_dir,
    time_budget,
    time_budget_approx,
    overwrite,
    model_config_name=None,
    model_config=None
):
    logging.info("#" * 50)
    logging.info("Begin running local test using")
    logging.info("code_dir = {}".format(get_basename(code_dir)))
    logging.info("dataset_dir = {}".format(get_basename(dataset_dir)))
    logging.info("#" * 50)

    # Current directory containing this script
    starting_kit_dir = os.path.dirname(os.path.realpath(__file__))
    path_ingestion = get_path_to_ingestion_program(starting_kit_dir)

    ingestion_output_dir = "{}/predictions".format(experiment_dir)
    score_dir = "{}/score".format(experiment_dir)

    os.makedirs(experiment_dir, exist_ok=overwrite)
    os.makedirs(score_dir, exist_ok=overwrite)
    remove_dir(ingestion_output_dir)
    #remove_dir(score_dir)

    ingestion_fn(
        dataset_dir,
        code_dir,
        time_budget,
        time_budget_approx,
        ingestion_output_dir,
        score_dir,
        model_config_name=model_config_name,
        model_config=model_config
    )
    return score_fn(dataset_dir, ingestion_output_dir, score_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    default_dataset_dir = os.path.join(_HERE(), "sample_data", "miniciao")
    parser.add_argument("--dataset_dir", default=default_dataset_dir, help=" ")
    parser.add_argument("--experiment_main_dir", default = "/work/dlclarge2/ozturk-experiments")
    parser.add_argument("--experiment_group", default="test", help=" ")
    parser.add_argument("--experiment_name", default="default", help=" ")
    parser.add_argument("--model_config_name", default="default.yaml", help="The config in src/configs to use")
    parser.add_argument("--code_dir", default="src", help=" ")
    parser.add_argument("--time_budget", type=int, default=1200, help=" ")
    parser.add_argument("--time_budget_approx", type=int, default=600, help=" ")
    parser.add_argument("--overwrite", action="store_true", help="Do not delete submission dir")

    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    code_dir = args.code_dir
    time_budget = args.time_budget
    time_budget_approx = args.time_budget_approx
    overwrite = args.overwrite
    model_config_name = args.model_config_name
    experiment_dir = str(Path(args.experiment_main_dir, args.experiment_group, args.experiment_name))

    run_baseline(dataset_dir, code_dir, experiment_dir, time_budget, time_budget_approx, overwrite, model_config_name)