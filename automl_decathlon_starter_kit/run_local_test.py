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
python run_local_test.py -dataset_dir=./dev -code_dir=./sample_code_submission/
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
import argparse
import time
import shutil  # for deleting a whole directory
from multiprocessing import Process
from shlex import split
from subprocess import call

logging.basicConfig(
    level=getattr(logging, verbosity_level),
    format="%(asctime)s %(levelname)s %(filename)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def _HERE(*args):
    h = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(h, *args)


def get_path_to_ingestion_program(starting_kit_dir):
    return os.path.join(starting_kit_dir, "ingestion", "ingestion.py")


def get_path_to_scoring_program(starting_kit_dir):
    return os.path.join(starting_kit_dir, "scoring", "score.py")


def remove_dir(output_dir):
    """Remove the directory `output_dir`.

    This aims to clean existing output of last run of local test.
    """
    if os.path.isdir(output_dir):
        logging.info(
            "Cleaning existing output directory of last run: {}".format(output_dir)
        )
        shutil.rmtree(output_dir)


def get_basename(path):
    if len(path) == 0:
        return ""
    if path[-1] == os.sep:
        path = path[:-1]
    return path.split(os.sep)[-1]


def run_baseline(dataset_dir, code_dir, time_budget=1200):
    logging.info("#" * 50)
    logging.info("Begin running local test using")
    logging.info("code_dir = {}".format(get_basename(code_dir)))
    logging.info("dataset_dir = {}".format(get_basename(dataset_dir)))
    logging.info("#" * 50)

    # Current directory containing this script
    starting_kit_dir = os.path.dirname(os.path.realpath(__file__))
    path_ingestion = get_path_to_ingestion_program(starting_kit_dir)
    path_scoring = get_path_to_scoring_program(starting_kit_dir)

    # Run ingestion and scoring at the same time
    command_ingestion = f"python {path_ingestion} --dataset_dir={dataset_dir} --code_dir={code_dir} --time_budget={time_budget}"
    command_scoring = f"python {path_scoring} --dataset_dir={dataset_dir}"

    ingestion_output_dir = os.path.join(starting_kit_dir, "sample_result_submission")
    score_dir = os.path.join(starting_kit_dir, "scoring_output")

    # Clear previous outputs
    remove_dir(ingestion_output_dir)
    remove_dir(score_dir)

    cmd_ing = split(command_ingestion)
    cmd_sco = split(command_scoring)

    call(cmd_ing)
    call(cmd_sco)


if __name__ == "__main__":
    default_starting_kit_dir = _HERE()
    # The default dataset is 'miniciao' under the folder sample_data/
    default_dataset_dir = os.path.join(default_starting_kit_dir, "dev")
    default_code_dir = os.path.join(default_starting_kit_dir, "sample_code_submission")
    default_time_budget = 1200
    default_dataset_name = "ninapro"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_dir",
        type=str,
        help="Directory storing the dataset (containing " + "e.g. ninapro/)",
    )
    parser.add_argument(
        "--code_dir",
        type=str,
        help="Directory containing a `model.py` file. Specify this "
        "argument if you want to test on a different algorithm.",
    )
    parser.add_argument(
        "--time_budget",
        type=float,
        default=default_time_budget,
        help="Directory storing the ingestion program "
        + "`ingestion.py` and other necessary packages.",
    )

    args = parser.parse_args()
    dataset_dir = args.dataset_dir
    code_dir = args.code_dir
    time_budget = args.time_budget

    run_baseline(dataset_dir, code_dir, time_budget)
