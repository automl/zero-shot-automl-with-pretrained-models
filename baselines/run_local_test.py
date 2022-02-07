################################################################################
# Name:         Run Local Test Tool
# Author:       Zhengying Liu
# Created on:   20 Sep 2018
# Update time:  5 May 2019
# Usage: 		    python run_local_test.py -dataset_dir=<dataset_dir> -code_dir=<code_dir>

VERISION = "v20190505"
DESCRIPTION =\
"""This script allows participants to run local test of their method within the
downloaded starting kit folder (and avoid using submission quota on CodaLab). To
do this, run:
```
python run_local_test.py -dataset_dir=./AutoDL_sample_data/miniciao -code_dir=./AutoDL_sample_code_submission/
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
verbosity_level = 'INFO'

import logging
import os
import tensorflow as tf
import time
import shutil # for deleting a whole directory
import webbrowser
from multiprocessing import Process

logging.basicConfig(
    level=getattr(logging, verbosity_level),
    format='%(asctime)s %(levelname)s %(filename)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def _HERE(*args):
    h = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(h, *args)

def get_path_to_ingestion_program(starting_kit_dir):
  return os.path.join(starting_kit_dir,
                      'AutoDL_ingestion_program', 'ingestion.py')

def get_path_to_scoring_program(starting_kit_dir):
  return os.path.join(starting_kit_dir,
                      'AutoDL_scoring_program', 'score.py')

def remove_dir(output_dir):
  """Remove the directory `output_dir`.

  This aims to clean existing output of last run of local test.
  """
  if os.path.isdir(output_dir):
    logging.info("Cleaning existing output directory of last run: {}"\
                .format(output_dir))
    shutil.rmtree(output_dir)

def get_basename(path):
  if len(path) == 0:
    return ""
  if path[-1] == os.sep:
    path = path[:-1]
  return path.split(os.sep)[-1]

def run_baseline(dataset_dir, code_dir, result_dir, time_budget=1200):
  logging.info("#"*50)
  logging.info("Begin running local test using")
  logging.info("code_dir = {}".format(get_basename(code_dir)))
  logging.info("dataset_dir = {}".format(get_basename(dataset_dir)))
  logging.info("#"*50)

  # Current directory containing this script
  starting_kit_dir = os.path.dirname(os.path.realpath(__file__))
  path_ingestion = get_path_to_ingestion_program(starting_kit_dir)
  path_scoring = get_path_to_scoring_program(starting_kit_dir)

  path_res_ingestion = os.path.join(result_dir, "ingestion")
  path_res_scoring = os.path.join(result_dir, "score")
  os.makedirs(path_res_ingestion, exist_ok = True)
  os.makedirs(path_res_scoring, exist_ok = True)
  
  # Run ingestion and scoring at the same time
  command_ingestion =\
    "python {} --dataset_dir={} --output_dir={} --code_dir={} --score_dir={} --time_budget={}"\
    .format(path_ingestion, dataset_dir, path_res_ingestion, code_dir, path_res_scoring, time_budget)
  command_scoring =\
    'python {} --solution_dir={} --prediction_dir={} --score_dir={}'\
    .format(path_scoring, dataset_dir, path_res_ingestion, path_res_scoring)
  def run_ingestion():
    print(command_ingestion)
    exit_code = os.system(command_ingestion)
    assert exit_code == 0
  def run_scoring():
    print(command_scoring)
    exit_code = os.system(command_scoring)
    assert exit_code == 0
  ingestion_process = Process(name='ingestion', target=run_ingestion)
  scoring_process = Process(name='scoring', target=run_scoring)
  
  ingestion_output_dir = os.path.join(starting_kit_dir, result_dir, "ingestion")
  score_dir = os.path.join(starting_kit_dir, result_dir, "score")
  remove_dir(ingestion_output_dir)
  remove_dir(score_dir)
  ingestion_process.start()
  scoring_process.start()
  
  ingestion_process.join()
  scoring_process.join()
  if not ingestion_process.exitcode == 0:
    logging.info("Some error occurred in ingestion program.")
  if not scoring_process.exitcode == 0:
    raise Exception("Some error occurred in scoring program.")

if __name__ == '__main__':
  default_starting_kit_dir = _HERE()
  # The default dataset is 'miniciao' under the folder AutoDL_sample_data/
  default_dataset_dir = os.path.join(default_starting_kit_dir,
                                     'AutoDL_sample_data', 'miniciao')
  default_code_dir = os.path.join(default_starting_kit_dir,
                                     'AutoDL_sample_code_submission')

  default_result_dir = os.path.join(default_starting_kit_dir,
                                     'AutoDL_scoring_output')
  default_time_budget = 1200

  tf.flags.DEFINE_string('dataset_dir', default_dataset_dir,
                        "Directory containing the content (e.g. adult.data/ + "
                        "adult.solution) of an AutoDL dataset. Specify this "
                        "argument if you want to test on a different dataset.")

  tf.flags.DEFINE_string('code_dir', default_code_dir,
                        "Directory containing a `model.py` file. Specify this "
                        "argument if you want to test on a different algorithm."
                        )

  tf.flags.DEFINE_string('result_dir', default_result_dir,
                        "Directory containing scoring outputs"
                        )

  tf.flags.DEFINE_float('time_budget', default_time_budget,
                        "Time budget for running ingestion " +
                        "(training + prediction)."
                        )

  FLAGS = tf.flags.FLAGS
  dataset_dir = FLAGS.dataset_dir
  code_dir = FLAGS.code_dir
  result_dir = FLAGS.result_dir
  time_budget = FLAGS.time_budget

  run_baseline(dataset_dir, code_dir, result_dir, time_budget)
