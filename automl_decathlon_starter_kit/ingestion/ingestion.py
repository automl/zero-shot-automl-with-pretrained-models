################################################################################
# Name:         Ingestion Program
# Author:       Zhengying Liu, Isabelle Guyon, Adrien Pavao, Zhen Xu
# Update time:  07 Jul 2022
# Usage: python ingestion.py --dataset_dir=<dataset_dir> --output_dir=<prediction_dir> --ingestion_program_dir=<ingestion_program_dir> --code_dir=<code_dir>

# AS A PARTICIPANT, DO NOT MODIFY THIS CODE.

VERSION = "v20220707"
DESCRIPTION = """This is the "ingestion program" written by the organizers. It takes the
code written by participants (with `model.py`) and one dataset as input,
run the code on the dataset and produce predictions on test set. For more
information on the code/directory structure, please see comments in this
code (ingestion.py) and the README file of the starting kit.
"""
# The dataset directory dataset_dir (e.g. sample_data/) contains one dataset
# folder (e.g. ninapro) with the training set and test set. The directory structure
# will look like
#
#   dev
#   ├── md ### contains metadata info for each dataset
#   │   ├── ninapro
#   │       │── train_metadata.json
#   │       └── test_metadata.json
#   └── processed_data ### contains actual data in .npy files for each dataset
#       ├── ninapro
#           │── x_train.npy
#           │── y_train.npy
#           │── x_test.npy
#           └── y_test.npy
#
# The output directory output_dir (e.g. sample_result_submission/)
# will first have a start.txt file written by ingestion then receive
# all predictions made during the whole train/predict process
# (thus this directory is updated when a new prediction is made):
#   ninapro.predict_0
#   ninapro.predict_1
#   ninapro.predict_2
#        ...
# after ingestion has finished, a final prediction file ninapro.predict and
# end.txt will be written, containing info on the duration ingestion used.
# Both files are used for scoring
#
# The code directory submission_program_dir (e.g. sample_code_submission/)
# should contain your code submission model.py (and possibly other functions
# it depends upon).
#
# We implemented several classes:
# 1) DATA LOADING:
#    ------------
# dev_datasets.py
# dataset.DecathlonMetadata: Read metadata in .json metadata files
# dataset.DecathlonDataset: Read data in .npy files
# 2) LEARNING MACHINE:
#    ----------------
# model.py
# model.Model.train
# model.Model.test
#

# =========================== BEGIN OPTIONS ==============================

# Verbosity level of logging:
##############
# Can be: NOTSET, DEBUG, INFO, WARNING, ERROR, CRITICAL
verbosity_level = "DEBUG"

# Some common useful packages
from contextlib import contextmanager
from os.path import join
from sys import path
import argparse
import logging
import math
import numpy as np
import os
import sys
import signal
import time
import yaml

from dev_datasets import extract_metadata

import data_io


def get_logger(verbosity_level, use_error_log=False):
    """Set logging format to something like:
    2019-04-25 12:52:51,924 INFO score.py: <message>
    """
    logger = logging.getLogger(__file__)
    logging_level = getattr(logging, verbosity_level)
    logger.setLevel(logging_level)
    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(filename)s: %(message)s"
    )
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging_level)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)
    if use_error_log:
        stderr_handler = logging.StreamHandler(sys.stderr)
        stderr_handler.setLevel(logging.WARNING)
        stderr_handler.setFormatter(formatter)
        logger.addHandler(stderr_handler)
    logger.propagate = False
    return logger


logger = get_logger(verbosity_level)


def get_time_budget(dataset_name):
    # TODO
    hours = 60 * 60
    if dataset_name == "navierstokes":
        return 5 * hours
    elif dataset_name == "spherical":
        return 5 * hours
    elif dataset_name == "ninapro":
        return 5 * hours
    elif dataset_name == "fsd50k":
        return 20 * hours
    elif dataset_name == "cosmic":
        return 5 * hours
    elif dataset_name == "ecg":
        return 10 * hours
    elif dataset_name == "deepsea":
        return 5 * hours
    elif dataset_name == "nottingham":
        return 5 * hours
    elif dataset_name == "crypto":
        return 5 * hours
    elif dataset_name == "ember":
        return 20 * hours
    else:
        return 5 * hours


def _HERE(*args):
    """Helper function for getting the current directory of this script."""
    h = os.path.dirname(os.path.realpath(__file__))
    return os.path.abspath(os.path.join(h, *args))


def write_start_file(output_dir, start_time=None, time_budget=None, task_name=None):
    """Create start file 'start.txt' in `output_dir` with ingestion's pid and
    start time.

    The content of this file will be similar to:
        ingestion_pid: 1
        task_name: beatriz
        time_budget: 7200
        start_time: 1557923830.3012087
        0: 1557923854.504741
        1: 1557923860.091236
        2: 1557923865.9630117
        3: 1557923872.3627956
        <more timestamps of predictions>
    """
    ingestion_pid = os.getpid()
    start_filename = "start.txt"
    start_filepath = os.path.join(output_dir, start_filename)
    with open(start_filepath, "w") as f:
        f.write("ingestion_pid: {}\n".format(ingestion_pid))
        f.write("task_name: {}\n".format(task_name))
        f.write("time_budget: {}\n".format(time_budget))
        f.write("start_time: {}\n".format(start_time))
    logger.debug("Finished writing 'start.txt' file.")


def write_timestamp(output_dir, predict_idx, timestamp):
    start_filename = "start.txt"
    start_filepath = os.path.join(output_dir, start_filename)
    with open(start_filepath, "a") as f:
        f.write("{}: {}\n".format(predict_idx, timestamp))
    logger.debug(
        "Wrote timestamp {} to 'start.txt' for prediction {}.".format(
            timestamp, predict_idx
        )
    )


class ModelApiError(Exception):
    pass


class BadPredictionShapeError(Exception):
    pass


class NoPredictionError(Exception):
    pass


class TimeoutException(Exception):
    pass


class Timer:
    def __init__(self):
        self.duration = 0
        self.total = None
        self.remain = None
        self.exec = None

    def set(self, time_budget):
        self.total = time_budget
        self.remain = time_budget
        self.exec = 0

    @contextmanager
    def time_limit(self, pname):
        def signal_handler(signum, frame):
            raise TimeoutException("Timed out!")

        signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(int(math.ceil(self.remain)))
        start_time = time.time()

        try:
            yield
        finally:
            exec_time = time.time() - start_time
            signal.alarm(0)
            self.exec += exec_time
            self.duration += exec_time
            self.remain = self.total - self.exec

            logger.info("{} success, time spent so far {} sec".format(pname, self.exec))

            if self.remain <= 0:
                raise TimeoutException("Timed out for the process: {}!".format(pname))


# =========================== BEGIN PROGRAM ================================


def ingestion_main(ingestion_success, args, dataset_name):
    logger.debug("Parsed args are: " + str(args))
    logger.debug("-" * 50)
    dataset_dir = args.dataset_dir
    output_dir = args.output_dir
    ingestion_program_dir = args.ingestion_program_dir
    code_dir = args.code_dir
    if args.time_budget is None:
        time_budget = get_time_budget(dataset_name)
        logger.debug("Task-specified time_budget: {}".format(time_budget))
    else:
        time_budget = args.time_budget
        logger.debug("User-specified time_budget: {}".format(time_budget))

    # Show directories for debugging
    logger.debug("sys.argv = " + str(sys.argv))
    logger.debug("Using dataset_dir: " + dataset_dir)
    logger.debug("Using output_dir: " + output_dir)
    logger.debug("Using ingestion_program_dir: " + ingestion_program_dir)
    logger.debug("Using code_dir: " + code_dir)

    # Our libraries
    path.append(ingestion_program_dir)
    path.append(code_dir)
    # IG: to allow submitting the starting kit as sample submission
    path.append(code_dir + "/test_model")

    from dev_datasets import DecathlonDataset

    data_io.mkdir(output_dir)

    basename = dataset_name

    logger.info("************************************************")
    logger.info("******** Processing dataset " + basename.capitalize() + " ********")
    logger.info("************************************************")
    logger.debug("Version: {}. Description: {}".format(VERSION, DESCRIPTION))

    ##### Begin creating training set and test set #####
    logger.info("Reading {} training set and test set...".format(basename))
    D_train = DecathlonDataset(basename, dataset_dir, "train")
    D_test = DecathlonDataset(basename, dataset_dir, "test")
    D_val = (
        DecathlonDataset(basename, dataset_dir, "val") if basename == "fsd50k" else None
    )
    logger.info("Created training and test datasets")
    ##### End creating training set and test set #####

    ## Get the metadata for model.train()
    train_metadata = extract_metadata(D_train)
    dataset_name = train_metadata.get_dataset_name()
    num_examples_train = train_metadata.size()

    ## Get correct prediction shape
    test_metadata = extract_metadata(D_test)
    num_examples_test = test_metadata.size()  # scalar int
    output_dim = np.prod(test_metadata.get_output_shape())  # tuple
    correct_prediction_shape = (num_examples_test, output_dim) 

    ## if fsd50k, get corresponding validation metadata
    val_metadata = extract_metadata(D_val) if D_val else None

    logger.info("Creating model...")
    from model import Model  # in participants' model.py

    M = Model(
        train_metadata
    )  # The metadata of D_train and D_test only differ in sample_count

    # Mark starting time of ingestion
    start = time.time()
    logger.info(
        "=" * 5
        + " Start core part of ingestion program. "
        + "Version: {} ".format(VERSION)
        + "=" * 5
    )

    write_start_file(
        output_dir,
        start_time=start,
        time_budget=time_budget,
        task_name=basename.split(".")[0],
    )

    """try:"""
    # Check if the model has methods `train` and `test`.
    for attr in ["train", "test"]:
        if not hasattr(M, attr):
            raise ModelApiError(
                "Your model object doesn't have the method "
                + "`{}`. Please implement it in model.py."
            )

    # Keeping track of how many predictions are made
    prediction_order_number = 0

    # Start the CORE PART: train/predict process
    remaining_time_budget = start + time_budget - time.time()

    # Train the model
    logger.info("Begin training the model...")
    M.train(
        D_train,
        D_val,
        val_metadata,
        remaining_time_budget=remaining_time_budget,
    )
    logger.info("Finished training the model.")
    remaining_time_budget = start + time_budget - time.time()

    # Make predictions using the trained model
    logger.info("Begin testing the model by making predictions " + "on test set...")
    Y_pred = M.test(D_test, remaining_time_budget=remaining_time_budget)
    logger.info("Finished making predictions.")

    # Check if the prediction has good shape
    prediction_shape = tuple(Y_pred.shape)
    if prediction_shape[1:] != correct_prediction_shape[1:]:
        raise BadPredictionShapeError(
            "Bad prediction shape! Expected {} but got {}.".format(
                correct_prediction_shape, prediction_shape
            )
        )
    # Write timestamp to 'start.txt'
    write_timestamp(
        output_dir, predict_idx=prediction_order_number, timestamp=time.time()
    )
    # Prediction files: ninapro.predict_0, ninapro.predict_1, ...
    filename_test = basename + ".predict_" + str(prediction_order_number)
    # Write predictions to output_dir
    data_io.write(os.path.join(output_dir, filename_test), Y_pred)
    prediction_order_number += 1
    logger.info(
        "[+] {0:d} predictions made, time spent so far {1:.2f} sec".format(
            prediction_order_number, time.time() - start
        )
    )
    remaining_time_budget = start + time_budget - time.time()
    logger.info("[+] Time left {0:.2f} sec".format(remaining_time_budget))
    """
    except Exception as e:
        ingestion_success = False
        logger.info("Failed to run ingestion.")
        logger.error("Encountered exception:\n" + str(e), exc_info=True)
    """
    ### write final prediction result, will be used to calculate the score
    filename_test = basename + ".predict"
    if Y_pred is None:
        raise NoPredictionError(
            "No prediction was generated. "
            + "Terminate ingestion program. Scoring won’t run into success"
        )
    data_io.write(os.path.join(output_dir, filename_test), Y_pred)

    # Finishing ingestion program
    end_time = time.time()
    overall_time_spent = end_time - start

    # Write overall_time_spent to a end.txt file
    end_filename = "end.txt"
    with open(os.path.join(output_dir, end_filename), "w") as f:
        f.write("ingestion_duration: " + str(overall_time_spent) + "\n")
        f.write("ingestion_success: " + str(int(ingestion_success)) + "\n")
        f.write("end_time: " + str(end_time) + "\n")
        logger.info(
            "Wrote the file {} marking the end of ingestion.".format(end_filename)
        )
        if ingestion_success:
            logger.info("[+] Done. Ingestion program successfully terminated.")
            logger.info("[+] Overall time spent %5.2f sec " % overall_time_spent)
        else:
            logger.info("[-] Done, but encountered some errors during ingestion.")
            logger.info("[-] Overall time spent %5.2f sec " % overall_time_spent)

    logger.info("[Ingestion terminated]")


if __name__ == "__main__":

    #### Check whether everything went well
    ingestion_success = True

    # Parse directories from input arguments
    root_dir = _HERE(os.pardir)
    default_dataset_dir = join(root_dir, "dev_public")
    default_output_dir = join(root_dir, "sample_result_submission")
    default_ingestion_program_dir = join(root_dir, "ingestion")
    default_code_dir = join(root_dir, "sample_code_submission")

    default_time_budget = 1200
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=default_dataset_dir,
        help="Directory storing the dataset (containing " + "e.g. ninapro)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=default_output_dir,
        help="Directory storing the predictions. It will "
        + "contain e.g. [start.txt, ninapro.predict_0, "
        + "ninapro.predict_1, ..., ninapro.predict end.txt] "
        + "when ingestion terminates.",
    )
    parser.add_argument(
        "--ingestion_program_dir",
        type=str,
        default=default_ingestion_program_dir,
        help="Directory storing the ingestion program "
        + "`ingestion.py` and other necessary packages.",
    )
    parser.add_argument(
        "--code_dir",
        type=str,
        default=default_code_dir,
        help="Directory storing the submission code "
        + "`model.py` and other necessary packages.",
    )
    parser.add_argument(
        "--time_budget",
        type=float,
        help="Time budget for running ingestion program.",
    )
    args = parser.parse_args()

    task_yaml_file = os.path.join(args.code_dir, "tasks_to_run.yaml")

    if os.path.exists(task_yaml_file):
        with open(task_yaml_file, "r") as f:
            content = yaml.load(f, Loader=yaml.FullLoader)
        tasks = content["tasks"]
        logger.info("Found user-specified task list: {}".format(" ".join(tasks)))
    else:
        tasks = [
            "navierstokes",
            "spherical",
            "ninapro",
            "cosmic",
            "ecg",
            "deepsea",
            "nottingham",
            "crypto",
            "ember",
            "fsd50k",
        ]
        logger.info("Default task list: {}".format(" ".join(tasks)))

    base_output_dir = args.output_dir

    for task in tasks:
        logger.info("Starting ingestion for {}".format(task))
        args.output_dir = os.path.join(base_output_dir, task)

        try:
            time_budget = (
                args.time_budget
                if args.time_budget is not None
                else get_time_budget(task)
            )
            timer = Timer()
            timer.set(time_budget)
            with timer.time_limit("Ingestion"):
                ##### Begin creating model #####
                logger.info(
                    "Starting ingestion for {}, this has a time constraint of {} s.".format(
                        task, time_budget
                    )
                )
                ingestion_main(ingestion_success, args, task)
        except TimeoutException as e:
            logger.info(
                "Ingestion timed out on {}; will remove prediction directory".format(
                    task
                )
            )
            data_io.rmdir(args.output_dir)

        except Exception as e:
            logger.error("Ingestion failed!")
            logger.error("Encountered exception:\n" + str(e), exc_info=True)
            data_io.rmdir(args.output_dir)

        logger.info("Ended ingestion for {}".format(task))
