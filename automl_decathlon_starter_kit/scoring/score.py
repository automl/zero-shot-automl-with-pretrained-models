################################################################################
# Usage:        python score.py --dataset_dir=<dataset_dir> --prediction_dir=<prediction_dir> --score_dir=<score_dir>
#           prediction_dir should contain e.g. start.txt, adult.predict_0, adult.predict_1,..., end.txt.
#           score_dir should contain scores.txt, detailed_results.html

VERSION = "v20220710"
DESCRIPTION = """This is the scoring program for the AutoML Decathlon. It takes the predictions made by ingestion program as input and compare to the solution file and produces scores."""

# Scoring program for the AutoML Decathlon.

################################################################################
# User defined constants
################################################################################

# Verbosity level of logging.
# Can be: NOTSET, DEBUG, INFO, WARNING, ERROR, CRITICAL
verbosity_level = "DEBUG"

from libscores import read_array, mkdir, ls, mvmean, tiedrank, _HERE, get_logger
from os.path import join
from sys import argv
from sklearn.metrics import auc
import argparse
import base64
import datetime
import decathlon_metrics
import matplotlib

matplotlib.use("Agg")  # Solve the Tkinter display issue
import matplotlib.pyplot as plt
import numpy as np
import os
import psutil
import sys
import time
import yaml
from random import randrange

import os
import sys

score_dir = os.path.dirname(os.path.abspath(__file__))
# sys.path.insert(0, os.path.join(score_dir, "..", "ingestion"))
from dev_datasets import DecathlonDataset, OHE
from fsd50kutils.audio_dataset import _collate_fn_eval
from torch.utils.data import Dataset, DataLoader


class IngestionNotEndedError(Exception):
    pass


logger = get_logger(verbosity_level)

################################################################################
# Functions
################################################################################

# Metric used to compute the score


def decathlon_scorer(solution, prediction, dataset_name):
    score = None
    if dataset_name in [
        "ninapro",
        "spherical",
        "ember",
    ]:
        # Report 0-1 error
        score = decathlon_metrics.zero_one_error(solution, prediction)
    elif dataset_name == "cosmic":
        # Report false negative rate
        score = decathlon_metrics.false_negative_rate(solution, prediction)
    elif dataset_name == "deepsea":
        # Report 1 - AUROC
        score = decathlon_metrics.inv_auroc_score(solution, prediction)
    elif dataset_name == "ecg":
        # Report 1 - F1 score
        score = decathlon_metrics.inv_f1_score(solution, prediction)
    elif dataset_name == "fsd50k":
        # Report 1 - MAP
        score = decathlon_metrics.inv_map_score(solution, prediction)
    elif dataset_name in ["navierstokes", "crypto"]:
        # Report L2 relative error
        score = decathlon_metrics.l2_relative_error(
            solution.reshape(prediction.shape[0], -1),
            prediction.reshape(prediction.shape[0], -1),
        )
    elif dataset_name == "nottingham":
        # Report negative log likelihood
        score = decathlon_metrics.nll_score(solution, prediction)
    else:
        raise NotImplementedError
    return score


scoring_functions = {"metric": decathlon_scorer}


def get_valid_columns(solution):
    """Get a list of column indices for which the column has more than one class.
    This is necessary when computing BAC or AUC which involves true positive and
    true negative in the denominator. When some class is missing, these scores
    don't make sense (or you have to add an epsilon to remedy the situation).

    Args:
      solution: array, a matrix of binary entries, of shape
        (num_examples, num_features)
    Returns:
      valid_columns: a list of indices for which the column has more than one
        class.
    """
    num_examples = solution.shape[0]
    col_sum = np.sum(solution, axis=0)
    valid_columns = np.where(
        1 - np.isclose(col_sum, 0) - np.isclose(col_sum, num_examples)
    )[0]
    return valid_columns


def is_one_hot_vector(x, axis=None, keepdims=False):
    """Check if a vector 'x' is one-hot (i.e. one entry is 1 and others 0)."""
    norm_1 = np.linalg.norm(x, ord=1, axis=axis, keepdims=keepdims)
    norm_inf = np.linalg.norm(x, ord=np.inf, axis=axis, keepdims=keepdims)
    return np.logical_and(norm_1 == 1, norm_inf == 1)


def is_multiclass(solution):
    """Return if a task is a multi-class classification task, i.e.  each example
    only has one label and thus each binary vector in `solution` only has
    one '1' and all the rest components are '0'.

    This function is useful when we want to compute metrics (e.g. accuracy) that
    are only applicable for multi-class task (and not for multi-label task).

    Args:
      solution: a numpy.ndarray object of shape [num_examples, num_classes].
    """
    return all(is_one_hot_vector(solution, axis=1))


def get_solution(dataset_dir, task_name): 
    """
    CHANGES:
    dataset_dir: the directory where all the dev data is held; by the default structure, this would be something like '/{home}/automl_decathlon_starting_kit/dev/processed_data/'
    """

    data_dir = os.path.join(dataset_dir, "processed_data")
    if task_name == "ninapro":
        solution = np.load(os.path.join(data_dir, "ninapro", "y_test.npy")).astype(
            "int"
        )
        solution = OHE(solution, 18)
    elif task_name == "spherical":
        solution = np.load(os.path.join(data_dir, "spherical", "y_test.npy")).astype(
            "int"
        )
        solution = OHE(solution, 100)
    elif task_name == "ecg":
        solution = np.load(os.path.join(data_dir, "ecg", "y_test.npy")).astype("int")
        solution = OHE(solution, 4)
    elif task_name == "deepsea":
        solution = np.load(os.path.join(data_dir, "deepsea", "y_test.npy"))
    elif task_name == "navierstokes":
        solution = np.load(os.path.join(data_dir, "navierstokes", "y_test.npy"))
    elif task_name == "ember":
        solution = np.load(os.path.join(data_dir, "ember", "y_test.npy")).astype("int")
        solution = OHE(solution, 2)
    elif task_name == "crypto":
        solution = np.load(os.path.join(data_dir, "crypto", "y_test.npy"))
    elif task_name == "nottingham" or task_name == "cosmic" or task_name == "fsd50k":
        """
        For these tasks there is processing done on the y vectors, so we create the dataloaders and form the batches into a final solution matrix
        """
        dataset = DecathlonDataset(
            task=task_name, root=os.path.join(data_dir, ".."), split="test"
        )
        if task_name == "fsd50k":
            dataloader = DataLoader(
                dataset, batch_size=64, shuffle=False, collate_fn=_collate_fn_eval
            )
        else:
            dataloader = DataLoader(dataset, batch_size=256, shuffle=False)

        batches = []
        # if task_name == "cosmic":
        #    for _, y, _ in dataloader:
        #        batches.append(y.detach().numpy())
        # else:
        for _, y in dataloader:
            batches.append(y.detach().numpy())
        solution = np.concatenate(batches, axis=0)

    else:
        raise NotImplementedError
        # TODO: add all the other task options

    logger.info("solution shape={}".format(solution.shape))
    return solution


def get_task_name(prediction_dir):
    """Get the task name from prediction directory."""
    pred_file = ls(os.path.join(prediction_dir, "*.predict"))
    if len(pred_file) != 1:  # Assert only one file is found
        logger.warning(
            "{} solution files found: {}! ".format(len(pred_file), pred_file)
            + "Return `None` as task name."
        )
        return None

    task_name = pred_file[0].split(os.sep)[-1].split(".")[0]
    return task_name


def auc_step(X, Y):
    """Compute area under curve using step function (in 'post' mode)."""
    if len(X) != len(Y):
        raise ValueError(
            "The length of X and Y should be equal but got "
            + "{} and {} !".format(len(X), len(Y))
        )
    area = 0
    for i in range(len(X) - 1):
        delta_X = X[i + 1] - X[i]
        area += delta_X * Y[i]
    return area


def get_ingestion_info(prediction_dir, filename):
    """Get info on ingestion program: PID, start time, etc. from 'start.txt'.

    Args:
      prediction_dir: a string, directory containing predictions (output of
        ingestion)
    Returns:
      A dictionary with keys 'ingestion_pid' and 'start_time' if the file
        'start.txt' exists. Otherwise return `None`.
    """
    start_filepath = os.path.join(prediction_dir, filename)
    if os.path.exists(start_filepath):
        with open(start_filepath, "r") as f:
            ingestion_info = yaml.safe_load(f)
        return ingestion_info
    else:
        return None


def compute_scores_bootstrap(scoring_function, solution, prediction, n=10):
    """Compute a list of scores using bootstrap.

    Args:
      scoring function: scoring metric taking y_true and y_pred
      solution: ground truth vector
      prediction: proposed solution
      n: number of scores to compute
    """
    scores = []
    l = len(solution)
    for _ in range(n):  # number of scoring
        size = solution.shape[0]
        idx = np.random.randint(0, size, size)  # bootstrap index
        scores.append(scoring_function(solution[idx], prediction[idx]))
    return scores


def end_file_generated(prediction_dir):
    """Check if ingestion is still alive by checking if the file 'end.txt'
    is generated in the folder of predictions.
    """
    end_filepath = os.path.join(prediction_dir, "end.txt")
    logger.debug("CPU usage: {}%".format(psutil.cpu_percent()))
    logger.debug("Virtual memory: {}".format(psutil.virtual_memory()))
    return os.path.isfile(end_filepath)


def is_process_alive(pid):
    """Check if a process is alive according to its PID."""
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    else:
        return True


def terminate_process(pid):
    """Kill a process according to its PID."""
    process = psutil.Process(pid)
    process.terminate()
    logger.debug("Terminated process with pid={} in scoring.".format(pid))


class IngestionError(Exception):
    pass


class ScoringError(Exception):
    pass


class Evaluator(object):
    def __init__(
        self,
        dataset_dir=None,  # edit, see comment at start of __init__()
        prediction_dir=None,
        score_dir=None,
        scoring_functions=None,
        task_name=None,
        participant_name=None,
        algorithm_name=None,
        submission_id=None,
    ):
        """
        Args:
          data_dir: the directory where all the dev data is held; by the default structure, this would be something lik'/{root}/automl_decathlon_starting_kit/dev/processed_data/'
          scoring_functions: a dict containing (string, scoring_function) pairs
        """
        self.start_time = time.time()

        self.dataset_dir = dataset_dir
        self.prediction_dir = prediction_dir
        self.score_dir = score_dir
        self.scoring_functions = scoring_functions

        self.task_name = task_name or get_task_name(prediction_dir)
        self.participant_name = participant_name
        self.algorithm_name = algorithm_name
        self.submission_id = submission_id

        # State variables
        self.scoring_success = None
        self.time_limit_exceeded = None
        self.prediction_files_so_far = []
        self.new_prediction_files = []
        self.score_per_task = {}
        self.score = -1.0
        self.scores_so_far = {"nauc": []}
        self.relative_timestamps = []

        # Resolve info from directories
        self.solution = self.get_solution() 
        # Check if the task is multilabel (i.e. with one hot label)

        self.fetch_ingestion_info()

    def get_solution(self):
        """Get solution as NumPy array from `self.data_dir`."""
        solution = get_solution(self.dataset_dir, self.task_name)
        logger.debug(
            "Successfully loaded solution from dataset_dir={}".format(self.dataset_dir)
        )
        return solution

    def fetch_ingestion_info(self):
        """Resolve some information from output of ingestion program. This includes
        especially: `ingestion_start`, `ingestion_pid`, `time_budget`.

        Raises:
          IngestionError if no sign of ingestion starting detected after 1800
          seconds.
        """
        logger.debug("Fetching ingestion info...")
        prediction_dir = self.prediction_dir
        # Wait 1800 seconds for ingestion to start and write 'start.txt',
        # Otherwise, raise an exception.

        ingestion_info = get_ingestion_info(prediction_dir, "start.txt")
        if ingestion_info is None:
            raise IngestionError("[-] Failed: scoring didn't detected start.txt")

        # Get ingestion start time
        ingestion_start = ingestion_info["start_time"]
        # Get ingestion PID
        ingestion_pid = ingestion_info["ingestion_pid"]
        # Get time_budget for ingestion
        assert "time_budget" in ingestion_info
        time_budget = ingestion_info["time_budget"]
        # Set attributes
        self.ingestion_info = ingestion_info
        self.ingestion_start = ingestion_start
        self.ingestion_pid = ingestion_pid
        self.time_budget = time_budget

        ingestion_info = get_ingestion_info(prediction_dir, "end.txt")
        if ingestion_info is None:
            raise IngestionError("[-] Failed: scoring didn't detected end.txt")
        self.ingestion_success = ingestion_info["ingestion_success"]
        self.ingestion_duration = ingestion_info["ingestion_duration"]
        self.ingestion_end = ingestion_info["end_time"]

        logger.debug("Ingestion start time: {}".format(self.ingestion_start))
        logger.debug("Ingestion end time: {}".format(self.ingestion_end))
        logger.debug("Ingestion duration: {}".format(self.ingestion_duration))
        logger.debug("Scoring start time: {}".format(self.start_time))
        logger.debug("Ingestion info successfully fetched.")

    def end_file_generated(self):
        return end_file_generated(self.prediction_dir)

    def ingestion_is_alive(self):
        return is_process_alive(self.ingestion_pid)

    def kill_ingestion(self):
        terminate_process(self.ingestion_pid)
        assert not self.ingestion_is_alive()

    def prediction_filename_pattern(self):
        return "{}.predict_*".format(self.task_name)

    def prediction_file(self):
        return os.path.join(self.prediction_dir, "{}.predict".format(self.task_name))

    def compute_score_per_task(self):
        pred = read_array(self.prediction_file())
        score = decathlon_scorer(self.solution, pred, self.task_name)
        self.score_per_task[self.task_name] = score
        self.score = score


# =============================== MAIN ========================================


def scoring_main(args, task_name):
    dataset_dir = args.dataset_dir
    prediction_dir = args.prediction_dir
    score_dir = args.score_dir
    mkdir(score_dir)

    logger.debug("Version: {}. Description: {}".format(VERSION, DESCRIPTION))
    logger.debug("Using dataset_dir: " + str(dataset_dir))
    logger.debug("Using prediction_dir: " + str(prediction_dir))
    logger.debug("Using score_dir: " + str(score_dir))

    #################################################################
    # Initialize an evaluator (scoring program) object
    evaluator = Evaluator(
        dataset_dir,
        prediction_dir,
        score_dir,
        task_name=task_name,
        scoring_functions=scoring_functions,
    )
    #################################################################

    scoring_start = evaluator.start_time
    evaluator.compute_score_per_task()

    if evaluator.ingestion_success == 0:
        logger.error(
            "[-] Some error occurred in ingestion program. "
            + "Please see output/error log of Ingestion Step."
        )
    else:
        logger.info(
            "[+] Successfully finished scoring! "
            + "Scoring duration: {:.2f} sec. ".format(time.time() - scoring_start)
            + "Ingestion duration: {:.2f} sec. ".format(evaluator.ingestion_duration)
            + "The score of your algorithm on the task '{}' is: {:.6f}.".format(
                evaluator.task_name, evaluator.score
            )
        )

    logger.info("[Scoring terminated]")
    return evaluator.score, evaluator.ingestion_duration


if __name__ == "__main__":
    logger.info(
        "=" * 5 + " Start scoring program. " + "Version: {} ".format(VERSION) + "=" * 5
    )

    # Default I/O directories:
    root_dir = _HERE(os.pardir)
    default_dataset_dir = join(root_dir, "dev")
    default_prediction_dir = join(root_dir, "sample_result_submission")
    default_score_dir = join(root_dir, "scoring_output")

    # Parse directories from input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=default_dataset_dir,
        help="Directory storing the solution with true " + "labels, e.g. Y_test.npy.",
    )
    parser.add_argument(
        "--prediction_dir",
        type=str,
        default=default_prediction_dir,
        help="Directory storing the predictions. It should"
        + "contain e.g. [start.txt, adult.predict_0, "
        + "adult.predict_1, ..., adult.predict, end.txt].",
    )
    parser.add_argument(
        "--score_dir",
        type=str,
        default=default_score_dir,
        help="Directory storing the scoring output "
        + "e.g. `scores.txt` and `detailed_results.html`.",
    )
    args = parser.parse_args()
    logger.debug("Parsed args are: " + str(args))
    logger.debug("-" * 50)

    base_prediction_dir = args.prediction_dir
    all_tasks = [
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
    tasks = [x for x in os.listdir(base_prediction_dir) if x in all_tasks]
    tasks_not_completed = set(all_tasks) - set(tasks)
    logger.info("Found prediction directories for tasks: {}".format(" ".join(tasks)))

    for task in all_tasks:
        if task in set(tasks):
            logger.info("Start scoring for task: {}".format(task))
            args.prediction_dir = os.path.join(base_prediction_dir, task)
            score, duration = scoring_main(args, task)
        else:
            logger.info(
                "Task was not completed (not included in run or timed out): {}".format(
                    task
                )
            )
            mkdir(args.score_dir)
            score, duration = (
                999999,
                999999,
            )  # these are just placeholders (higher is worse).

        # Write results to file
        score_dir = args.score_dir
        score_filename = os.path.join(score_dir, "scores.txt")
        with open(score_filename, "a") as f:
            f.write(f"score_{task}: " + str(score) + "\n")
            f.write(f"duration_{task}: " + str(duration) + "\n")
        logger.debug(
            "Wrote to score_filename={} with score={}, duration={}".format(
                score_filename, score, duration
            )
        )
        logger.info("Ended scoring for task: {}".format(task))

    aupp = 0.0  # True aupp calculation is done offline in the notebook
    with open(score_filename, "a") as f:
        f.write(f"aupp: " + str(aupp) + "\n")
    logger.debug(
        "Wrote to score_filename={} with AUPP score={}".format(score_filename, aupp)
    )
