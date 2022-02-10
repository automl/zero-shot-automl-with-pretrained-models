import argparse
import logging
import os
import random
import sys
import time
from pathlib import Path

sys.path.append(os.getcwd())

import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres
import numpy as np
import tensorflow as tf
import torch
from hpbandster.optimizers import BOHB as BOHB

from src.hpo.aggregate_worker import AggregateWorker, SingleWorker, get_configspace

def run_master(args):
    NS = hpns.NameServer(run_id=args.run_id, nic_name=args.nic_name, working_directory=args.bohb_root_path)
    ns_host, ns_port = NS.start()

    w = SingleWorker(
        run_id=args.run_id,
        host=ns_host,
        nameserver=ns_host,
        nameserver_port=ns_port,
        dataset_parent_dir = args.dataset_root,
        config_filename = args.config_filename,
        working_directory=args.bohb_root_path,
        n_repeat=args.n_repeat,
        dataset=args.dataset,
        time_budget=args.time_budget,
        time_budget_approx=args.time_budget_approx
    )
    w.run(background=True)

    # Create an optimizer
    result_logger = hpres.json_result_logger(directory=args.bohb_root_path, overwrite=False)

    if args.previous_run_dir is not None:
        previous_result = hpres.logged_results_to_HBS_result(args.previous_run_dir)
    else:
        previous_result = None

    logger = logging.getLogger(__file__)
    logging_level = getattr(logging, args.logger_level)
    logger.setLevel(logging_level)

    logger.info(args)

    optimizer = BOHB(
        configspace=get_configspace(),
        run_id=args.run_id,
        host=ns_host,
        nameserver=ns_host,
        nameserver_port=ns_port,
        min_budget=args.n_repeat_lower_budget,
        max_budget=args.n_repeat_upper_budget,
        result_logger=result_logger,
        logger=logger,
        previous_result=previous_result
    )

    res = optimizer.run(n_iterations=args.n_iterations)

    # Shutdown
    optimizer.shutdown(shutdown_workers=True)
    NS.shutdown()


def main(args):
    args.run_id = args.job_id or args.experiment_name
    args.host = hpns.nic_name_to_host(args.nic_name)

    args.bohb_root_path = str(Path(args.experiment_group_dir, args.experiment_name)) 
    args.dataset = args.experiment_name

    # Handle case of budget dictating n_repeat vs. n_repeat directly
    if args.n_repeat_lower_budget is not None and args.n_repeat_upper_budget is not None:
        args.n_repeat = None
    else:
        args.n_repeat_lower_budget = 1
        args.n_repeat_upper_budget = 1

    # Set previous run dir path
    if args.previous_run_dir is not None:
    	args.previous_run_dir = str(Path(args.previous_run_dir, (args.experiment_name).split("/")[-1]))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    tf.set_random_seed(args.seed)

    run_master(args)


if __name__ == '__main__':
    p = argparse.ArgumentParser("", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # fmt: off
    p.add_argument("--dataset_root", default="../../../data/datasets")
    p.add_argument("--experiment_group_dir", default = "../../../data/per_icgen_augmentation_hpo", help = "The BOHB result dir")
    p.add_argument("--experiment_name", help="The dataset path under the dataset root. e.g 0/cifar100/")
    p.add_argument("--config_filename", default="default.yaml")

    p.add_argument("--n_repeat_lower_budget", type=int, default=None, help="Overrides n_repeat")
    p.add_argument("--n_repeat_upper_budget", type=int, default=None, help="")
    p.add_argument("--n_repeat", type=int, default=3, help="Number of evaluations per BOHB iteration")
    p.add_argument("--n_iterations", type=int, default=100, help="Number of evaluations per BOHB run")

    p.add_argument("--job_id", default=None)
    p.add_argument("--seed", type=int, default=2, help="random seed")
    p.add_argument("--nic_name", default="eth0", help="The network interface to use")
    p.add_argument("--previous_run_dir", default=None, help="Path to a previous run to warmstart from")

    p.add_argument("--time_budget_approx",
                type=int,
                default=300,
                help="Specifies <lower_time> to simulate cutting a run with budget <actual_time> after <lower-time> seconds.")
    p.add_argument("--time_budget",
                type=int,
                default=1200,
                help="Specifies <actual_time> (see argument --time_budget_approx")
    p.add_argument("--logger_level",
                type=str,
                default="DEBUG",
                help= "Sets the logger level. Choose from ['INFO', 'DEBUG', 'NOTSET', 'WARNING', 'ERROR', 'CRITICAL']")

    args = p.parse_args()
    main(args)
