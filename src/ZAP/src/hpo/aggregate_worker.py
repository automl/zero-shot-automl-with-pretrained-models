import random
from pathlib import Path

import os
import sys
sys.path.append("../")

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import yaml
from hpbandster.core.worker import Worker
from available_datasets import all_datasets
from src.competition.run_local_test import run_baseline as evaluate_on_dataset
from src.hpo.utils import construct_model_config


def _run_on_dataset(dataset, config_experiment_path, model_config, dataset_dir, n_repeat, time_budget,time_budget_approx):
    experiment_path = config_experiment_path
    dataset_path = Path(dataset_dir, dataset)

    repetition_scores = []
    for _ in range(n_repeat):
        score = evaluate_on_dataset(
            dataset_dir=str(dataset_path),
            code_dir="src",
            experiment_dir=str(experiment_path),
            time_budget=time_budget,
            time_budget_approx=time_budget_approx,
            overwrite=True,
            model_config_name=None,
            model_config=model_config
        )
        repetition_scores.append(score)

    repetition_mean = np.mean(repetition_scores)
    return repetition_scores, repetition_mean


def get_configspace():
    cs = CS.ConfigurationSpace()

    # yapf: disable
    # Dataset sizes
    cv_valid_ratio = CSH.UniformFloatHyperparameter("cv_valid_ratio", lower=0.05, upper=0.2)
    max_valid_count = CSH.UniformIntegerHyperparameter("max_valid_count", lower=128, upper=512, log=True)
    max_size = CSH.UniformIntegerHyperparameter("log2_max_size", lower=5, upper=7)
    # max_times = CSH.UniformIntegerHyperparameter("max_times", lower=4, upper=10)
    train_info_sample = CSH.UniformIntegerHyperparameter("train_info_sample", lower=128, upper=512, log=True)
    # enough_count_video = CSH.UniformIntegerHyperparameter("enough_count_video", lower=100,
    #                                                       upper=10000, log=True)
    # enough_count_image = CSH.UniformIntegerHyperparameter("enough_count_image", lower=1000,
    #                                                       upper=100000, log=True)

    # Report intervalls
    steps_per_epoch = CSH.UniformIntegerHyperparameter("steps_per_epoch", lower=5, upper=250, log=True)
    early_epoch = CSH.UniformIntegerHyperparameter("early_epoch", lower=1, upper=3)
    skip_valid_score_threshold = CSH.UniformFloatHyperparameter("skip_valid_score_threshold", lower=0.7, upper=0.95)
    test_after_at_least_seconds = CSH.UniformIntegerHyperparameter("test_after_at_least_seconds", lower=1, upper=3)
    test_after_at_least_seconds_max = CSH.UniformIntegerHyperparameter("test_after_at_least_seconds_max", lower=60, upper=120)
    test_after_at_least_seconds_step = CSH.UniformIntegerHyperparameter("test_after_at_least_seconds_step", lower=2, upper=10)
    # threshold_valid_score_diff = CSH.UniformFloatHyperparameter("threshold_valid_score_diff",
    #                                                             lower=0.0001, upper=0.01, log=True)
    max_inner_loop_ratio = CSH.UniformFloatHyperparameter("max_inner_loop_ratio", lower=0.1, upper=0.3)

    # Optimization
    batch_size = CSH.UniformIntegerHyperparameter("batch_size", lower=16, upper=64, log=True)
    lr = CSH.UniformFloatHyperparameter('lr', lower=1e-5, upper=1e-1, log=True)
    min_lr = CSH.UniformFloatHyperparameter('min_lr', lower=1e-8, upper=1e-5, log=True)
    wd = CSH.UniformFloatHyperparameter('wd', lower=1e-5, upper=1e-2, log=True)
    momentum = CSH.UniformFloatHyperparameter('momentum', lower=0.01, upper=0.99, log=False)
    optimizer = CSH.CategoricalHyperparameter('optimizer', ['SGD', 'Adam', 'AdamW'])
    nesterov = CSH.CategoricalHyperparameter('nesterov', ['True', 'False'])
    amsgrad = CSH.CategoricalHyperparameter('amsgrad', ['True', 'False'])
    scheduler = CSH.CategoricalHyperparameter('scheduler', ['plateau', 'cosine'])
    freeze_portion = CSH.CategoricalHyperparameter('freeze_portion', list(np.arange(0, 0.6, 0.1)))
    warmup_multiplier = CSH.CategoricalHyperparameter('warmup_multiplier', [1.0, 1.5, 2.0, 2.5, 3.0])
    warm_up_epoch = CSH.UniformIntegerHyperparameter('warm_up_epoch', lower=3, upper=6, log=False)

    # simple classifier
    first_simple_model = CSH.CategoricalHyperparameter('first_simple_model', ['True', 'False'])
    simple_model = CSH.CategoricalHyperparameter('simple_model', ['SVC', 'NuSVC', 'RF', 'LR'])

    # Architecture
    architecture = CSH.CategoricalHyperparameter("architecture", ['ResNet18', 'efficientnetb0', 'efficientnetb1', 'efficientnetb2'])
    #arch_family = CSH.CategoricalHyperparameter("arch_family", ['ResNet', 'EffNet'])
    #TODO: think about if we want to change the way this is designed
    #efficientnet = CSH.CategoricalHyperparameter("efficientnet",
    #                                             ['efficientnetb%d'%x for x in
    #                                             range(2)])
    #resnet = CSH.Constant("resnet", 'ResNet18')
    #condition_1 = CS.EqualsCondition(efficientnet, arch_family, 'EffNet')
    #condition_2 = CS.EqualsCondition(resnet, arch_family, 'ResNet')
    # yapf: enable

    cs.add_hyperparameters(
        [
            cv_valid_ratio, max_valid_count, max_size, train_info_sample, steps_per_epoch,
            early_epoch, skip_valid_score_threshold, test_after_at_least_seconds,
            test_after_at_least_seconds_max, test_after_at_least_seconds_step, max_inner_loop_ratio,
            batch_size, lr, min_lr, architecture, wd, momentum, optimizer, nesterov, amsgrad,
            scheduler, freeze_portion, warmup_multiplier, warm_up_epoch,
            first_simple_model, simple_model
        ]
    )

    condition_1 = CS.EqualsCondition(momentum, optimizer, 'SGD')
    condition_2 = CS.EqualsCondition(nesterov, optimizer, 'SGD')
    condition_3 = CS.OrConjunction(CS.EqualsCondition(amsgrad, optimizer, 'Adam'), CS.EqualsCondition(amsgrad, optimizer, 'AdamW'))
    condition_4 = CS.EqualsCondition(simple_model, first_simple_model, 'True')
    cs.add_conditions([condition_1, condition_2, condition_3, condition_4])
    
    return cs


class SingleWorker(Worker):
    def __init__(self, dataset_parent_dir, config_filename, working_directory, n_repeat, dataset, time_budget, time_budget_approx, **kwargs):
        super().__init__(**kwargs)

        with Path("src/configs/", config_filename).open() as in_stream:
            self._default_config = yaml.safe_load(in_stream)

        self._dataset_dir = dataset_parent_dir
        self._working_directory = working_directory
        self.n_repeat = n_repeat
        self.dataset = dataset
        self.time_budget = time_budget
        self.time_budget_approx = time_budget_approx

    def compute(self, config_id, config, budget, *args, **kwargs):
        config_id_formated = "_".join(map(str, config_id))
        config_experiment_path = Path(self._working_directory, config_id_formated, str(budget))

        model_config = construct_model_config(config, default_config=self._default_config)
        repetition_scores, repetition_mean = _run_on_dataset(
            self.dataset,
            config_experiment_path,
            model_config,
            dataset_dir=self._dataset_dir,
            n_repeat=self.n_repeat,
            time_budget=self.time_budget,
            time_budget_approx=self.time_budget_approx
        )

        info = {
            "{}_repetition_{}_score".format(self.dataset, n): score
            for n, score in enumerate(repetition_scores)
        }

        return ({
            'loss': -repetition_mean,  # remember: HpBandSter always minimizes!
            'info': info
        })


def _normalize(list_, mean, std):
    return [(x - mean) / std for x in list_]


class AggregateWorker(Worker):
    def __init__(
        self,
        dataset_parent_dir,
        config_filename,
        working_directory,
        n_repeat,
        has_repeats_as_budget,
        time_budget,
        time_budget_approx,
        performance_matrix=None,
        **kwargs
    ):
        super().__init__(**kwargs)

        with Path("src/configs/", config_filename).open() as in_stream:
            self._default_config = yaml.safe_load(in_stream)

        if performance_matrix is not None:
            pm = pd.read_csv(performance_matrix, index_col=0)
            self.dataset_to_mean = pm.mean(axis=1).to_dict()
            self.dataset_to_std = pm.std(axis=1).to_dict()
        else:
            self.dataset_to_mean = None
            self.dataset_to_std = None

        self._dataset_dir = dataset_parent_dir
        self._working_directory = working_directory
        self.n_repeat = n_repeat
        self.has_repeats_as_budget = has_repeats_as_budget
        self.time_budget = time_budget
        self.time_budget_approx = time_budget_approx

    def compute(self, config_id, config, budget, *args, **kwargs):
        config_id_formated = "_".join(map(str, config_id))
        config_experiment_path = Path(self._working_directory, config_id_formated, str(budget))

        model_config = construct_model_config(config, default_config=self._default_config)

        if self.has_repeats_as_budget:
            n_repeat = int(budget)
        else:
            n_repeat = self.n_repeat

        per_dataset_score = []
        for dataset in all_datasets:
            try:
                repetition_scores, _ = _run_on_dataset(
                    dataset,
                    config_experiment_path / dataset,
                    model_config,
                    dataset_dir=self._dataset_dir,
                    n_repeat=n_repeat,
                    time_budget=self.time_budget,
                    time_budget_approx=self.time_budget_approx
                )
            except RuntimeError:
                repetition_scores = n_repeat * [0]

            do_normalize = self.dataset_to_mean is not None and self.dataset_to_std is not None
            if do_normalize:
                mean, std = self.dataset_to_mean[dataset], self.dataset_to_std[dataset]
                repetition_scores = _normalize(repetition_scores, mean, std)

            mean_score = sum(repetition_scores) / len(repetition_scores)
            per_dataset_score.append(mean_score)

        mean_score = sum(per_dataset_score) / len(per_dataset_score)
        return ({
            'loss': -mean_score,  # remember: HpBandSter always minimizes!
            'info': None
        })

