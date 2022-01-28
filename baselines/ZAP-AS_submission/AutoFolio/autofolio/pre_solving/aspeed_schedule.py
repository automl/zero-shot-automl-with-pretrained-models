import os
import sys
import logging
import math

import numpy as np
import pandas as pd
import subprocess

from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter
from ConfigSpace.conditions import EqualsCondition, InCondition
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace import Configuration

from ASlibScenario.aslib_scenario.aslib_scenario import ASlibScenario

__author__ = "Marius Lindauer"
__license__ = "BSD"


class Aspeed(object):

    @staticmethod
    def add_params(cs: ConfigurationSpace, cutoff: int):
        '''
            adds parameters to ConfigurationSpace

            Arguments
            ---------
            cs: ConfigurationSpace
                configuration space to add new parameters and conditions
            cutoff: int
                maximal possible time for aspeed
        '''

        pre_solving = CategoricalHyperparameter(
            "presolving", choices=[True, False], default_value=False)
        cs.add_hyperparameter(pre_solving)
        pre_cutoff = UniformIntegerHyperparameter(
            "pre:cutoff", lower=1, upper=cutoff, default_value=math.ceil(cutoff * 0.1), log=True)
        cs.add_hyperparameter(pre_cutoff)
        cond = InCondition(child=pre_cutoff, parent=pre_solving, values=[True])
        cs.add_condition(cond)

    def __init__(self, clingo: str=None, runsolver: str=None, enc_fn: str=None):
        '''
            Constructor

            Arguments
            ---------
            clingo: str
                path to clingo binary
            runsolver: str
                path to runsolver binary
            enc_fn: str
                path to encoding file name
        '''
        self.logger = logging.getLogger("Aspeed")

        if not runsolver:
            self.runsolver = os.path.join(
                os.path.dirname(sys.argv[0]), "..", "aspeed", "runsolver")
        else:
            self.runsolver = runsolver
        if not clingo:
            self.clingo = os.path.join(
                os.path.dirname(sys.argv[0]), "..", "aspeed", "clingo")
        else:
            self.clingo = clingo
        if not enc_fn:
            self.enc_fn = os.path.join(
                os.path.dirname(sys.argv[0]), "..", "aspeed", "enc1.lp")
        else:
            self.enc_fn = enc_fn

        self.mem_limit = 2000  # mb
        self.cutoff = 60

        self.data_threshold = 300  # minimal number of instances to use
        self.data_fraction = 0.3  # fraction of instances to use

        self.schedule = []

    def fit(self, scenario: ASlibScenario, config: Configuration):
        '''
            fit pca object to ASlib scenario data

            Arguments
            ---------
            scenario: data.aslib_scenario.ASlibScenario
                ASlib Scenario with all data in pandas
            config: ConfigSpace.Configuration
                configuration
            classifier_class: selector.classifier.*
                class for classification
        '''

        if config["presolving"]:
            self.logger.info("Compute Presolving Schedule with Aspeed")

            X = scenario.performance_data.values

            # if the instance set is too large, we subsample it
            if X.shape[0] > self.data_threshold:
                random_indx = np.random.choice(
                    range(X.shape[0]),
                    size=min(X.shape[0], max(int(X.shape[0] * self.data_fraction), self.data_threshold)), 
                    replace=True)
                X = X[random_indx, :]

            self.logger.debug("#Instances for pre-solving schedule: %d" %(X.shape[0]))
            times = ["time(i%d, %d, %d)." % (i, j, max(1,math.ceil(X[i, j])))
                     for i in range(X.shape[0]) for j in range(X.shape[1])]

            kappa = "kappa(%d)." % (config["pre:cutoff"])

            data_in = " ".join(times) + " " + kappa

            # call aspeed and save schedule
            self._call_clingo(data_in=data_in, algorithms=scenario.performance_data.columns)

    def _call_clingo(self, data_in: str, algorithms: list):
        '''
            call clingo on self.enc_fn and facts from data_in

            Arguments
            ---------
            data_in: str
                facts in format time(I,A,T) and kappa(C)
            algorithms: list
                list of algorithm names
        '''
        cmd = "%s -C %d -M %d -w /dev/null %s %s -" % (
            self.runsolver, self.cutoff, self.mem_limit, self.clingo, self.enc_fn)

        self.logger.info("Call: %s" % (cmd))

        p = subprocess.Popen(cmd,
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, shell=True, universal_newlines=True)
        stdout, stderr = p.communicate(input=data_in)
        
        self.logger.debug(stdout)
        
        schedule_dict = {}
        for line in stdout.split("\n"):
            if line.startswith("slice"):
                schedule_dict = {}  # reinitizalize for every found schedule
                slices_str = line.split(" ")
                for slice in slices_str:
                    s_tuple = slice.replace("slice(", "").rstrip(")").split(",")
                    algo = algorithms[int(s_tuple[1])]
                    budget = int(s_tuple[2])
                    schedule_dict[algo] = budget
        
        self.schedule = sorted(schedule_dict.items(), key=lambda x: x[1])
        
        self.logger.info("Fitted Schedule: %s" % (self.schedule))
        
    def predict(self, scenario: ASlibScenario):
        '''
            transform ASLib scenario data

            Arguments
            ---------
            scenario: data.aslib_scenario.ASlibScenario
                ASlib Scenario with all data in pandas

            Returns
            -------
                schedule:{inst -> (solver, time)}
                    schedule of solvers with a running time budget
        '''

        return dict((inst, self.schedule) for inst in scenario.instances)
