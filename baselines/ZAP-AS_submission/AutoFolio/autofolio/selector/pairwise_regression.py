import logging
import traceback

import numpy as np
import pandas as pd

from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter
from ConfigSpace.conditions import EqualsCondition, InCondition
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace import Configuration

from ASlibScenario.aslib_scenario.aslib_scenario import ASlibScenario

__author__ = "Marius Lindauer"
__license__ = "BSD"


class PairwiseRegression(object):

    @staticmethod
    def add_params(cs: ConfigurationSpace):
        '''
            adds parameters to ConfigurationSpace 
        '''

        selector = cs.get_hyperparameter("selector")
        regressor = cs.get_hyperparameter("regressor")
        if "PairwiseRegressor" in selector.choices:
            cond = InCondition(child=regressor, parent=selector, values=["PairwiseRegressor"])
            cs.add_condition(cond)

    def __init__(self, regressor_class):
        '''
            Constructor
        '''
        self.regressors = []
        self.logger = logging.getLogger("PairwiseRegressor")
        self.regressor_class = regressor_class

    def fit(self, scenario: ASlibScenario, config: Configuration):
        '''
            fit pca object to ASlib scenario data

            Arguments
            ---------
            scenario: data.aslib_scenario.ASlibScenario
                ASlib Scenario with all data in pandas
            config: ConfigSpace.Configuration
                configuration
        '''
        self.logger.info("Fit PairwiseRegressor with %s" %
                         (self.regressor_class))

        self.algorithms = scenario.algorithms

        n_algos = len(scenario.algorithms)
        X = scenario.feature_data.values
        for i in range(n_algos):
            for j in range(i + 1, n_algos):
                y_i = scenario.performance_data[scenario.algorithms[i]].values
                y_j = scenario.performance_data[scenario.algorithms[j]].values
                y = y_i - y_j
                reg = self.regressor_class()
                reg.fit(X, y, config)
                self.regressors.append(reg)

    def predict(self, scenario: ASlibScenario):
        '''
            predict schedules for all instances in ASLib scenario data

            Arguments
            ---------
            scenario: data.aslib_scenario.ASlibScenario
                ASlib Scenario with all data in pandas

            Returns
            -------
                schedule: {inst -> (solver, time)}
                    schedule of solvers with a running time budget
        '''

        if scenario.algorithm_cutoff_time:
            cutoff = scenario.algorithm_cutoff_time
        else:
            cutoff = 2**31

        n_algos = len(scenario.algorithms)
        X = scenario.feature_data.values
        scores = np.zeros((X.shape[0], n_algos))
        reg_indx = 0
        for i in range(n_algos):
            for j in range(i + 1, n_algos):
                reg = self.regressors[reg_indx]
                Y = reg.predict(X)
                scores[:, i] += Y
                scores[:, j] += -1 * Y
                reg_indx += 1

        #self.logger.debug(
        #   sorted(list(zip(scenario.algorithms, scores)), key=lambda x: x[1], reverse=True))
        algo_indx = np.argmin(scores, axis=1)
        
        schedules = dict((str(inst),[s]) for s,inst in zip([(scenario.algorithms[i], cutoff+1) for i in algo_indx], scenario.feature_data.index))
        #self.logger.debug(schedules)
        return schedules

    def get_attributes(self):
        '''
            returns a list of tuples of (attribute,value) 
            for all learned attributes
            
            Returns
            -------
            list of tuples of (attribute,value) 
        '''
        reg_attr = self.regressors[0].get_attributes()
        attr = [{self.regressor_class.__name__:reg_attr}]

        return attr