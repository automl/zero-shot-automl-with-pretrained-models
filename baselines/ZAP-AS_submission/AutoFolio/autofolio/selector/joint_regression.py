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


class JointRegression(object):

    @staticmethod
    def add_params(cs: ConfigurationSpace):
        '''
            adds parameters to ConfigurationSpace 
        '''

        selector = cs.get_hyperparameter("selector")
        regressor = cs.get_hyperparameter("regressor")
        if "JointRegressor" in selector.choices:
            cond = InCondition(child=regressor, parent=selector, values=["JointRegressor"])
            cs.add_condition(cond)

    def __init__(self, regressor_class):
        '''
            Constructor
        '''
        self.regressors = []
        self.logger = logging.getLogger("JointRegressor")
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
        Xs = None
        Ys = None
        
        for i in range(n_algos):
            y = scenario.performance_data[scenario.algorithms[i]].values
            x_i = np.zeros((X.shape[0], n_algos))
            x_i[:,i] = 1
            X_i = np.hstack([X,x_i])
            if i == 0:
                Xs = X_i
                Ys = y
            else:
                Xs = np.vstack([Xs,X_i])
                Ys = np.hstack([Ys,y])
            
        reg = self.regressor_class()
        reg.fit(Xs, Ys, config)
        self.regressors = reg

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
        for i in range(n_algos):
            x_i = np.zeros((X.shape[0], n_algos))
            x_i[:,i] = 1
            X_i = np.hstack([X,x_i])
            Y = self.regressors.predict(X_i)
            scores[:, i] += Y

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