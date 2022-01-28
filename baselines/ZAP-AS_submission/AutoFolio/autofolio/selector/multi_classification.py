import logging
import traceback

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter
from ConfigSpace.conditions import EqualsCondition, InCondition
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace import Configuration

from ASlibScenario.aslib_scenario.aslib_scenario import ASlibScenario

__author__ = "Marius Lindauer"
__license__ = "BSD"


class MultiClassifier(object):

    @staticmethod
    def add_params(cs: ConfigurationSpace):
        '''
            adds parameters to ConfigurationSpace 
        '''
        
        selector = cs.get_hyperparameter("selector")
        classifier = cs.get_hyperparameter("classifier")
        if "MultiClassifier" in selector.choices:
            cond = InCondition(child=classifier, parent=selector, values=["MultiClassifier"])
            cs.add_condition(cond)

    def __init__(self, classifier_class):
        '''
            Constructor
        '''
        self.classifiers = []
        self.logger = logging.getLogger("MultiClassifier")
        self.classifier_class = classifier_class
        self.normalizer = MinMaxScaler()

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
        self.logger.info("Fit PairwiseClassifier with %s" %
                         (self.classifier_class))

        self.algorithms = scenario.algorithms

        from sklearn.utils import check_array
        from sklearn.tree._tree import DTYPE

        n_algos = len(scenario.algorithms)
        X = scenario.feature_data.values
        # since sklearn (at least the RFs) 
        # uses float32 and we pass float64,
        # the normalization ensures that floats
        # are not converted to inf or -inf
        #X = (X - np.min(X)) / (np.max(X) - np.min(X))
        X = self.normalizer.fit_transform(X)
        y = np.argmin(scenario.performance_data.values,axis=1)
        weights = scenario.performance_data.std(axis=1)
        clf = self.classifier_class()
        clf.fit(X, y, config, weights)
        self.classifier = clf
         
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
        X = self.normalizer.transform(X)
        algo_indx = self.classifier.predict(X)
        
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
        class_attr = self.classifiers[0].get_attributes()
        attr = [{self.classifier_class.__name__:class_attr}]

        return attr