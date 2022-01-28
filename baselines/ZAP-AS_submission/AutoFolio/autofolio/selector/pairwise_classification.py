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


class PairwiseClassifier(object):

    @staticmethod
    def add_params(cs: ConfigurationSpace):
        '''
            adds parameters to ConfigurationSpace 
        '''
        
        selector = cs.get_hyperparameter("selector")
        classifier = cs.get_hyperparameter("classifier")
        if "PairwiseClassifier" in selector.choices:
            cond = InCondition(child=classifier, parent=selector, values=["PairwiseClassifier"])
            cs.add_condition(cond)

    def __init__(self, classifier_class):
        '''
            Constructor
        '''
        self.classifiers = []
        self.logger = logging.getLogger("PairwiseClassifier")
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
        for i in range(n_algos):
            for j in range(i + 1, n_algos):
                y_i = scenario.performance_data[scenario.algorithms[i]].values
                y_j = scenario.performance_data[scenario.algorithms[j]].values
                y = y_i < y_j
                weights = np.abs(y_i - y_j)
                clf = self.classifier_class()
                clf.fit(X, y, config, weights)
                self.classifiers.append(clf)

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
        scores = np.zeros((X.shape[0], n_algos))
        clf_indx = 0
        for i in range(n_algos):
            for j in range(i + 1, n_algos):
                clf = self.classifiers[clf_indx]
                Y = clf.predict(X)
                scores[Y == 1, i] += 1
                scores[Y == 0, j] += 1
                clf_indx += 1

        #self.logger.debug(
        #   sorted(list(zip(scenario.algorithms, scores)), key=lambda x: x[1], reverse=True))
        algo_indx = np.argmax(scores, axis=1)
        
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