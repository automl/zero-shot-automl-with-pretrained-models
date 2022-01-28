import numpy as np
import pandas as pd

from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter
from ConfigSpace.conditions import EqualsCondition, InCondition
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace import Configuration

from aslib_scenario.aslib_scenario import ASlibScenario

import sklearn.ensemble

__author__ = "Marius Lindauer"
__license__ = "BSD"


class RandomForestRegressor(object):

    @staticmethod
    def add_params(cs: ConfigurationSpace):
        '''
            adds parameters to ConfigurationSpace 
        '''

        try:
            regressor = cs.get_hyperparameter("regressor")
            if "RandomForestRegressor" not in regressor.choices:
                return

            n_estimators = UniformIntegerHyperparameter(
                name="rfreg:n_estimators", lower=10, upper=100, default_value=10, log=True)
            cs.add_hyperparameter(n_estimators)
            max_features = CategoricalHyperparameter(
                name="rfreg:max_features", choices=["sqrt", "log2", "None"], default_value="sqrt")
            cs.add_hyperparameter(max_features)
            max_depth = UniformIntegerHyperparameter(
                name="rfreg:max_depth", lower=10, upper=2 ** 31, default_value=2 ** 31, log=True)
            cs.add_hyperparameter(max_depth)
            min_samples_split = UniformIntegerHyperparameter(
                name="rfreg:min_samples_split", lower=2, upper=100, default_value=2, log=True)
            cs.add_hyperparameter(min_samples_split)
            min_samples_leaf = UniformIntegerHyperparameter(
                name="rfreg:min_samples_leaf", lower=2, upper=100, default_value=10, log=True)
            cs.add_hyperparameter(min_samples_leaf)
            bootstrap = CategoricalHyperparameter(
                name="rfreg:bootstrap", choices=[True, False], default_value=True)
            cs.add_hyperparameter(bootstrap)

            cond = InCondition(
                child=n_estimators, parent=regressor, values=["RandomForestRegressor"])
            cs.add_condition(cond)
            cond = InCondition(
                child=max_features, parent=regressor, values=["RandomForestRegressor"])
            cs.add_condition(cond)
            cond = InCondition(
                child=max_depth, parent=regressor, values=["RandomForestRegressor"])
            cs.add_condition(cond)
            cond = InCondition(
                child=min_samples_split, parent=regressor, values=["RandomForestRegressor"])
            cs.add_condition(cond)
            cond = InCondition(
                child=min_samples_leaf, parent=regressor, values=["RandomForestRegressor"])
            cs.add_condition(cond)
            cond = InCondition(
                child=bootstrap, parent=regressor, values=["RandomForestRegressor"])
            cs.add_condition(cond)

        except:
            return

    def __init__(self):
        '''
            Constructor
        '''

        self.model = None

    def __str__(self):
        return "RandomForestRegressor"

    def fit(self, X, y, config: Configuration):
        '''
            fit pca object to ASlib scenario data

            Arguments
            ---------
            X: numpy.array
                feature matrix
            y: numpy.array
                label vector
            weights: numpy.array
                vector with sample weights
            config: ConfigSpace.Configuration
                configuration

        '''

        self.model = sklearn.ensemble.RandomForestRegressor(n_estimators=config["rfreg:n_estimators"],
                                            max_features=config[
                                                "rfreg:max_features"] if config[
                                                "rfreg:max_features"] != "None" else None,
                                            max_depth=config["rf:max_depth"],
                                            min_samples_split=config[
                                                "rfreg:min_samples_split"],
                                            min_samples_leaf=config[
                                                "rfreg:min_samples_leaf"],
                                            bootstrap=config["rfreg:bootstrap"],
                                            random_state=12345)
        self.model.fit(X, y)

    def predict(self, X):
        '''
            transform ASLib scenario data

            Arguments
            ---------
            X: numpy.array
                instance feature matrix

            Returns
            -------

        '''

        return self.model.predict(X)
    
    def get_attributes(self):
        '''
            returns a list of tuples of (attribute,value) 
            for all learned attributes
            
            Returns
            -------
            list of tuples of (attribute,value) 
        '''
        attr = []
        attr.append("max_depth = %d" % (self.model.max_depth))
        attr.append("min_samples_split = %d" % (self.model.min_samples_split))
        attr.append("min_samples_leaf = %d" % (self.model.min_samples_leaf))
        attr.append("n_estimators = %d" % (self.model.n_estimators))
        attr.append("max_features = %s" % (self.model.max_features))
        return attr
        
