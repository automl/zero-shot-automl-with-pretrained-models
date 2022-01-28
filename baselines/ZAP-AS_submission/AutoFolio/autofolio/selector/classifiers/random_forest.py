import numpy as np
import pandas as pd

from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter
from ConfigSpace.conditions import EqualsCondition, InCondition
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace import Configuration

from ASlibScenario.aslib_scenario.aslib_scenario import ASlibScenario

from sklearn.ensemble import RandomForestClassifier

__author__ = "Marius Lindauer"
__license__ = "BSD"


class RandomForest(object):

    @staticmethod
    def add_params(cs: ConfigurationSpace):
        '''
            adds parameters to ConfigurationSpace 
        '''
        try:
            classifier = cs.get_hyperparameter("classifier")
            if "RandomForest" not in classifier.choices:
                return

            n_estimators = UniformIntegerHyperparameter(
                name="rf:n_estimators", lower=10, upper=100, default_value=10, log=True)
            cs.add_hyperparameter(n_estimators)
            criterion = CategoricalHyperparameter(
                name="rf:criterion", choices=["gini", "entropy"], default_value="gini")
            cs.add_hyperparameter(criterion)
            max_features = CategoricalHyperparameter(
                name="rf:max_features", choices=["sqrt", "log2", "None"], default_value="sqrt")
            cs.add_hyperparameter(max_features)
            max_depth = UniformIntegerHyperparameter(
                name="rf:max_depth", lower=10, upper=2**31, default_value=2**31, log=True)
            cs.add_hyperparameter(max_depth)
            min_samples_split = UniformIntegerHyperparameter(
                name="rf:min_samples_split", lower=2, upper=100, default_value=2, log=True)
            cs.add_hyperparameter(min_samples_split)
            min_samples_leaf = UniformIntegerHyperparameter(
                name="rf:min_samples_leaf", lower=2, upper=100, default_value=10, log=True)
            cs.add_hyperparameter(min_samples_leaf)
            bootstrap = CategoricalHyperparameter(
                name="rf:bootstrap", choices=[True, False], default_value=True)
            cs.add_hyperparameter(bootstrap)

            cond = InCondition(
                child=n_estimators, parent=classifier, values=["RandomForest"])
            cs.add_condition(cond)
            cond = InCondition(
                child=criterion, parent=classifier, values=["RandomForest"])
            cs.add_condition(cond)
            cond = InCondition(
                child=max_features, parent=classifier, values=["RandomForest"])
            cs.add_condition(cond)
            cond = InCondition(
                child=max_depth, parent=classifier, values=["RandomForest"])
            cs.add_condition(cond)
            cond = InCondition(
                child=min_samples_split, parent=classifier, values=["RandomForest"])
            cs.add_condition(cond)
            cond = InCondition(
                child=min_samples_leaf, parent=classifier, values=["RandomForest"])
            cs.add_condition(cond)
            cond = InCondition(
                child=bootstrap, parent=classifier, values=["RandomForest"])
            cs.add_condition(cond)
            print(cs)
        except:
            return

    def __init__(self):
        '''
            Constructor
        '''

        self.model = None

    def __str__(self):
        return "RandomForest"

    def fit(self, X, y, config: Configuration, weights=None):
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

        self.model = RandomForestClassifier(n_estimators=config["rf:n_estimators"],
                                            max_features= config[
                                                "rf:max_features"] if config[
                                                "rf:max_features"] != "None" else None,
                                            criterion=config["rf:criterion"],
                                            max_depth=config["rf:max_depth"],
                                            min_samples_split=config[
                                                "rf:min_samples_split"],
                                            min_samples_leaf=config[
                                                "rf:min_samples_leaf"],
                                            bootstrap=config["rf:bootstrap"],
                                            random_state=12345)
        self.model.fit(X, y, weights)

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
        attr.append("max_depth = %d" %(self.model.max_depth))
        attr.append("min_samples_split = %d" %(self.model.min_samples_split))
        attr.append("min_samples_leaf = %d" %(self.model.min_samples_leaf))
        attr.append("criterion = %s" %(self.model.criterion))
        attr.append("n_estimators = %d" %(self.model.n_estimators))
        attr.append("max_features = %s" %(self.model.max_features))
        return attr
        