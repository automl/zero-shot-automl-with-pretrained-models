import numpy as np
import pandas as pd

from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter
from ConfigSpace.conditions import EqualsCondition, InCondition
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace import Configuration

from ASlibScenario.aslib_scenario.aslib_scenario import ASlibScenario

import xgboost as xgb

__author__ = "Marius Lindauer"
__license__ = "BSD"


class XGBoost(object):

    @staticmethod
    def add_params(cs: ConfigurationSpace):
        '''
            adds parameters to ConfigurationSpace 
        '''
        
        try:
            classifier = cs.get_hyperparameter("classifier")
            if "XGBoost" not in classifier.choices:
                return 

            num_round = UniformIntegerHyperparameter(
                name="xgb:num_round", lower=10, upper=100, default_value=50, log=True)
            cs.add_hyperparameter(num_round)
            alpha = UniformFloatHyperparameter(
                name="xgb:alpha", lower=0, upper=10, default_value=1)
            cs.add_hyperparameter(alpha)
            lambda_ = UniformFloatHyperparameter(
                name="xgb:lambda", lower=1, upper=10, default_value=1)
            cs.add_hyperparameter(lambda_)
            colsample_bylevel = UniformFloatHyperparameter(
                name="xgb:colsample_bylevel", lower=0.5, upper=1, default_value=1)
            cs.add_hyperparameter(colsample_bylevel)
            colsample_bytree = UniformFloatHyperparameter(
                name="xgb:colsample_bytree", lower=0.5, upper=1, default_value=1)
            cs.add_hyperparameter(colsample_bytree)
            subsample = UniformFloatHyperparameter(
                name="xgb:subsample", lower=0.01, upper=1, default_value=1)
            cs.add_hyperparameter(subsample)
            max_delta_step = UniformFloatHyperparameter(
                name="xgb:max_delta_step", lower=0, upper=10, default_value=0)
            cs.add_hyperparameter(max_delta_step)
            min_child_weight = UniformFloatHyperparameter(
                name="xgb:min_child_weight", lower=0, upper=20, default_value=1)
            cs.add_hyperparameter(min_child_weight)
            max_depth = UniformIntegerHyperparameter(
                name="xgb:max_depth", lower=1, upper=10, default_value=6)
            cs.add_hyperparameter(max_depth)
            gamma = UniformFloatHyperparameter(
                name="xgb:gamma", lower=0, upper=10, default_value=0)
            cs.add_hyperparameter(gamma)
            eta = UniformFloatHyperparameter(
                name="xgb:eta", lower=0, upper=1, default_value=0.3)
            cs.add_hyperparameter(eta)

            cond = InCondition(
                child=num_round, parent=classifier, values=["XGBoost"])
            cs.add_condition(cond)
            cond = InCondition(
                child=alpha, parent=classifier, values=["XGBoost"])
            cs.add_condition(cond)
            cond = InCondition(
                child=lambda_, parent=classifier, values=["XGBoost"])
            cs.add_condition(cond)
            cond = InCondition(
                child=colsample_bylevel, parent=classifier, values=["XGBoost"])
            cs.add_condition(cond)
            cond = InCondition(
                child=colsample_bytree, parent=classifier, values=["XGBoost"])
            cs.add_condition(cond)
            cond = InCondition(
                child=subsample, parent=classifier, values=["XGBoost"])
            cs.add_condition(cond)
            cond = InCondition(
                child=max_delta_step, parent=classifier, values=["XGBoost"])
            cs.add_condition(cond)
            cond = InCondition(
                child=min_child_weight, parent=classifier, values=["XGBoost"])
            cs.add_condition(cond)
            cond = InCondition(
                child=max_depth, parent=classifier, values=["XGBoost"])
            cs.add_condition(cond)
            cond = InCondition(
                child=gamma, parent=classifier, values=["XGBoost"])
            cs.add_condition(cond)
            cond = InCondition(
                child=eta, parent=classifier, values=["XGBoost"])
            cs.add_condition(cond)
        except: 
            return
        

    def __init__(self):
        '''
            Constructor
        '''

        self.model = None
        self.attr = []

    def __str__(self):
        return "XGBoost"

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
        
        xgb_config = {'nthread': 1,
         'silent': 1, 
         'objective': 'binary:logistic',
         'seed': 12345}
        for param in config:
            if param.startswith("xgb:") and config[param] is not None:
                self.attr.append("%s=%s"%(param[4:],config[param]))
            if param == "xgb:num_round":
                continue
            xgb_config[param[4:]] = config[param]
            
        dtrain = xgb.DMatrix(X, label=y, weight=weights)
        self.model = xgb.train(xgb_config, dtrain, config["xgb:num_round"])
        

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
        preds = np.array(self.model.predict(xgb.DMatrix(X)))
        preds[preds < 0.5] = 0
        preds[preds >= 0.5] = 1
        return preds

    def get_attributes(self):
        '''
            returns a list of tuples of (attribute,value) 
            for all learned attributes

            Returns
            -------
            list of tuples of (attribute,value) 
        '''
        return self.attr
