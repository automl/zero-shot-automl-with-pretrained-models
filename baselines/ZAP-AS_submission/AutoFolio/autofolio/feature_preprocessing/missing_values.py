import logging

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer

from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter
from ConfigSpace import Configuration
from ConfigSpace.configuration_space import ConfigurationSpace

from ASlibScenario.aslib_scenario.aslib_scenario import ASlibScenario

__author__ = "Marius Lindauer"
__license__ = "BSD"


class ImputerWrapper(object):

    @staticmethod
    def add_params(cs: ConfigurationSpace):
        '''
            adds parameters to ConfigurationSpace 
        '''
        stratgey = CategoricalHyperparameter(
            "imputer_strategy", choices=["mean", "median", "most_frequent"], default_value="mean")
        cs.add_hyperparameter(stratgey)

    def __init__(self):
        '''
            Constructor
        '''
        self.imputer = None
        self.active = False

        self.logger = logging.getLogger("MissingValueImputation")

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

        self.imputer = SimpleImputer(strategy=config.get("imputer_strategy"))
        self.imputer.fit(scenario.feature_data.values)
        self.active = True

    def transform(self, scenario: ASlibScenario):
        '''
            transform ASLib scenario data

            Arguments
            ---------
            scenario: data.aslib_scenario.ASlibScenario
                ASlib Scenario with all data in pandas

            Returns
            -------
            data.aslib_scenario.ASlibScenario
        '''
        self.logger.debug("Impute Missing Feature Values")

        values = self.imputer.transform(
            np.array(scenario.feature_data.values))
        scenario.feature_data = pd.DataFrame(
            data=values, index=scenario.feature_data.index, columns=scenario.feature_data.columns)

        return scenario

    def fit_transform(self, scenario: ASlibScenario, config: Configuration):
        '''
            fit and transform

            Arguments
            ---------
            scenario: data.aslib_scenario.ASlibScenario
                ASlib Scenario with all data in pandas
            config: ConfigSpace.Configuration
                configuration

            Returns
            -------
            data.aslib_scenario.ASlibScenario
        '''
        self.fit(scenario, config)
        scenario = self.transform(scenario)
        return scenario
    
    def get_attributes(self):
        '''
            returns a list of tuples of (attribute,value) 
            for all learned attributes
            
            Arguments
            ---------
            config: ConfigSpace.Configuration
                configuration

            
            Returns
            -------
            list of tuples of (attribute,value) 
        '''
        return ["Strategy=%s" %(self.imputer.strategy)]