import logging

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter
from ConfigSpace.conditions import EqualsCondition, InCondition
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace import Configuration

from ASlibScenario.aslib_scenario.aslib_scenario import ASlibScenario

__author__ = "Marius Lindauer"
__license__ = "BSD"


class StandardScalerWrapper(object):

    @staticmethod
    def add_params(cs: ConfigurationSpace):
        '''
            adds parameters to ConfigurationSpace 
        '''
        switch = CategoricalHyperparameter(
            "StandardScaler", choices=[True, False], default_value=True)
        cs.add_hyperparameter(switch)

    def __init__(self):
        '''
            Constructor
        '''
        self.scaler = None
        self.active = False
        self.logger = logging.getLogger("StandardScaler")

    def fit(self, scenario: ASlibScenario, config: Configuration):
        '''
            fit StandardScaler object to ASlib scenario data

            Arguments
            ---------
            scenario: data.aslib_scenario.ASlibScenario
                ASlib Scenario with all data in pandas
            config: ConfigSpace.Configuration
                configuration
        '''

        if config.get("StandardScaler"):
            self.active = True
            self.scaler = StandardScaler()
            self.scaler.fit(scenario.feature_data.values)

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
        if self.scaler:
            self.logger.debug("Applying StandardScaler")
            
            values = self.scaler.transform(
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
