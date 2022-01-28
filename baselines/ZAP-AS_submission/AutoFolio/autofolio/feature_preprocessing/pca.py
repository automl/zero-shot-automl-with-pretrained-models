import logging

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA

from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter
from ConfigSpace.conditions import EqualsCondition, InCondition
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace import Configuration

from ASlibScenario.aslib_scenario.aslib_scenario import ASlibScenario

__author__ = "Marius Lindauer"
__license__ = "BSD"


class PCAWrapper(object):

    @staticmethod
    def add_params(cs: ConfigurationSpace):
        '''
            adds parameters to ConfigurationSpace 
        '''
        pca_switch = CategoricalHyperparameter(
            "pca", choices=[True, False], default_value=False)
        n_components = UniformIntegerHyperparameter(
            "pca_n_components", lower=1, upper=20, default_value=7, log=True)
        cs.add_hyperparameter(pca_switch)
        cs.add_hyperparameter(n_components)
        cond = InCondition(
            child=n_components, parent=pca_switch, values=[True])
        cs.add_condition(cond)

    def __init__(self):
        '''
            Constructor
        '''
        self.pca = None
        self.active = False

        self.logger = logging.getLogger("PCA")

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

        if config.get("pca"):
            self.pca = PCA(n_components=config.get("pca_n_components"))
            self.pca.fit(scenario.feature_data.values)
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
        if self.pca:
            self.logger.debug("Applying PCA")
            values = self.pca.transform(
                np.array(scenario.feature_data.values))

            scenario.feature_data = pd.DataFrame(
                data=values, index=scenario.feature_data.index, columns=["f%d" % (i) for i in range(values.shape[1])])

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
            
            Returns
            -------
            list of tuples of (attribute,value) 
        '''
        return ["Dimensions=%s" %(self.pca.n_components)]
