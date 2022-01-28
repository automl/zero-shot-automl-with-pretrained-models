import logging

import numpy as np
import pandas as pd

from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter

__author__ = "Marius Lindauer"
__license__ = "BSD"

class FeatureGroupFiltering(object):
    '''
        based on the selected feature group, we remove all features that are not available;
        we also add the feature costs for each individual instance
    '''

    @staticmethod
    def add_params(cs):
        '''
            adds parameters to ConfigurationSpace 
        '''

    def __init__(self):
        '''
            Constructor
        '''
        self.logger = logging.getLogger("FeatureGroupFiltering")
        self.active_features = []
        self.active_groups = []
        self.active = False

    def fit(self, scenario, config):
        '''
            fit pca object to ASlib scenario data

            Arguments
            ---------
            scenario: data.aslib_scenario.ASlibScenario
                ASlib Scenario with all data in pandas
            config: ConfigSpace.Configuration
                configuration
        '''
        self.active = True
        active_groups = []
        for param in config:
            if param.startswith("fgroup_") and config[param]:
                active_groups.append(param.replace("fgroup_", ""))
        
        active_groups.sort() # to ensure same order of features always
        
        # check requirements for each step
        change = True
        while change:
            change = False
            for group in active_groups:
                if scenario.feature_group_dict[group].get("requires"):
                    valid = True
                    for req_group in scenario.feature_group_dict[group].get("requires"):
                        if req_group not in active_groups:
                            valid = False
                            break
                    if not valid:
                        active_groups.remove(group)
                        change = True

        self.logger.debug("Active feature groups: %s" %(active_groups))
        self.active_groups = active_groups
        
        # get active features
        for group in active_groups:
            if scenario.feature_group_dict[group].get("provides"):
                self.active_features.extend(scenario.feature_group_dict[group].get("provides"))
        
        self.logger.debug("Active features (%d): %s" %(len(self.active_features), self.active_features))
            
        if not self.active_features:
            self.logger.warn("No active features left after filtering according to selected feature steps")


    def transform(self, scenario):
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
        
        
        scenario.feature_data = scenario.feature_data[self.active_features]
        scenario.used_feature_groups = self.active_groups
        
        return scenario

    def fit_transform(self, scenario, config):
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
        return [{"Feature Groups":self.active_groups}]