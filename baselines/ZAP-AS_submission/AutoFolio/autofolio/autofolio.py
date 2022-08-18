import logging
import functools
import traceback
import random
from itertools import tee
import pickle
import os
import sys

import numpy as np
import pandas as pd
import yaml

# SMAC3
try:
    from smac.tae.execute_func import ExecuteTAFuncDict
    from smac.scenario.scenario import Scenario
    from smac.stats.stats import Stats as AC_Stats
    from smac.facade.smac_hpo_facade import SMAC4HPO as SMAC
except ImportError:
    pass

sys.path.insert(0, os.path.abspath("ZAP-AS_submission/AutoFolio"))
sys.path.insert(0, os.path.abspath("ZAP-AS_submission/ASlibScenario"))

from ASlibScenario.aslib_scenario import aslib_scenario  # needed for opening the model file
from ASlibScenario.aslib_scenario.aslib_scenario import ASlibScenario
from AutoFolio.autofolio.feature_preprocessing.feature_group_filtering import FeatureGroupFiltering
from AutoFolio.autofolio.feature_preprocessing.missing_values import ImputerWrapper

# feature preprocessing
from AutoFolio.autofolio.feature_preprocessing.pca import PCAWrapper
from AutoFolio.autofolio.feature_preprocessing.standardscaler import StandardScalerWrapper

from AutoFolio.autofolio.io.cmd import CMDParser
# presolving
from AutoFolio.autofolio.pre_solving.aspeed_schedule import Aspeed
# classifiers
from AutoFolio.autofolio.selector.classifiers.random_forest import RandomForest
from AutoFolio.autofolio.selector.classifiers.xgboost import XGBoost
from AutoFolio.autofolio.selector.ind_regression import IndRegression
from AutoFolio.autofolio.selector.joint_regression import JointRegression
from AutoFolio.autofolio.selector.multi_classification import MultiClassifier
# selectors
from AutoFolio.autofolio.selector.pairwise_classification import PairwiseClassifier
from AutoFolio.autofolio.selector.pairwise_regression import PairwiseRegression
# regressors
from AutoFolio.autofolio.selector.regressors.random_forest import RandomForestRegressor
# validation
from AutoFolio.autofolio.validation.validate import Stats, Validator

from ConfigSpace.configuration_space import Configuration, ConfigurationSpace
from ConfigSpace.hyperparameters import (CategoricalHyperparameter, UniformFloatHyperparameter, UniformIntegerHyperparameter)


__author__ = "Marius Lindauer"
__license__ = "BSD"
__version__ = "2.2.0"


class AutoFolio(object):

    def __init__(self, random_seed: int=12345):
        ''' Constructor 

            Arguments
            ---------
            random_seed: int
                random seed for numpy and random packages
        '''

        np.random.seed(random_seed)  # fix seed
        random.seed(random_seed)

        # I don't know the reason, but without an initial print with
        # logging.info we don't get any output
        logging.info("Init AutoFolio")
        self._root_logger = logging.getLogger()
        self.logger = logging.getLogger("AutoFolio")
        self.cs = None

        self.overwrite_args = None

    def run_cli(self):
        '''
            main method of AutoFolio based on command line interface
        '''

        cmd_parser = CMDParser()
        args_, self.overwrite_args = cmd_parser.parse()

        self._root_logger.setLevel(args_.verbose)

        if args_.load:
            pred = self.read_model_and_predict(
                model_fn=args_.load, feature_vec=list(map(float, args_.feature_vec.split(" "))))
            print("Selected Schedule [(algorithm, budget)]: %s" % (pred))

        else:

            scenario = ASlibScenario()
            if args_.scenario:
                scenario.read_scenario(args_.scenario)
            elif args_.performance_csv and args_.feature_csv:
                scenario.read_from_csv(perf_fn=args_.performance_csv,
                                       feat_fn=args_.feature_csv,
                                       objective=args_.objective,
                                       runtime_cutoff=args_.runtime_cutoff,
                                       maximize=args_.maximize,
                                       cv_fn=args_.cv_csv)
            else:
                raise ValueError("Missing inputs to read scenario data.")

            test_scenario = None
            if args_.performance_test_csv and args_.feature_test_csv:
                test_scenario = ASlibScenario()
                test_scenario.read_from_csv(perf_fn=args_.performance_test_csv,
                                       feat_fn=args_.feature_test_csv,
                                       objective=args_.objective,
                                       runtime_cutoff=args_.runtime_cutoff,
                                       maximize=args_.maximize,
                                       cv_fn=None)

            config = {}
            if args_.config is not None:
                self.logger.info("Reading yaml config file")
                config = yaml.load(open(args_.config))
            if not config.get("wallclock_limit"):
                config["wallclock_limit"] = args_.wallclock_limit
            if not config.get("runcount_limit"):
                config["runcount_limit"] = args_.runcount_limit
            if not config.get("output-dir"):
                config["output-dir"] = args_.output_dir

            self.cs = self.get_cs(scenario, config)

            if args_.outer_cv:
                self._outer_cv(scenario, config, args_.outer_cv_fold, 
                    args_.out_template, smac_seed=args_.smac_seed)
                return 0
            
            if args_.tune:
                config = self.get_tuned_config(scenario,
                                               wallclock_limit=args_.wallclock_limit,
                                               runcount_limit=args_.runcount_limit,
                                               autofolio_config=config,
                                               seed=args_.smac_seed)
            else:
                config = self.cs.get_default_configuration()
            self.logger.debug(config)

            if args_.save:
                feature_pre_pipeline, pre_solver, selector = self.fit(
                    scenario=scenario, config=config)
                self._save_model(
                    args_.save, scenario, feature_pre_pipeline, pre_solver, selector, config)
            else:
                self.run_cv(config=config, scenario=scenario, folds=int(scenario.cv_data.max().max()))

            if test_scenario is not None:
                stats = self.run_fold(config=config,
                                      fold=0,
                                      return_fit=False,
                                      scenario=scenario,
                                      test_scenario=test_scenario)

    def _outer_cv(self, scenario: ASlibScenario, autofolio_config:dict=None, 
            outer_cv_fold:int=None, out_template:str=None,
            smac_seed:int=42):
        '''
            Evaluate on a scenario using an "outer" cross-fold validation
            scheme. In particular, this ensures that SMAC does not use the test
            set during hyperparameter optimization.

            Arguments
            ---------
            scenario: ASlibScenario
                ASlib Scenario at hand
            
            autofolio_config: dict, or None
                An optional dictionary of configuration options

            outer_cv_fold: int, or None
                If given, then only the single outer-cv fold is processed

            out_template: str, or None
                If given, the learned configurations are written to the 
                specified locations. The string is considered a template, and
                "%fold%" will be replaced with the fold.

            smac_seed:int 
                random seed for SMAC

            Returns
            -------
            stats: validate.Stats
                Performance over all outer-cv folds

        '''
        import string

        outer_stats = None

        # For each outer split
        outer_cv_folds = range(1, 11)
        if outer_cv_fold is not None:
            outer_cv_folds = range(outer_cv_fold, outer_cv_fold+1)

        for cv_fold in outer_cv_folds:
            
            # Use ‘ASlibScenario.get_split()’ to get the outer split
            outer_testing, outer_training = scenario.get_split(cv_fold)
            
            msg = ">>>>> Outer CV fold: {} <<<<<".format(cv_fold)
            self.logger.info(msg)

            # Use ASlibScenario.create_cv_splits() to get an inner-cv
            outer_training.create_cv_splits(n_folds=10)
            
            # Use ‘AutoFolio.get_tuned_config()’ to tune on inner-cv
            config = self.get_tuned_config(
                outer_training, 
                autofolio_config=autofolio_config,
                seed=smac_seed
            )
            
            # Use `AutoFolio.run_fold()’ to get the performance on the outer split
            stats, fit, schedule = self.run_fold(
                config, 
                scenario, 
                cv_fold, 
                return_fit=True
            )

            feature_pre_pipeline, pre_solver, selector = fit

            if outer_stats is None:
                outer_stats = stats
            else:
                outer_stats.merge(stats)

            # save the model, if given an output location
            if out_template is not None:
                out_template_ = string.Template(out_template)
                model_fn = out_template_.substitute(fold=cv_fold, type="pkl")
                
                msg = "Writing model to: {}".format(model_fn)
                self.logger.info(msg)

                self._save_model(
                    model_fn, 
                    scenario, 
                    feature_pre_pipeline, 
                    pre_solver, 
                    selector, 
                    config
                )

                # convert the schedule to a data frame
                schedule_df = pd.Series(schedule, name="solver")
                schedule_df.index.name = "instance"
                schedule_df = schedule_df.reset_index()

                # just keep the solver name; we don't care about the time

                # x[0] gets the first pair in the schedule list
                # and x[0][0] gets the name of the solver from that pair
                schedule_df['solver'] = schedule_df['solver'].apply(lambda x: x[0][0])

                selections_fn = out_template_.substitute(fold=cv_fold, type="csv")

                msg = "Writing solver choices to: {}".format(selections_fn)
                self.logger.info(msg)

                schedule_df.to_csv(selections_fn, index=False)

        self.logger.info(">>>>> Final Stats <<<<<")
        outer_stats.show()

    def _save_model(self, out_fn: str, scenario: ASlibScenario, feature_pre_pipeline: list, pre_solver: Aspeed, selector, config: Configuration):
        '''
            save all pipeline objects for predictions

            Arguments
            ---------
            out_fn: str
                filename of output file
            scenario: AslibScenario
                ASlib scenario with all the data
            feature_pre_pipeline: list
                list of preprocessing objects
            pre_solver: Aspeed
                aspeed object with pre-solving schedule
            selector: autofolio.selector.*
                fitted selector object
            config: Configuration
                parameter setting configuration
        '''
        scenario.logger = None
        for fpp in feature_pre_pipeline:
            fpp.logger = None
        if pre_solver:
            pre_solver.logger = None
        selector.logger = None
        model = [scenario, feature_pre_pipeline, pre_solver, selector, config]
        with open(out_fn, "bw") as fp:
            pickle.dump(model, fp)

    def read_model_and_predict(self, model_fn: str, feature_vec: list):
        '''
            reads saved model from disk and predicts the selected algorithm schedule for a given feature vector

            Arguments
            --------
            model_fn: str
                file name of saved model
            feature_vec: list
                instance feature vector as a list of floats 

            Returns
            -------
            list of tuple
                Selected schedule [(algorithm, budget)]
        '''
        with open(model_fn, "br") as fp:
            scenario, feature_pre_pipeline, pre_solver, selector, config = pickle.load(fp)

        for fpp in feature_pre_pipeline:
            fpp.logger = logging.getLogger("Feature Preprocessing")
        if pre_solver:
            pre_solver.logger = logging.getLogger("Aspeed PreSolving")
        selector.logger = logging.getLogger("Selector")

        # saved scenario is adapted to given feature vector
        feature_vec = np.array([feature_vec])
        scenario.feature_data = pd.DataFrame(
            feature_vec, index=["pseudo_instance"], columns=scenario.features)
        scenario.instances = ["pseudo_instance"]
        pred = self.predict(scenario=scenario, config=config,
                            feature_pre_pipeline=feature_pre_pipeline, pre_solver=pre_solver, selector=selector)

        return pred["pseudo_instance"]

    def get_cs(self, scenario: ASlibScenario, autofolio_config:dict=None):
        '''
            returns the parameter configuration space of AutoFolio
            (based on the automl config space: https://github.com/automl/ConfigSpace)

            Arguments
            ---------
            scenario: aslib_scenario.aslib_scenario.ASlibScenario
                aslib scenario at hand

            autofolio_config: dict, or None
                An optional dictionary of configuration options
        '''

        self.cs = ConfigurationSpace()

        # only allow the feature groups specified in the config file
        # by default, though, all of the feature groups are allowed.
        allowed_feature_groups = autofolio_config.get("allowed_feature_groups", 
            scenario.feature_steps)

        if len(allowed_feature_groups) == 0:
            msg = "Please ensure at least one feature group is allowed"
            raise ValueError(msg)


        if len(allowed_feature_groups) == 1: 
            choices = [True] # if we only have one feature group, it has to be active 
        else:
            choices = [True, False]
        default = True

        for fs in allowed_feature_groups:
            
            fs_param = CategoricalHyperparameter(name="fgroup_%s" % (fs),
                choices=choices, default_value=default)
            self.cs.add_hyperparameter(fs_param)

        # preprocessing
        if autofolio_config.get("pca", True):
            PCAWrapper.add_params(self.cs)

        if autofolio_config.get("impute", True):
            ImputerWrapper.add_params(self.cs)

        if autofolio_config.get("scale", True):
            StandardScalerWrapper.add_params(self.cs)

        # Pre-Solving
        if scenario.performance_type[0] == "runtime":
            if autofolio_config.get("presolve", True):
                Aspeed.add_params(
                    cs=self.cs, cutoff=scenario.algorithm_cutoff_time)

        if autofolio_config.get("classifier"):
            # fix parameter
            cls_choices = [autofolio_config["classifier"]]
            cls_def = autofolio_config["classifier"]
        else:
            cls_choices = ["RandomForest","XGBoost"]
            cls_def = "RandomForest"
        classifier = CategoricalHyperparameter(
                "classifier", choices=cls_choices, 
                default_value=cls_def)

        self.cs.add_hyperparameter(classifier)

        RandomForest.add_params(self.cs)
        XGBoost.add_params(self.cs)

        if autofolio_config.get("regressor"):
            # fix parameter
            reg_choices = [autofolio_config["regressor"]]
            reg_def = autofolio_config["regressor"]
        else:
            reg_choices = ["RandomForestRegressor"]
            reg_def = "RandomForestRegressor"

        regressor = CategoricalHyperparameter(
                "regressor", choices=reg_choices, default_value=reg_def)
        self.cs.add_hyperparameter(regressor)
        RandomForestRegressor.add_params(self.cs)

        # selectors
        if autofolio_config.get("selector"):
            # fix parameter
            sel_choices = [autofolio_config["selector"]]
            sel_def = autofolio_config["selector"]
        else:
            sel_choices = ["PairwiseClassifier","PairwiseRegressor"]
            sel_def = "PairwiseClassifier"
            
        selector = CategoricalHyperparameter(
                "selector", choices=sel_choices, default_value=sel_def)
        self.cs.add_hyperparameter(selector)
        PairwiseClassifier.add_params(self.cs)
        PairwiseRegression.add_params(self.cs)  

        self.logger.debug(self.cs)

        return self.cs

    def get_tuned_config(self, scenario: ASlibScenario, 
                         runcount_limit:int=42,
                         wallclock_limit:int=300,
                         autofolio_config:dict=dict(),
                         seed:int=42):
        '''
            uses SMAC3 to determine a well-performing configuration in the configuration space self.cs on the given scenario

            Arguments
            ---------
            scenario: ASlibScenario
                ASlib Scenario at hand
            runcount_limit: int
                runcount_limit for SMAC scenario
            wallclock_limit: int
                wallclock limit in sec for SMAC scenario
                (overwritten by autofolio_config)
            autofolio_config: dict, or None
                An optional dictionary of configuration options
            seed: int
                random seed for SMAC

            Returns
            -------
            Configuration
                best incumbent configuration found by SMAC
        '''

        wallclock_limit = autofolio_config.get("wallclock_limit", wallclock_limit)
        runcount_limit = autofolio_config.get("runcount_limit", runcount_limit)

        taf = functools.partial(self.called_by_smac, scenario=scenario)
        max_fold = scenario.cv_data.max().max()
        max_fold = int(max_fold)

        ac_scenario = Scenario({"run_obj": "quality",  # we optimize quality
                                "runcount-limit": runcount_limit,
                                "cs": self.cs,  # configuration space
                                "deterministic": "true",
                                "instances": [[str(i)] for i in range(1, max_fold+1)],
                                "wallclock-limit": wallclock_limit,
                                "output-dir" : "" if not autofolio_config.get("output-dir",None) else autofolio_config.get("output-dir") 
                                })

        # necessary to use stats options related to scenario information
        AC_Stats.scenario = ac_scenario

        # Optimize
        self.logger.info(
            ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        self.logger.info("Start Configuration")
        self.logger.info(
            ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        smac = SMAC(scenario=ac_scenario, tae_runner=taf,
                    rng=np.random.RandomState(seed))
        incumbent = smac.optimize()

        self.logger.info("Final Incumbent: %s" % (incumbent))

        return incumbent

    def called_by_smac(self, config: Configuration, scenario: ASlibScenario, instance:str=None, seed:int=1):
        '''
            run a cross fold validation based on the given data from cv.arff

            Arguments
            ---------
            config: Configuration
                parameter configuration to use for preprocessing
            scenario: aslib_scenario.aslib_scenario.ASlibScenario
                aslib scenario at hand
            instance: str
                cv-fold index 
            seed: int
                random seed (not used)
                
            Returns
            -------
            float: average performance
        '''
        
        if instance is None:
            perf = self.run_cv(config=config, scenario=scenario)
        else:
            try:
                stats = self.run_fold(config=config, scenario=scenario, fold=int(instance))
                perf = stats.show()
            except ValueError:
                if scenario.performance_type[0] == "runtime":
                    perf = scenario.algorithm_cutoff_time * 20
                else:
                    # try to impute a worst case perf
                    perf = scenario.performance_data.max().max()
                
        if scenario.maximize[0]:
            perf *= -1
        
        return perf

    def run_cv(self, config: Configuration, scenario: ASlibScenario, folds:int=10):
        '''
            run a cross fold validation based on the given data from cv.arff

            Arguments
            ---------
            scenario: aslib_scenario.aslib_scenario.ASlibScenario
                aslib scenario at hand
            config: Configuration
                parameter configuration to use for preprocessing
            folds: int
                number of cv-splits
            seed: int
                random seed (not used)
        '''
        #TODO: use seed and instance in an appropriate way
        try:
            if scenario.performance_type[0] == "runtime":
                cv_stat = Stats(runtime_cutoff=scenario.algorithm_cutoff_time)
            else:
                cv_stat = Stats(runtime_cutoff=0)
            for i in range(1, folds + 1):
                self.logger.info("CV-Iteration: %d" % (i))
                stats = self.run_fold(config=config,
                                      scenario=scenario,
                                      fold=i)
                cv_stat.merge(stat=stats)

            self.logger.info(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            self.logger.info("CV Stats")
            par10 = cv_stat.show()
        except ValueError:
            traceback.print_exc()
            par10 = scenario.algorithm_cutoff_time * 10

        if scenario.maximize[0]:
            par10 *= -1

        return par10

    def run_fold(self, config: Configuration, scenario:ASlibScenario, fold:int, test_scenario=None, return_fit:bool=False):
        '''
            run a given fold of cross validation
            
            Arguments
            ---------
            scenario: aslib_scenario.aslib_scenario.ASlibScenario
                aslib scenario at hand
            config: Configuration
                parameter configuration to use for preprocessing
            fold: int
                fold id
            test_scenario:aslib_scenario.aslib_scenario.ASlibScenario
                aslib scenario with test data for validation
                generated from <scenario> if None

            return_fit: bool
                optionally, the learned preprocessing options, presolver and
                selector can be returned
                
            Returns
            -------
            Stats()

            (pre_pipeline, pre_solver, selector):
                only present if return_fit is True
                the pipeline components fit with the configuration options

            schedule: dict of string -> list of (solver, cutoff) pairs
                only present if return_fit is True
                the solver choices for each instance
                
                
        '''

        if test_scenario is None:
            self.logger.info("CV-Iteration: %d" % (fold))
            test_scenario, training_scenario = scenario.get_split(indx=fold)
        else:
            self.logger.info("Validation on test data")
            training_scenario = scenario

        feature_pre_pipeline, pre_solver, selector = self.fit(
            scenario=training_scenario, config=config)

        schedules = self.predict(
            test_scenario, config, feature_pre_pipeline, pre_solver, selector)

        val = Validator()
        if scenario.performance_type[0] == "runtime":
            stats = val.validate_runtime(
                schedules=schedules, test_scenario=test_scenario, train_scenario=training_scenario)
        elif scenario.performance_type[0] == "solution_quality":
            stats = val.validate_quality(
                schedules=schedules, test_scenario=test_scenario, train_scenario=training_scenario)
        else:
            raise ValueError("Unknown: %s" %(scenario.performance_type[0]))
        
        if return_fit:
            return stats, (feature_pre_pipeline, pre_solver, selector), schedules
        else:
            return stats

    def fit(self, scenario: ASlibScenario, config: Configuration):
        '''
            fit AutoFolio on given ASlib Scenario

            Arguments
            ---------
            scenario: aslib_scenario.aslib_scenario.ASlibScenario
                aslib scenario at hand
            config: Configuration
                parameter configuration to use for preprocessing

            Returns
            -------
                list of fitted feature preproccessing objects
                pre-solving object
                fitted selector
        '''
        self.logger.info("Given Configuration: %s" % (config))

        if self.overwrite_args:
            config = self._overwrite_configuration(
                config=config, overwrite_args=self.overwrite_args)
            self.logger.info("Overwritten Configuration: %s" % (config))

        scenario, feature_pre_pipeline = self.fit_transform_feature_preprocessing(
            scenario, config)

        pre_solver = self.fit_pre_solving(scenario, config)

        selector = self.fit_selector(scenario, config)

        return feature_pre_pipeline, pre_solver, selector

    def _overwrite_configuration(self, config: Configuration, overwrite_args: list):
        '''
            overwrites a given configuration with some new settings

            Arguments
            ---------
            config: Configuration
                initial configuration to be adapted
            overwrite_args: list
                new parameter settings as a list of strings

            Returns
            -------
            Configuration
        '''

        def pairwise(iterable):
            a, b = tee(iterable)
            next(b, None)
            return zip(a, b)

        dict_conf = config.get_dictionary()
        for param, value in pairwise(overwrite_args):
            try:
                ok = self.cs.get_hyperparameter(param)
            except KeyError:
                ok = None
            if ok is not None:
                if type(self.cs.get_hyperparameter(param)) is UniformIntegerHyperparameter:
                    dict_conf[param] = int(value)
                elif type(self.cs.get_hyperparameter(param)) is UniformFloatHyperparameter:
                    dict_conf[param] = float(value)
                elif value == "True":
                    dict_conf[param] = True
                elif value == "False":
                    dict_conf[param] = False
                else:
                    dict_conf[param] = value
            else:
                self.logger.warn(
                    "Unknown given parameter: %s %s" % (param, value))
        config = Configuration(self.cs, values=dict_conf, allow_inactive_with_values=True)

        return config

    def fit_transform_feature_preprocessing(self, scenario: ASlibScenario, config: Configuration):
        '''
            performs feature preprocessing on a given ASlib scenario wrt to a given configuration

            Arguments
            ---------
            scenario: aslib_scenario.aslib_scenario.ASlibScenario
                aslib scenario at hand
            config: Configuration
                parameter configuration to use for preprocessing

            Returns
            -------
                list of fitted feature preproccessing objects
        '''

        pipeline = []
        fgf = FeatureGroupFiltering()
        scenario = fgf.fit_transform(scenario, config)

        imputer = ImputerWrapper()
        scenario = imputer.fit_transform(scenario, config)

        scaler = StandardScalerWrapper()
        scenario = scaler.fit_transform(scenario, config)

        pca = PCAWrapper()
        scenario = pca.fit_transform(scenario, config)

        return scenario, [fgf, imputer, scaler, pca]

    def fit_pre_solving(self, scenario: ASlibScenario, config: Configuration):
        '''
            fits an pre-solving schedule using Aspeed [Hoos et al, 2015 TPLP) 

            Arguments
            ---------
            scenario: aslib_scenario.aslib_scenario.ASlibScenario
                aslib scenario at hand
            config: Configuration
                parameter configuration to use for preprocessing

            Returns
            -------
            instance of Aspeed() with a fitted pre-solving schedule if performance_type of scenario is runtime; else None
        '''
        if scenario.performance_type[0] == "runtime":
            aspeed = Aspeed()
            aspeed.fit(scenario=scenario, config=config)
            return aspeed
        else:
            return None

    def fit_selector(self, scenario: ASlibScenario, config: Configuration):
        '''
            fits an algorithm selector for a given scenario wrt a given configuration

            Arguments
            ---------
            scenario: aslib_scenario.aslib_scenario.ASlibScenario
                aslib scenario at hand
            config: Configuration
                parameter configuration
        '''

        if config.get("selector") == "PairwiseClassifier":
            clf_class = None
            if config.get("classifier") == "RandomForest":
                clf_class = RandomForest
            if config.get("classifier") == "XGBoost":
                clf_class = XGBoost

            selector = PairwiseClassifier(classifier_class=clf_class)
            selector.fit(scenario=scenario, config=config)

        if config.get("selector") == "MultiClassifier":
            clf_class = None
            if config.get("classifier") == "RandomForest":
                clf_class = RandomForest
            if config.get("classifier") == "XGBoost":
                clf_class = XGBoost

            selector = MultiClassifier(classifier_class=clf_class)
            selector.fit(scenario=scenario, config=config)

        if config.get("selector") == "IndRegressor":
            reg_class = None
            if config.get("regressor") == "RandomForestRegressor":
                reg_class = RandomForestRegressor
                
            selector = IndRegression(regressor_class=reg_class)
            selector.fit(scenario=scenario, config=config)
            
        if config.get("selector") == "JointRegressor":
            reg_class = None
            if config.get("regressor") == "RandomForestRegressor":
                reg_class = RandomForestRegressor
                
            selector = JointRegression(regressor_class=reg_class)
            selector.fit(scenario=scenario, config=config)

        if config.get("selector") == "PairwiseRegressor":
            reg_class = None
            if config.get("regressor") == "RandomForestRegressor":
                reg_class = RandomForestRegressor
                
            selector = PairwiseRegression(regressor_class=reg_class)
            selector.fit(scenario=scenario, config=config)

        return selector

    def predict(self, scenario: ASlibScenario, config: Configuration, feature_pre_pipeline: list, pre_solver: Aspeed, selector):
        '''
            predicts algorithm schedules wrt a given config
            and given pipelines

            Arguments
            ---------
            scenario: aslib_scenario.aslib_scenario.ASlibScenario
                aslib scenario at hand
            config: Configuration
                parameter configuration
            feature_pre_pipeline: list
                list of fitted feature preprocessors
            pre_solver: Aspeed
                pre solver object with a saved static schedule
            selector: autofolio.selector.*
                fitted selector object
        '''

        self.logger.info("Predict on Test")
        for f_pre in feature_pre_pipeline:
            scenario = f_pre.transform(scenario)

        if pre_solver:
            pre_solving_schedule = pre_solver.predict(scenario=scenario)
        else:
            pre_solving_schedule = {}

        pred_schedules = selector.predict(scenario=scenario)

        # combine schedules
        if pre_solving_schedule:
            return dict((inst, pre_solving_schedule.get(inst, []) + schedule) for inst, schedule in pred_schedules.items())
        else:
            return pred_schedules


def main():
    af = AutoFolio()
    af.run_cli()


if __name__ == "__main__":
    main()
