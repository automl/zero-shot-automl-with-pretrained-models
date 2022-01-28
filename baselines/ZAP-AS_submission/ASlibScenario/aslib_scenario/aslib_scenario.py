import os
import sys
import logging
import yaml
import functools
import arff  # liac-arff
import copy
import collections

from sklearn.model_selection import KFold
import pandas as pd
import numpy as np

__author__ = "Marius Lindauer"
__version__ = "2.0.0"
__license__ = "BSD"

MAXINT = 2**32


class ASlibScenario(object):
    '''
        all data about an algorithm selection scenario
    '''

    def __init__(self):
        '''
            Constructor
        '''

        self.logger = logging.getLogger("ASlibScenario")

        # listed in description.txt
        self.scenario = None  # string
        self.performance_measure = []  # list of strings
        self.performance_type = []  # list of "runtime" or "solution_quality"
        self.maximize = []  # list of "true" or "false"
        self.algorithm_cutoff_time = None  # float
        self.algorithm_cutoff_memory = None  # integer
        self.features_cutoff_time = None  # float
        self.features_cutoff_memory = None  # integer
        self.features_deterministic = []  # list of strings
        self.features_stochastic = []  # list of strings
        self.algorithms = []  # list of strings
        self.algortihms_deterministics = []  # list of strings
        self.algorithms_stochastic = []  # list of strings
        self.feature_group_dict = {}  # string -> [] of strings
        self.feature_steps = []
        self.feature_steps_default = []

        # extracted in other files
        self.features = []
        self.ground_truths = {}  # type -> [values]

        self.feature_data = None
        self.performance_data = None
        self.performance_data_all = []
        self.runstatus_data = None
        self.feature_cost_data = None
        self.feature_runstatus_data = None
        self.ground_truth_data = None
        self.cv_data = None

        self.instances = None  # list

        self.found_files = []
        self.read_funcs = {
            "description.txt": self.read_description,
            "algorithm_runs.arff": self.read_algorithm_runs,
            "feature_costs.arff": self.read_feature_costs,
            "feature_values.arff": self.read_feature_values,
            "feature_runstatus.arff": self.read_feature_runstatus,
            "ground_truth.arff": self.read_ground_truth,
            "cv.arff": self.read_cv
        }

        self.CHECK_VALID = True

    def __getstate__(self):
        '''
            method for pickling the object;
        '''
        #  state_dict = copy.copy(self.__dict__)
        state_dict = self.__dict__

        # adding explicitly the feature names as used before
        state_dict["feature_names"] = list(self.feature_data.columns)

        return state_dict

    def read_from_csv(self, perf_fn: str, 
                      feat_fn: str, 
                      objective: str, 
                      runtime_cutoff: float, 
                      maximize: bool,
                      cv_fn: str=None):
        '''
            create an internal ASlib scenario from csv

            Arguments
            ---------
            perf_fn: str
                performance file name in csv format
            feat_fn: str
                instance feature file name in csv format
            objective: str
                "solution_quality" or "runtime"
            runtime_cutoff: float
                maximal runtime cutoff
            maximize: bool
                whether to maximize or minimize the objective values
            cv_fn: str
                cv split file in csv format
        '''

        self.scenario = None  # string
        self.performance_measure = ["dummy"]  # list of strings
        # list of "runtime" or "solution_quality"
        self.performance_type = [objective]
        self.maximize = [maximize]  # list of "true" or "false"
        self.algorithm_cutoff_time = runtime_cutoff  # float
        self.algorithm_cutoff_memory = None  # integer
        self.features_cutoff_time = None  # float
        self.features_cutoff_memory = None  # integer

        self.feature_data = pd.read_csv(feat_fn, index_col=0)
        self.performance_data = pd.read_csv(perf_fn, index_col=0)
        self.performance_data_all = [self.performance_data]

        self.algorithms = list(
            self.performance_data.columns)  # list of strings
        # self.algortihms_deterministics = self.algorithms  # list of strings
        # self.algorithms_stochastic = []  # list of strings

        self.features_deterministic = list(
            self.feature_data.columns)  # list of strings
        self.features_stochastic = []  # list of strings
        self.feature_group_dict = {
            "all": {"provides": self.features_deterministic}}
        self.feature_steps = ["all"]
        self.feature_steps_default = ["all"]

        self.instances = list(self.feature_data.index)  # lis

        self.runstatus_data = pd.DataFrame(
            data=np.array(
                [["ok"] * len(self.algorithms)] * len(self.instances)),
            index=self.performance_data.index,
            columns=self.performance_data.columns)

        if objective == "runtime":
            self.runstatus_data[
                self.performance_data >= runtime_cutoff] = "timeout"

        self.feature_runstatus_data = pd.DataFrame(
            data=["ok"] * len(self.instances), index=self.instances, columns=["all"])

        self.feature_cost_data = None
        self.ground_truth_data = None

        # extracted in other files
        self.features = self.features_deterministic
        self.ground_truths = {}  # type -> [values]

        if cv_fn:
            self.cv_data = pd.read_csv(cv_fn, index_col=0)
        else:
            self.create_cv_splits()

        if self.CHECK_VALID:
            self.check_data()

    def read_scenario(self, dn):
        '''
            read an ASlib scenario from disk

            Arguments
            ---------
            dn: str
                directory name with ASlib files
        '''
        self.logger.info("Read ASlib scenario: %s" % (dn))

        # add command line arguments in metainfo
        self.dir_ = dn
        self.find_files()
        self.read_files()

        if self.CHECK_VALID:
            self.check_data()

    def find_files(self):
        '''
            find all expected files in self.dir_
            fills self.found_files
        '''
        expected = ["description.txt", "algorithm_runs.arff",
                    "feature_values.arff", "feature_runstatus.arff"]
        optional = ["ground_truth.arff", "feature_costs.arff", "cv.arff"]

        for expected_file in expected:
            full_path = os.path.join(self.dir_, expected_file)
            if not os.path.isfile(full_path):
                self.logger.error("Required file not found: %s" % (full_path))
                sys.exit(2)
            else:
                self.found_files.append(full_path)

        for expected_file in optional:
            full_path = os.path.join(self.dir_, expected_file)
            if not os.path.isfile(full_path):
                self.logger.warning(
                    "Optional file not found: %s" % (full_path))
            else:
                self.found_files.append(full_path)

    def read_files(self):
        '''
            iterates over all found files (self.found_files) and 
            calls the corresponding function to validate file
        '''
        for fn in self.found_files:
            read_func = self.read_funcs.get(os.path.basename(fn))
            if read_func:
                read_func(fn)

    def read_description(self, fn):
        '''
            reads description file
            and saves all meta information
        '''
        self.logger.info("Read %s" % (fn))

        with open(fn, "r") as fh:
            description = yaml.load(fh, Loader=yaml.SafeLoader)

        self.scenario = description.get('scenario_id')
        self.performance_measure = description.get('performance_measures')

        self.performance_measure = description.get('performance_measures') if isinstance(description.get('performance_measures'), list) else \
            [description.get('performance_measures')]

        maximize = description.get('maximize')
        self.maximize = maximize if isinstance(maximize, list) else [maximize]
        for maxi in self.maximize:
            if not isinstance(maxi, bool):
                raise ValueError(
                    "\"maximize\" in description.txt has to be a bool (i.e., not a string).")

        performance_type = description.get('performance_type')
        self.performance_type  = performance_type if isinstance(performance_type, list) else \
            [performance_type]

        self.algorithm_cutoff_time = description.get('algorithm_cutoff_time')
        self.algorithm_cutoff_memory = description.get(
            'algorithm_cutoff_memory')
        self.features_cutoff_time = description.get('features_cutoff_time')
        self.features_cutoff_memory = description.get('features_cutoff_memory')
        self.features_deterministic = description.get('features_deterministic')
        if self.features_deterministic is None:
            self.features_deterministic = set()
        self.features_stochastic = description.get('features_stochastic')
        if self.features_stochastic is None:
            self.features_stochastic = set()
        self.feature_group_dict = description.get('feature_steps')
        self.feature_steps = list(self.feature_group_dict.keys())
        self.feature_steps_default = description.get('default_steps')

        for step, d in self.feature_group_dict.items():
            if d.get("requires") and not isinstance(d["requires"], list):
                self.feature_group_dict[step]["requires"] = [d["requires"]]

        for algo, meta_data in description.get("metainfo_algorithms").items():
            self.algorithms.append(algo)
            if meta_data["deterministic"]:
                self.algortihms_deterministics.append(algo)
            else:
                self.algorithms_stochastic.append(algo)

        # if algorithms as numerical IDs, yaml interprets them as integers and
        # not as string
        self.algorithms = list(map(str, self.algorithms))

        # ERRORS
        error_found = False
        if not self.scenario:
            self.logger.warning("Have not found SCENARIO_ID")
        if not self.performance_measure or self.performance_measure == "?":
            self.logger.error("Have not found PERFORMANCE_MEASURE")
            error_found = True
        if not self.performance_type or self.performance_type == "?":
            self.logger.error("Have not found PERFORMANCE_TYPE")
            error_found = True
        if not self.maximize or self.maximize == "?":
            self.logger.error("Have not found MAXIMIZE")
            error_found = True
        if (not self.algorithm_cutoff_time or self.algorithm_cutoff_time == "?") and (self.performance_type == "quality"):
            self.logger.error("Have not found algorithm_cutoff_time")
            error_found = True
        elif self.algorithm_cutoff_time == "?":
            self.algorithm_cutoff_time = None
        if not self.feature_group_dict:
            self.logger.error("Have not found any feature step")
            error_found = True

        if error_found:
            sys.exit(3)

        # WARNINGS
        if not self.algorithm_cutoff_memory or self.algorithm_cutoff_memory == "?":
            self.logger.warning("Have not found algorithm_cutoff_memory")
            self.algorithm_cutoff_memory = None
        if not self.features_cutoff_time or self.features_cutoff_time == "?":
            self.logger.warning("Have not found features_cutoff_time")
            self.logger.debug(
                "Assumption FEATURES_CUTOFF_TIME == ALGORITHM_CUTOFF_TIME ")
            self.features_cutoff_time = self.algorithm_cutoff_time
        if not self.features_cutoff_memory or self.features_cutoff_memory == "?":
            self.logger.warning("Have not found features_cutoff_memory")
            self.features_cutoff_memory = None
        if not self.features_deterministic:
            self.logger.warning("Have not found features_deterministic")
            self.features_deterministic = []
        if not self.features_stochastic:
            self.logger.warning("Have not found features_stochastic")
            self.features_stochastic = []

        feature_intersec = set(self.features_deterministic).intersection(
            self.features_stochastic)
        if feature_intersec:
            self.logger.warning("Intersection of deterministic and stochastic features is not empty: %s" % (
                str(feature_intersec)))
        algo_intersec = set(self.algortihms_deterministics).intersection(
            self.algorithms_stochastic)
        if algo_intersec:
            self.logger.warning(
                "Intersection of deterministic and stochastic algorithms is not empty: %s" % (str(algo_intersec)))

        if self.performance_type[0] == "solution_quality":
            self.algorithm_cutoff_time = 1  # pseudo number for schedules
            self.logger.debug(
                "Since we optimize quality, we use runtime cutoff of 1.")

    def read_algorithm_runs(self, fn):
        '''
            read performance file
            and saves information
            add Instance() in self.instances

            unsuccessful runs are replaced by algorithm_cutoff_time if performance_type is runtime

            EXPECTED HEADER:
            @RELATION ALGORITHM_RUNS_2013-SAT-Competition

            @ATTRIBUTE instance_id STRING
            @ATTRIBUTE repetition NUMERIC
            @ATTRIBUTE algorithm STRING
            @ATTRIBUTE PAR10 NUMERIC
            @ATTRIBUTE Number_of_satisfied_clauses NUMERIC
            @ATTRIBUTE runstatus {ok, timeout, memout, not_applicable, crash, other}
        '''
        self.logger.info("Read %s" % (fn))

        with open(fn, "r") as fp:
            try:
                arff_dict = arff.load(fp)
            except arff.BadNominalValue:
                self.logger.error(
                    "Parsing of arff file failed (%s) - maybe conflict of header and data." % (fn))

        if arff_dict["attributes"][0][0].upper() != "INSTANCE_ID":
            self.logger.error(
                "instance_id as first attribute is missing in %s" % (fn))
            sys.exit(3)
        if arff_dict["attributes"][1][0].upper() != "REPETITION":
            self.logger.error(
                "repetition as second attribute is missing in %s" % (fn))
            sys.exit(3)
        if arff_dict["attributes"][2][0].upper() != "ALGORITHM":
            self.logger.error(
                "algorithm as third attribute is missing in %s" % (fn))
            sys.exit(3)

        i = 0
        for performance_measure in self.performance_measure:
            if arff_dict["attributes"][3 + i][0].upper() != performance_measure.upper():
                self.logger.error(
                    "\"%s\" as attribute is missing in %s" % (performance_measure, fn))
                sys.exit(3)
            i += 1

        if arff_dict["attributes"][3 + i][0].upper() != "RUNSTATUS":
            self.logger.error(
                "runstatus as last attribute is missing in %s" % (fn))
            sys.exit(3)

        algo_inst_col = ['instance_id', 'repetition', 'algorithm']
        perf_col = []
        for perf in self.performance_measure:
            perf_col.append(perf)
        status_col = ['runstatus']

        perf_data = pd.DataFrame(arff_dict['data'],
                                 columns=algo_inst_col + perf_col + status_col)

        # group performance data by mean value across repetitions
        for perf in self.performance_measure:
            self.performance_data_all.append(
                perf_data.groupby(['instance_id', 'algorithm']).median().unstack(
                    'algorithm')[perf]
            )

        self.performance_data = self.performance_data_all[0]

        # group runstatus by most frequent runstatus across repetitions
        self.runstatus_data = \
            perf_data.groupby(['instance_id', 'algorithm'])["runstatus"].aggregate(
                lambda x: collections.Counter(x).most_common(1)[0][0]
            ).unstack('algorithm')

        if self.performance_data.isnull().sum().sum() > 0:
            self.logger.error("Performance data has missing values")
            sys.exit(3)

        self.instances = list(self.performance_data.index)

    def read_feature_values(self, fn):
        '''
            reads feature file
            and saves them in self.instances

            Expected Header:
            @RELATION FEATURE_VALUES_2013-SAT-Competition

            @ATTRIBUTE instance_id STRING
            @ATTRIBUTE repetition NUMERIC
            @ATTRIBUTE number_of_variables NUMERIC
            @ATTRIBUTE number_of_clauses NUMERIC
            @ATTRIBUTE first_local_min_steps NUMERIC
        '''

        self.logger.info("Read %s" % (fn))

        with open(fn, "r") as fp:
            try:
                arff_dict = arff.load(fp)
            except arff.BadNominalValue:
                self.logger.error(
                    "Parsing of arff file failed (%s) - maybe conflict of header and data." % (fn))
                sys.exit(3)

        if arff_dict["attributes"][0][0].upper() != "INSTANCE_ID":
            self.logger.error(
                "instance_id as first attribute is missing in %s" % (fn))
            sys.exit(3)
        if arff_dict["attributes"][1][0].upper() != "REPETITION":
            self.logger.error(
                "repetition as second attribute is missing in %s" % (fn))
            sys.exit(3)

        feature_set = set(self.features_deterministic).union(
            self.features_stochastic)

        for f_name in arff_dict["attributes"][2:]:
            f_name = f_name[0]
            self.features.append(f_name)
            if not f_name in feature_set:
                self.logger.error(
                    "Feature \"%s\" was not defined as deterministic or stochastic" % (f_name))
                sys.exit(3)

        pairs_inst_rep = []
        encoutered_features = []
        inst_feats = {}
        for data in arff_dict["data"]:
            inst_name = data[0]
            repetition = data[1]
            features = data[2:]

            if len(features) != len(self.features):
                self.logger.error(
                    "Number of features in attributes does not match number of found features; instance: %s" % (inst_name))
                sys.exit(3)

            # TODO: handle feature repetitions
            inst_feats[inst_name] = features

            #===================================================================
            # # not only Nones in feature vector and previously seen
            # if functools.reduce(lambda x, y: True if (x or y) else False, features, False) and features in encoutered_features:
            #     self.logger.warning(
            #         "Feature vector found twice: %s" % (",".join(map(str, features))))
            # else:
            #     encoutered_features.append(features)
            #===================================================================

            if (inst_name, repetition) in pairs_inst_rep:
                self.logger.warning(
                    "Pair (%s,%s) is not unique in %s" % (inst_name, repetition, fn))
            else:
                pairs_inst_rep.append((inst_name, repetition))

        # convert to pandas
        cols = list(map(lambda x: x[0], arff_dict["attributes"]))
        self.feature_data = pd.DataFrame(arff_dict["data"], columns=cols)

        self.feature_data = self.feature_data.groupby(['instance_id']).aggregate(np.mean)
        self.feature_data = self.feature_data.drop("repetition", axis=1)
        
        duplicates = self.feature_data.duplicated().sum()
        if duplicates > 0:
            self.logger.warn("Found %d duplicated feature vectors" %(duplicates))
            self.logger.warn(self.feature_data[self.feature_data.duplicated(keep=False)].index)

    def read_feature_costs(self, fn):
        '''
            reads feature time file
            and saves in self.instances

            Expected header:
            @RELATION FEATURE_COSTS_2013-SAT-Competition

            @ATTRIBUTE instance_id STRING
            @ATTRIBUTE repetition NUMERIC
            @ATTRIBUTE preprocessing NUMERIC
            @ATTRIBUTE local_search_probing NUMERIC

        '''
        self.logger.info("Read %s" % (fn))

        with open(fn, "r") as fp:
            try:
                arff_dict = arff.load(fp)
            except arff.BadNominalValue:
                self.logger.error(
                    "Parsing of arff file failed (%s) - maybe conflict of header and data." % (fn))
                sys.exit(3)

        if arff_dict["attributes"][0][0].upper() != "INSTANCE_ID":
            self.logger.error(
                "\"instance_id\" as first attribute is missing in %s" % (fn))
            sys.exit(3)
        if arff_dict["attributes"][1][0].upper() != "REPETITION":
            self.logger.error(
                "\"repetition\" as second attribute is missing in %s" % (fn))
            sys.exit(3)
        found_groups = list(
            map(str, sorted(map(lambda x: x[0], arff_dict["attributes"][2:]))))
        for meta_group in self.feature_group_dict.keys():
            if meta_group not in found_groups:
                self.logger.error(
                    "\"%s\" as attribute is missing in %s" % (meta_group, fn))
                sys.exit(3)

        inst_cost = {}

        # impute missing values with 0
        # convert to pandas
        data = np.array(arff_dict["data"])
        cols = list(map(lambda x: x[0], arff_dict["attributes"][1:]))
        imputed_feature_cost_data = pd.DataFrame(
            data[:,1:], columns=cols, dtype=np.float)
        
        # imputation has to be before the grouping
        imputed_feature_cost_data[pd.isnull(imputed_feature_cost_data)] = 0
        
        # instance panda
        cols = list(map(lambda x: x[0], arff_dict["attributes"][:1]))
        instance_data = pd.DataFrame(
            data[:,:1], columns=cols)
        
        self.feature_cost_data = pd.concat([instance_data, imputed_feature_cost_data], axis=1)

        self.feature_cost_data = self.feature_cost_data.groupby(
            ['instance_id']).median()
        self.feature_cost_data = self.feature_cost_data.drop("repetition", axis=1)
        
    def read_feature_runstatus(self, fn):
        '''
            reads run stati of all pairs instance x feature step
            and saves them self.instances

            Expected header:
            @RELATION FEATURE_RUNSTATUS_2013 - SAT - Competition
            @ATTRIBUTE instance_id STRING
            @ATTRIBUTE repetition NUMERIC
            @ATTRIBUTE preprocessing { ok , timeout , memout , presolved , crash , other }
            @ATTRIBUTE local_search_probing { ok , timeout , memout , presolved , crash , other }
        '''
        self.logger.info("Read %s" % (fn))

        with open(fn, "r") as fp:
            try:
                arff_dict = arff.load(fp)
            except arff.BadNominalValue:
                self.logger.error(
                    "Parsing of arff file failed (%s) - maybe conflict of header and data." % (fn))
                sys.exit(3)

        if arff_dict["attributes"][0][0].upper() != "INSTANCE_ID":
            self.logger.error(
                "instance_id as first attribute is missing in %s" % (fn))
            sys.exit(3)
        if arff_dict["attributes"][1][0].upper() != "REPETITION":
            self.logger.error(
                "repetition as second attribute is missing in %s" % (fn))
            sys.exit(3)

        for f_name in arff_dict["attributes"][2:]:
            f_name = f_name[0]
            if not f_name in self.feature_group_dict.keys():
                self.logger.error(
                    "Feature step \"%s\" was not defined in feature steps" % (f_name))
                sys.exit(3)

        if len(self.feature_group_dict.keys()) != len(arff_dict["attributes"][2:]):
            self.logger.error("Number of feature steps in description.txt (%d) and feature_runstatus.arff (%d) does not match." % (
                len(self.feature_group_dict.keys()), len(arff_dict["attributes"][2:-1])))
            sys.exit(3)

        # convert to pandas
        cols = list(map(lambda x: x[0], arff_dict["attributes"]))
        self.feature_runstatus_data = pd.DataFrame(arff_dict["data"], columns=cols)

        self.feature_runstatus_data = self.feature_runstatus_data.groupby(\
            ['instance_id']).aggregate(lambda x: collections.Counter(x).most_common(1)[0][0])
            
        self.feature_runstatus_data = self.feature_runstatus_data.drop("repetition", axis=1)

    def read_ground_truth(self, fn):
        '''
            read ground truths of all instances
            and save them in self.instances

            @RELATION GROUND_TRUTH_2013-SAT-Competition

            @ATTRIBUTE instance_id STRING
            @ATTRIBUTE SATUNSAT {SAT,UNSAT}
            @ATTRIBUTE OPTIMAL_VALUE NUMERIC
        '''

        self.logger.info("Read %s" % (fn))

        with open(fn, "r") as fp:
            try:
                arff_dict = arff.load(fp)
            except arff.BadNominalValue:
                self.logger.error(
                    "Parsing of arff file failed (%s) - maybe conflict of header and data." % (fn))
                sys.exit(3)

        if arff_dict["attributes"][0][0].upper() != "INSTANCE_ID":
            self.logger.error(
                "instance_id as first attribute is missing in %s" % (fn))
            sys.exit(3)

        # extract feature names
        for attr in arff_dict["attributes"][1:]:
            self.ground_truths[attr[0]] = attr[1]

        # convert to panda
        data = np.array(arff_dict["data"])
        cols = list(map(lambda x: x[0], arff_dict["attributes"][1:]))
        self.ground_truth_data = pd.DataFrame(
            data=data[:, 1:], index=data[:, 0].tolist(), columns=cols)

    def read_cv(self, fn):
        '''
            read cross validation <fn>
            only save first cv repetition!

            @RELATION CV_2013 - SAT - Competition
            @ATTRIBUTE instance_id STRING
            @ATTRIBUTE repetition NUMERIC
            @ATTRIBUTE fold NUMERIC
        '''
        self.logger.info("Read %s" % (fn))

        with open(fn, "r") as fp:
            try:
                arff_dict = arff.load(fp)
            except arff.BadNominalValue:
                self.logger.error(
                    "Parsing of arff file failed (%s) - maybe conflict of header and data." % (fn))
                sys.exit(3)

        if arff_dict["attributes"][0][0].upper() != "INSTANCE_ID":
            self.logger.error(
                "instance_id as first attribute is missing in %s" % (fn))
            sys.exit(3)
        if arff_dict["attributes"][1][0].upper() != "REPETITION":
            self.logger.error(
                "repetition as second attribute is missing in %s" % (fn))
            sys.exit(3)
        if arff_dict["attributes"][2][0].upper() != "FOLD":
            self.logger.error(
                "fold as third attribute is missing in %s" % (fn))
            sys.exit(3)

        # convert to pandas
        data = np.array(arff_dict["data"])
        cols = list(map(lambda x: x[0], arff_dict["attributes"][1:]))
        self.cv_data = pd.DataFrame(
            data[:, 1:], index=data[:, 0], columns=cols, dtype=np.float)
        # use only first cv repetitions
        self.cv_data = self.cv_data[self.cv_data["repetition"] == 1]
        self.cv_data = self.cv_data.drop("repetition", axis=1)

    def check_data(self):
        '''
            checks whether all data objects are valid according to ASlib specification
            and makes some transformations
        '''

        for perf_type_i, perf_type in enumerate(self.performance_type):

            if pd.isnull(self.performance_data_all[perf_type_i]).sum().sum() > 0:
                self.logger.error("Performance data cannot have missing entries")
                sys.exit(3)

            if perf_type == "runtime" and self.maximize[perf_type_i]:
                self.logger.error("Maximizing runtime is not supported")
                sys.exit(3)

            if perf_type == "runtime":
                # replace all non-ok scores with par10 values
                self.logger.debug(
                    "Replace all runtime data with PAR10 values for non-OK runs")
                self.performance_data_all[perf_type_i][
                    self.runstatus_data != "ok"] = self.algorithm_cutoff_time * 10

            if perf_type == "solution_quality" and self.maximize[perf_type_i]:
                self.logger.info(
                    "Multiply all performance data by -1, since autofolio minimizes the scores but the objective is to maximize")
                self.performance_data_all[perf_type_i] *= -1

        all_data = [self.feature_data, self.feature_cost_data,
                    self.feature_runstatus_data, self.ground_truth_data,
                    self.cv_data]

        for perf_data in self.performance_data_all:
            all_data.append(perf_data)

        # all data should have the same instances
        set_insts = set(self.instances)
        for data in all_data:
            if data is not None and set_insts.difference(data.index):
                self.logger.error("Not all data matrices have the same instances: %s" % (
                    set_insts.difference(data.index)))
                sys.exit(3)

            # each instance should be listed only once
            if data is not None and len(list(set(data.index))) != len(data.index):
                self.logger.error(all_data)
                self.logger.error("Some instances are listed more than once")
                sys.exit(3)

    def get_split(self, indx=1):
        '''
            returns a copy of self but only with the data of the i-th cross validation split according to cv.arff

            Arguments
            ---------
                indx : int
                    indx of the cv split (should be in most cases within [1,10]

            Returns
            -------
                training split : ASlibScenario
                test split : ASlibScenario
        '''

        if self.cv_data is None:
            self.logger.warning(
                "The ASlib scenario has not provided any cv.arff; create CV split...")
            self.create_cv_splits()

        test_insts = self.cv_data[
            self.cv_data["fold"] == float(indx)].index.tolist()
        training_insts = self.cv_data[
            self.cv_data.fold != float(indx)].index.tolist()

        test = copy.copy(self)
        training = copy.copy(self)

        # feature_data
        test.feature_data = test.feature_data.drop(training_insts).sort_index()
        training.feature_data = training.feature_data.drop(
            test_insts).sort_index()
        # performance_data
        test.performance_data = test.performance_data.drop(
            training_insts).sort_index()
        training.performance_data = training.performance_data.drop(
            test_insts).sort_index()
        # runstatus_data
        test.runstatus_data = test.runstatus_data.drop(
            training_insts).sort_index()
        training.runstatus_data = training.runstatus_data.drop(
            test_insts).sort_index()
        # self.feature_runstatus_data
        test.feature_runstatus_data = test.feature_runstatus_data.drop(
            training_insts).sort_index()
        training.feature_runstatus_data = training.feature_runstatus_data.drop(
            test_insts).sort_index()
        # feature_cost_data
        if self.feature_cost_data is not None:
            test.feature_cost_data = test.feature_cost_data.drop(
                training_insts).sort_index()
            training.feature_cost_data = training.feature_cost_data.drop(
                test_insts).sort_index()
        # ground_truth_data
        if self.ground_truth_data is not None:
            test.ground_truth_data = test.ground_truth_data.drop(
                training_insts).sort_index()
            training.ground_truth_data = training.ground_truth_data.drop(
                test_insts).sort_index()
        test.cv_data = None
        training.cv_data = None

        test.instances = test_insts
        training.instances = training_insts

        self.used_feature_groups = None

        return test, training

    def create_cv_splits(self, n_folds: int=10):
        '''
            creates cv splits and saves them in self.cv_data

            Argumnents
            ----------
            n_folds: int
                number of splits
        '''

        kf = KFold(n_splits=n_folds, shuffle=True)
        self.cv_data = pd.DataFrame(
            data=np.zeros(len(self.instances)), index=self.instances, columns=["fold"], dtype=np.float)
        
        for indx, (train, test) in enumerate(kf.split(self.instances)):
            # print(self.cv_data.loc(np.array(self.instances[test]).tolist()))
            self.cv_data.iloc[test] = indx + 1.

    def change_perf_measure(self, measure_idx: int = None, measure_name: str = None):
        '''
            change self.performance_data to another performance measure. 
            Either measure_idx or measure_name needs to be specified --
            measure_name overwrites measure_idx

            Arguments
            ---------
            measure_idx : int
                index of performance measure
            measure_name: str
                name of performance measure
        '''
        if measure_name:
            measure_idx = self.performance_measure.index(measure_name)

        if measure_idx:
            self.performance_data = self.performance_data_all[measure_idx]
