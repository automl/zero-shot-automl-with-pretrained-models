import argparse
import sys
import os
import logging

__author__ = "Marius Lindauer"
__version__ = "2.0.0"
__license__ = "BSD"


class CMDParser(object):

    def __init__(self):
        '''
            Constructor
        '''
        self.logger = logging.getLogger("CMDParser")

        self._arg_parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        aslib = self._arg_parser.add_argument_group(
            "Reading from ASlib Format")
        aslib.add_argument("-s", "--scenario", default=None,
                           help="directory with ASlib scenario files (required if not using --load or csv input files")

        csv = self._arg_parser.add_argument_group("Reading from CSV Format")
        csv.add_argument("--performance_csv", default=None,
                         help="performance data in csv table (column: algorithm, row: instance, delimeter: ,)")
        csv.add_argument("--feature_csv", default=None,
                         help="instance features data in csv table (column: features, row: instance, delimeter: ,)")
        csv.add_argument("--performance_test_csv", default=None,
                         help="performance *test* data in csv table (column: algorithm, row: instance, delimeter: ,)")
        csv.add_argument("--feature_test_csv", default=None,
                         help="instance *test* features data in csv table (column: features, row: instance, delimeter: ,)")
        csv.add_argument("--cv_csv", default=None,
                         help="cross validation splits in csv table (column: split ID, row: instance, delimeter: ,)")
        csv.add_argument("--objective", default="solution_quality", choices=[
                         "runtime", "solution_quality"], help="Are the objective values in the performance data runtimes or an arbitrary solution quality (or cost) value")
        csv.add_argument("--runtime_cutoff", default=None, type=float,
                         help="cutoff time for each algorithm run for the performance data")
        csv.add_argument("--maximize", default=False, action="store_true", help="Set this parameter to indicate maximization of the performance metric (default: minimization)")

        opt = self._arg_parser.add_argument_group("Optional Options")
        opt.add_argument("-t", "--tune", action="store_true", default=False,
                         help="uses SMAC3 to determine a better hyperparameter configuration")
        opt.add_argument("--smac_seed", default=42, type=int,
                         help="Seed passed to SMAC")
        opt.add_argument("-p", "--pcs", default=None,
                         help="pcs file to be read")
        opt.add_argument("--output_dir", default=None,
                         help="output directory of SMAC")
        opt.add_argument("--runcount_limit", type=int, default=42,
                         help="maximal number of AS evaluations (SMAC budget)")
        opt.add_argument("--wallclock_limit", type=int, default=300,
                         help="wallclock time limit in sec (SMAC budget)")
        opt.add_argument(
            "-v", "--verbose", choices=["INFO", "DEBUG"], default="INFO", help="verbose level")
        opt.add_argument("--save", type=str, default=None,
                         help="trains AutoFolio and saves AutoFolio's state in the given filename")
        opt.add_argument("--load", type=str, default=None,
                         help="loads model (from --save); other modes are disabled with this options")
        opt.add_argument("--feature_vec", default=None, type=str,
                         help="feature vector to predict algorithm to use -- has to be used in combination with --load")

        opt.add_argument("--config", type=str, default=None,
                              help="(yaml) config file with run-specific "
                              "configuration options for autofolio")

        outer_cv = self._arg_parser.add_argument_group("Outer Cross-fold Validation Options")

        outer_cv.add_argument("--outer-cv", action="store_true", default=False,
                              help="Use an \"outer\" cross-fold validation scheme "
                              "for tuning to ensure that SMAC does not peek at "
                              "the test set during hyperparameter optimization.")

        outer_cv.add_argument("--outer-cv-fold", type=int, default=None,
                              help="If this argument is given in --outer-cv "
                              "mode, then only the specified outer-cv fold "
                              "will be processed. Presumably, the learned "
                              "model will be saved using --save and the "
                              "results for all folds will be combined later.")

        outer_cv.add_argument("--out-template", type=str, default=None,
                              help="If given, then the fit model and solver "
                              "choices will be saved to this location. The "
                              "string is considered a template. \"$fold\" "
                              "will be replaced with the fold, and "
                              "\"$type\" will be replaced with the "
                              "appropriate file extension, \"pkl\" for the "
                              "models and \"csv\" for the solver choices. See "
                              "string.Template for more details about valid "
                              "tempaltes.")

    def parse(self):
        '''
            uses the self._arg_parser object to parse the cmd line arguments

            Returns
            -------
                parsed arguments
                unknown arguments
        '''

        self.args_, misc_params = self._arg_parser.parse_known_args()

        return self.args_, misc_params

    def _check_args(self):
        '''
            checks whether all provides options are ok (e.g., existence of files)
        '''

        if not os.path.isdir(self.args_.scenario):
            self.logger.error(
                "ASlib Scenario directory not found: %s" % (self.args_.scenario))
            sys.exit(1)
