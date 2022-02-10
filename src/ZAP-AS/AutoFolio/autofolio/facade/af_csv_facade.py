import logging

import numpy as np
from aslib_scenario.aslib_scenario import ASlibScenario
from autofolio.autofolio import AutoFolio
from ConfigSpace.configuration_space import Configuration

__author__ = "Marius Lindauer"
__license__ = "BSD"
__version__ = "2.1.0"


class AFCsvFacade(object):
    def __init__(
        self,
        perf_fn: str,
        feat_fn: str,
        objective: str = "solution_quality",
        runtime_cutoff: float = None,
        maximize: bool = True,
        cv_fn: str = None,
        seed: int = 12345
    ):
        """ Constructor """
        self.scenario = ASlibScenario()
        self.scenario.read_from_csv(
            perf_fn=perf_fn,
            feat_fn=feat_fn,
            objective=objective,
            runtime_cutoff=runtime_cutoff,
            maximize=maximize,
            cv_fn=cv_fn
        )
        self.seed = seed

        self.af = AutoFolio(random_seed=seed)
        self.logger = logging.getLogger("AF Facade")

    def fit(self, config: Configuration = None, save_fn: str = None):
        """ Train AutoFolio on data from init"""
        self.logger.info("Fit")
        if config is None:
            cs = self.af.get_cs(self.scenario, {})
            config = cs.get_default_configuration()
        feature_pre_pipeline, pre_solver, selector = self.af.fit(
            scenario=self.scenario, config=config
        )

        if save_fn:
            self.af._save_model(
                save_fn, self.scenario, feature_pre_pipeline, pre_solver, selector, config
            )
        self.logger.info("AutoFolio model saved to %s" % (save_fn))

    def tune(
        self,
        wallclock_limit: int = 1200,
        runcount_limit: int = np.inf,
    ):

        config = self.af.get_tuned_config(
            self.scenario,
            wallclock_limit=wallclock_limit,
            runcount_limit=runcount_limit,
            autofolio_config={},
            seed=self.seed
        )
        self.logger.info("Optimized Configuration: %s" % (config))
        return config

    def cross_validation(self, config: Configuration):
        """ run a cross validation on given AutoFolio configuration"""
        score = -1 * self.af.run_cv(
            config=config, scenario=self.scenario, folds=int(self.scenario.cv_data.max().max())
        )
        self.logger.info("AF's final performance %f" % (score))

        return score

    @staticmethod
    def load_and_predict(vec: np.ndarray, load_fn: str):
        """ get predicted algorithm for given meta-feature vector"""
        af = AutoFolio(random_seed=42)  # random seed doesn't matter here
        pred = af.read_model_and_predict(model_fn=load_fn, feature_vec=vec)
        print("Selected Schedule [(algorithm, budget)]: %s" % (pred))
        return pred[0][0]
