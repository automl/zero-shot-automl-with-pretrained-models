import numpy as np

from autofolio.facade.af_csv_facade import AFCsvFacade

__author__ = "Marius Lindauer"
__license__ = "BSD"
__version__ = "2.1.0"

perf_fn = "perf_matrix.csv"
feat_fn = "meta_features.csv"


# will be created (or overwritten) by AutoFolio
model_fn = "af_model.pkl"

af = AFCsvFacade(perf_fn=perf_fn, feat_fn=feat_fn)

# fit AutoFolio; will use default hyperparameters of AutoFolio
af.fit()

# tune AutoFolio's hyperparameter configuration for 4 seconds
config = af.tune(wallclock_limit=360)

# evaluate configuration using a 10-fold cross validation
score = af.cross_validation(config=config)

# re-fit AutoFolio using the (hopefully) better configuration
# and save model to disk
af.fit(config=config, save_fn=model_fn)

# load AutoFolio model and
# get predictions for new meta-feature vector
pred = AFCsvFacade.load_and_predict(vec=np.array([3.,102., 6149., -1.,]), load_fn=model_fn)

print(pred)


