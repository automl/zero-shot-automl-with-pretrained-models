import numpy as np

from autofolio.facade.af_csv_facade import AFCsvFacade

__author__ = "Marius Lindauer"
__license__ = "BSD"
__version__ = "2.1.0"

perf_fn = "perf.csv"
feat_fn = "feats.csv"

# will be created (or overwritten) by AutoFolio
model_fn = "af_model.pkl"

af = AFCsvFacade(perf_fn=perf_fn, feat_fn=feat_fn)

# load AutoFolio model and
# get predictions for new meta-feature vector
pred = AFCsvFacade.load_and_predict(vec=np.array([1.]), load_fn=model_fn)

print(pred)


