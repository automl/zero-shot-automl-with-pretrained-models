import numpy as np

from autofolio.facade.af_csv_facade import AFCsvFacade

__author__ = "Marius Lindauer"
__license__ = "BSD"
__version__ = "2.1.0"

# will be created (or overwritten) by AutoFolio
model_fn = "af_model_final.pkl"

# load AutoFolio model and
# get predictions for new meta-feature vector
pred = AFCsvFacade.load_and_predict(vec=np.array([3.,102., 6149., -1.,]), load_fn=model_fn)

print(pred)


