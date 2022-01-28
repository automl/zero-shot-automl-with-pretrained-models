import pickle
import numpy as np

from autofolio.facade.af_csv_facade import AFCsvFacade

__author__ = "Marius Lindauer"
__license__ = "BSD"
__version__ = "2.1.0"

perf_fn = "/home/lindauer/git/AutoDLComp19/experiments/new_config_space_after_March_4/eval_new_cs_old_data_approx_600/perf_matrix.csv"
feat_fn = "/home/lindauer/git/AutoDLComp19/src/meta_features/non-nn/old_data/meta_features.csv"


# will be created (or overwritten) by AutoFolio
old_model_fn = "af_model.pkl"
new_model_fn = "af_model_final.pkl"
config_fn = "config.pkl"

af = AFCsvFacade(perf_fn=perf_fn, feat_fn=feat_fn)

#with open(old_model_fn, "br") as fp:
#_, _, _, _, config = pickle.load(fp)

#with open(config_fn, "bw") as fp:
#    pickle.dump(config, fp)

with open(config_fn, "br") as fp:
    config = pickle.load(fp)

print(config)

# evaluate configuration using a 10-fold cross validation
score = af.cross_validation(config=config)

# fit AF using a loaded configuration on all data!
af.fit(config=config, save_fn=new_model_fn)

# load AutoFolio model and
# get predictions for new meta-feature vector
pred = AFCsvFacade.load_and_predict(vec=np.array([3.,102., 6149., -1.,]), load_fn=new_model_fn)

print(pred)