import pickle
import numpy as np

from autofolio.facade.af_csv_facade import AFCsvFacade

__author__ = "Marius Lindauer"
__license__ = "BSD"
__version__ = "2.1.0"

perf_fn = "perf_matrix.csv"
feat_fn = "meta_features.csv"

af = AFCsvFacade(perf_fn=perf_fn, feat_fn=feat_fn)

config = {'StandardScaler': False, 'fgroup_all': True, 'imputer_strategy': 'mean', 'pca': False, 'selector': 'PairwiseClassifier', 'classifier': 'RandomForest', 'rf:bootstrap': False, 'rf:criterion': 'gini', 'rf:max_depth': 132, 'rf:max_features': 'log2', 'rf:min_samples_leaf': 3, 'rf:min_samples_split': 3, 'rf:n_estimators': 68}

# fit AF using a loaded configuration on all data!
af.fit(config=config)

# load AutoFolio model and
# get predictions for new meta-feature vector
pred = af.predict(vec=np.array([3.,102., 6149., -1.,]))

print(pred)