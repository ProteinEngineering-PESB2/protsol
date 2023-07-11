import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import Normalizer, RobustScaler
from sklearn.svm import LinearSVR
from tpot.builtins import StackingEstimator
from tpot.export_utils import set_param_recursive

import sys
import json
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from scipy.stats import (kendalltau, pearsonr, spearmanr)
from joblib import dump

name_doc = sys.argv[1]
path_export = sys.argv[2]

tpot_data = pd.read_csv(sys.argv[1], dtype=np.float64)

features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=42)

# Average CV score on the training set was: -587.3885721002492
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=LinearSVR(C=5.0, dual=True, epsilon=0.1, loss="epsilon_insensitive", tol=0.01)),
    StackingEstimator(estimator=GradientBoostingRegressor(alpha=0.75, learning_rate=0.01, loss="huber", max_depth=9, max_features=0.5, min_samples_leaf=19, min_samples_split=5, n_estimators=100, subsample=0.3)),
    RobustScaler(),
    Normalizer(norm="l2"),
    ElasticNetCV(l1_ratio=0.9500000000000001, tol=0.1)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 42)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)

r2_value = r2_score(testing_target, results)
mean_abs_error_value = mean_absolute_error(testing_target, results)
mean_square_error_value = mean_squared_error(testing_target, results)

dict_values = {
    "r2_value" : r2_value,
    "mae" : mean_abs_error_value,
    "mse" : mean_square_error_value,
    "kendalltau": kendalltau(testing_target, results)[0],
    "pearsonr": pearsonr(testing_target, results)[0],
    "spearmanr": spearmanr(testing_target, results)[0]
}

with open("{}summary_performance_prottrans_t5_uniref.json".format(path_export), 'w') as doc_export:
    json.dump(dict_values, doc_export)

name_export = "{}prottrans_t5_uniref_model.joblib".format(path_export)
dump(exported_pipeline, name_export)