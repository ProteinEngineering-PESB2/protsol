import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from tpot.export_utils import set_param_recursive
import sys

#for metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import matthews_corrcoef

#for exports results and jobs
import json
from joblib import dump

name_csv_data = sys.argv[1]
path_export = sys.argv[2]

tpot_data = pd.read_csv(sys.argv[1], dtype=np.float64)
features = tpot_data.drop('target', axis=1)

matrix_data = []

for i in range(1000):
    training_features, testing_features, training_target, testing_target = \
                train_test_split(features, tpot_data['target'], random_state=i)

    # Average CV score on the training set was: 0.7283344051446946
    exported_pipeline = make_pipeline(
        Normalizer(norm="l2"),
        SGDClassifier(alpha=0.0, eta0=0.01, fit_intercept=False, l1_ratio=0.25, learning_rate="constant", loss="squared_hinge", penalty="elasticnet", power_t=10.0)
    )
    # Fix random state for all the steps in exported pipeline
    set_param_recursive(exported_pipeline.steps, 'random_state', i)

    exported_pipeline.fit(training_features, training_target)
    results = exported_pipeline.predict(testing_features)

    accuracy_value = accuracy_score(testing_target, results)
    f1_score_value = f1_score(testing_target, results, average='weighted')
    precision_values = precision_score(testing_target, results, average='weighted')
    recall_values = recall_score(testing_target, results, average='weighted')
    mcc_values = matthews_corrcoef(testing_target, results)

    row = [i, accuracy_value, f1_score_value, precision_values, recall_values, mcc_values]
    matrix_data.append(row)

df_export = pd.DataFrame(matrix_data, columns=['index', 'accuracy', 'f1-score', 'precision', 'recall', 'mcc'])
df_export.to_csv("{}distribution_performances.csv".format(path_export), index=False)

'''
dict_performances = {
    "accuracy" : accuracy_value,
    "f_score" : f1_score_value,
    "precision" : precision_values,
    "recall" : recall_values
}

with open("{}summary_performances.json".format(path_export), 'w') as doc_export:
    json.dump(dict_performances, doc_export)

name_export_job = "{}optim_model.joblib".format(path_export)
dump(exported_pipeline, name_export_job)
'''