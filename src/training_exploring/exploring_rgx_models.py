import pandas as pd
import sys

#for metrics
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split

#for check overfitting
from sklearn.model_selection import cross_validate
import numpy as np

#for training models
from sklearn.svm import SVR, NuSVR, LinearSVR
from sklearn.linear_model import SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor

#function to obtain metrics using the testing dataset
def get_performances(description, predict_label, real_label):
    r2_value = r2_score(real_label, predict_label)
    mean_abs_error_value = mean_absolute_error(real_label, predict_label)
    mean_square_error_value = mean_squared_error(real_label, predict_label)
    
    row = [description, r2_value, mean_abs_error_value, mean_square_error_value]
    return row

#function to process average performance in cross val training process
def process_performance_cross_val(performances, keys):
    
    row_response = []
    for i in range(len(keys)):
        value = np.mean(performances[keys[i]])
        row_response.append(value)
    return row_response

#function to train a predictive model
def training_process(model, X_train, y_train, X_test, y_test, scores, cv_value, description, keys):
    print("Train model with cross validation")
    model.fit(X_train, y_train)
    response_cv = cross_validate(model, X_train, y_train, cv=cv_value, scoring=scores)
    performances_cv = process_performance_cross_val(response_cv, keys)

    print("Predict responses and make evaluation")
    responses_prediction = model.predict(X_test)
    response = get_performances(description, responses_prediction, y_test)
    response = response + performances_cv
    return response

df_data = pd.read_csv(sys.argv[1])
name_export = sys.argv[2]

response = df_data['target']
data_values = df_data.drop(columns=['target'])

#define the type of metrics
scoring = ['max_error', 'neg_mean_absolute_error', 'neg_mean_squared_error', 'neg_median_absolute_error', 'neg_root_mean_squared_error', 'r2']
keys = ['fit_time', 'score_time', 'test_max_error', 'test_neg_mean_absolute_error', 'test_neg_mean_squared_error', 'test_neg_median_absolute_error', 'test_neg_root_mean_squared_error', 'test_r2']

k_fold_value = 10

X_train, X_test, y_train, y_test = train_test_split(data_values, response, test_size=0.3, random_state=42)

print("Exploring Training predictive models")
matrix_data = []

print("Exploring SVR")
clf = SVR()
response = training_process(clf, X_train, y_train, X_test, y_test, scoring, k_fold_value, "SVR", keys)
matrix_data.append(response)

print("Exploring KNN")
clf = KNeighborsRegressor(n_jobs=-1)
response = training_process(clf, X_train, y_train, X_test, y_test, scoring, k_fold_value, "KNeighborsRegressor", keys)
matrix_data.append(response)
    
print("Exploring decision tree")
clf = DecisionTreeRegressor()
response = training_process(clf, X_train, y_train, X_test, y_test, scoring, k_fold_value, "DecisionTreeRegressor", keys)
matrix_data.append(response)
        
print("Exploring bagging method based DT")
clf = BaggingRegressor(n_jobs=-1)
response = training_process(clf, X_train, y_train, X_test, y_test, scoring, k_fold_value, "BaggingRegressor", keys)
matrix_data.append(response)
    
print("Exploring RF")
clf = RandomForestRegressor(n_jobs=-1)
response = training_process(clf, X_train, y_train, X_test, y_test, scoring, k_fold_value, "RandomForestRegressor", keys)
matrix_data.append(response)
        
print("Exploring Adaboost")
clf = AdaBoostRegressor()
response = training_process(clf, X_train, y_train, X_test, y_test, scoring, k_fold_value, "AdaBoostRegressor", keys)
matrix_data.append(response)
    
print("Exploring GradientTreeBoost")
clf = GradientBoostingRegressor()
response = training_process(clf, X_train, y_train, X_test, y_test, scoring, k_fold_value, "GradientBoostingRegressor", keys)
matrix_data.append(response)

print("Exploring NuSVR")
clf = NuSVR()
response = training_process(clf, X_train, y_train, X_test, y_test, scoring, k_fold_value, "NuSVR", keys)
matrix_data.append(response)

print("Exploring LinearSVR")
clf = LinearSVR()
response = training_process(clf, X_train, y_train, X_test, y_test, scoring, k_fold_value, "LinearSVR", keys)
matrix_data.append(response)

print("Exploring SGDRegressor")
clf = SGDRegressor()
response = training_process(clf, X_train, y_train, X_test, y_test, scoring, k_fold_value, "SGDRegressor", keys)
matrix_data.append(response)

df_export = pd.DataFrame(matrix_data, columns=['description', 'r2_value', 'mean_abs_error_value', 'mean_square_error_value', 'fit_time', 'score_time', 'test_max_error', 'test_neg_mean_absolute_error', 'test_neg_mean_squared_error', 'test_neg_median_absolute_error', 'test_neg_root_mean_squared_error', 'test_r2'])
df_export.to_csv(name_export, index=False)
