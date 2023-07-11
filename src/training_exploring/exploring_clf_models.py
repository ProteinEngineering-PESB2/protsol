import pandas as pd
import sys

#for metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from sklearn.model_selection import train_test_split

#for check overfitting
from sklearn.model_selection import cross_validate
import numpy as np

#for training models
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

#function to obtain metrics using the testing dataset
def get_performances(description, predict_label, real_label):
    accuracy_value = accuracy_score(real_label, predict_label)
    f1_score_value = f1_score(real_label, predict_label, average='weighted')
    precision_values = precision_score(real_label, predict_label, average='weighted')
    recall_values = recall_score(real_label, predict_label, average='weighted')

    row = [description, accuracy_value, f1_score_value, precision_values, recall_values]
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
scoring = ['f1_weighted', 'recall_weighted', 'precision_weighted', 'accuracy']
keys = ['fit_time', 'score_time', 'test_f1_weighted', 'test_recall_weighted', 'test_precision_weighted', 'test_accuracy']

k_fold_value = 10

X_train, X_test, y_train, y_test = train_test_split(data_values, response, test_size=0.3, random_state=42)

print("Exploring Training predictive models")
matrix_data = []

print("Exploring SVC")
clf = SVC()
response = training_process(clf, X_train, y_train, X_test, y_test, scoring, k_fold_value, "SVC", keys)
matrix_data.append(response)

print("Exploring KNN")
clf = KNeighborsClassifier(n_jobs=-1)
response = training_process(clf, X_train, y_train, X_test, y_test, scoring, k_fold_value, "KNN", keys)
matrix_data.append(response)
    
print("Exploring GausianNB")
clf = GaussianNB()
response = training_process(clf, X_train, y_train, X_test, y_test, scoring, k_fold_value, "GausianNB", keys)
matrix_data.append(response)

print("Exploring decision tree")
clf = DecisionTreeClassifier()
response = training_process(clf, X_train, y_train, X_test, y_test, scoring, k_fold_value, "DT", keys)
matrix_data.append(response)
        
print("Exploring bagging method based DT")
clf = BaggingClassifier(n_jobs=-1)
response = training_process(clf, X_train, y_train, X_test, y_test, scoring, k_fold_value, "bagging", keys)
matrix_data.append(response)
    
print("Exploring RF")
clf = RandomForestClassifier(n_jobs=-1)
response = training_process(clf, X_train, y_train, X_test, y_test, scoring, k_fold_value, "RF", keys)
matrix_data.append(response)
        

print("Exploring Adaboost")
clf = AdaBoostClassifier()
response = training_process(clf, X_train, y_train, X_test, y_test, scoring, k_fold_value, "Adaboost", keys)
matrix_data.append(response)
    

print("Exploring GradientTreeBoost")
clf = GradientBoostingClassifier()
response = training_process(clf, X_train, y_train, X_test, y_test, scoring, k_fold_value, "GradientBoostingClassifier", keys)
matrix_data.append(response)

print("Exploring NuSVC")
clf = NuSVC()
response = training_process(clf, X_train, y_train, X_test, y_test, scoring, k_fold_value, "NuSVC", keys)
matrix_data.append(response)

print("Exploring LinearSVC")
clf = LinearSVC()
response = training_process(clf, X_train, y_train, X_test, y_test, scoring, k_fold_value, "LinearSVC", keys)
matrix_data.append(response)

print("Exploring SGDClassifier")
clf = SGDClassifier(n_jobs=-1)
response = training_process(clf, X_train, y_train, X_test, y_test, scoring, k_fold_value, "SGDClassifier", keys)
matrix_data.append(response)

df_export = pd.DataFrame(matrix_data, columns=['description', 'test_accuracy', 'test_f1_score', 'test_precision', 'test_recall', 'fit_time', 'score_time', 'train_f1_weighted', 'train_recall_weighted', 'train_precision_weighted', 'train_accuracy'])
df_export.to_csv(name_export, index=False)
