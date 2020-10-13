#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 20:38:57 2020

@author: gerard
"""

# Import dependencies

# Start Python Imports
import math, time, random, datetime

# Data Manipulation
import numpy as np
import pandas as pd
import pickle

# Visualization 
import matplotlib.pyplot as plt
#import missingno
import seaborn as sns
plt.style.use('seaborn-whitegrid')

# Preprocessing
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, label_binarize

# Machine learning
#import catboost
#from catboost import CatBoostClassifier, Pool, cv
from sklearn.model_selection import train_test_split
from sklearn import model_selection, tree, preprocessing, metrics, linear_model
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier


# Let's be rebels and ignore warnings for now
import warnings
warnings.filterwarnings('ignore')

#%% Load the data

df = pd.read_csv("../data/Training.csv")
Y_train = df.prognosis
Y_train = pd.Series(LabelEncoder().fit_transform(Y_train))
X_train = df.drop("prognosis", axis=1)

print(f"Train data: {len(X_train)}")

df = pd.read_csv("../data/Testing.csv")
Y_test = df.prognosis
le = LabelEncoder()
filename = "label_encoder.sav"
pickle.dump(le, open(filename, 'wb'))
Y_test = pd.Series(le.fit_transform(Y_test))
X_test = df.drop("prognosis", axis=1)

print(f"Test data: {len(X_test)}")

#%% Train models

# Function that runs the requested algorithm and returns the accuracy metrics
def fit_ml_algo(algo, X_train, y_train, cv):
    
    # One Pass
    model = algo.fit(X_train, y_train)
    acc = round(model.score(X_train, y_train) * 100, 2)
    
    # Cross Validation 
    train_pred = model_selection.cross_val_predict(algo, 
                                                  X_train, 
                                                  y_train, 
                                                  cv=cv, 
                                                  n_jobs = -1)
    # Cross-validation accuracy metric
    acc_cv = round(metrics.accuracy_score(y_train, train_pred) * 100, 2)
    
    return train_pred, acc, acc_cv, model

#%% Logistic Regression
start_time = time.time()
train_pred_log, acc_log, acc_cv_log, model = fit_ml_algo(LogisticRegression(), 
                                                               X_train, 
                                                               Y_train, 
                                                                    10)
log_time = (time.time() - start_time)
print("Accuracy: %s" % acc_log)
print("Accuracy CV 10-Fold: %s" % acc_cv_log)
print("Running Time: %s" % datetime.timedelta(seconds=log_time))
preds = model.predict(X_test)
test_pred_log = round(metrics.accuracy_score(Y_test, preds) * 100, 2)
print("Test accuracy: %s" % test_pred_log)

#filename = 'log_model.sav'
#pickle.dump(model, open(filename, 'wb'))

#%% k-Nearest Neighbours
start_time = time.time()
train_pred_knn, acc_knn, acc_cv_knn, model = fit_ml_algo(KNeighborsClassifier(), 
                                                  X_train, 
                                                  Y_train, 
                                                  10)
knn_time = (time.time() - start_time)
print("Accuracy: %s" % acc_log)
print("Accuracy CV 10-Fold: %s" % acc_cv_log)
print("Running Time: %s" % datetime.timedelta(seconds=log_time))
preds = model.predict(X_test) 
test_pred_knn = round(metrics.accuracy_score(Y_test, preds) * 100, 2)
print("Test accuracy: %s" % test_pred_knn)

#%% Gaussian Naive Bayes
start_time = time.time()
train_pred_gaussian, acc_gaussian, acc_cv_gaussian, model = fit_ml_algo(GaussianNB(), 
                                                                      X_train, 
                                                                      Y_train, 
                                                                           10)
gaussian_time = (time.time() - start_time)
print("Accuracy: %s" % acc_log)
print("Accuracy CV 10-Fold: %s" % acc_cv_log)
print("Running Time: %s" % datetime.timedelta(seconds=log_time))
preds = model.predict(X_test) 
test_pred_gaussian = round(metrics.accuracy_score(Y_test, preds) * 100, 2)
print("Test accuracy: %s" % test_pred_gaussian)

#%% Linear SVC
start_time = time.time()
train_pred_svc, acc_linear_svc, acc_cv_linear_svc, model = fit_ml_algo(LinearSVC(),
                                                                X_train, 
                                                                Y_train, 
                                                                10)
linear_svc_time = (time.time() - start_time)
print("Accuracy: %s" % acc_log)
print("Accuracy CV 10-Fold: %s" % acc_cv_log)
print("Running Time: %s" % datetime.timedelta(seconds=log_time))
preds = model.predict(X_test) 
test_pred_svc = round(metrics.accuracy_score(Y_test, preds) * 100, 2)
print("Test accuracy: %s" % test_pred_svc)

#%% Stochastic Gradient Descent
start_time = time.time()
train_pred_sgd, acc_sgd, acc_cv_sgd, model = fit_ml_algo(SGDClassifier(), 
                                                  X_train, 
                                                  Y_train,
                                                  10)
sgd_time = (time.time() - start_time)
print("Accuracy: %s" % acc_log)
print("Accuracy CV 10-Fold: %s" % acc_cv_log)
print("Running Time: %s" % datetime.timedelta(seconds=log_time))
preds = model.predict(X_test) 
test_pred_sgd = round(metrics.accuracy_score(Y_test, preds) * 100, 2)
print("Test accuracy: %s" % test_pred_sgd)

#%% Decision Tree Classifier
start_time = time.time()
train_pred_dt, acc_dt, acc_cv_dt, model = fit_ml_algo(DecisionTreeClassifier(), 
                                                                X_train, 
                                                                Y_train,
                                                                10)
dt_time = (time.time() - start_time)
print("Accuracy: %s" % acc_log)
print("Accuracy CV 10-Fold: %s" % acc_cv_log)
print("Running Time: %s" % datetime.timedelta(seconds=log_time))
preds = model.predict(X_test) 
test_pred_dt = round(metrics.accuracy_score(Y_test, preds) * 100, 2)
print("Test accuracy: %s" % test_pred_dt)

#%% Gradient Boosting Trees
start_time = time.time()
train_pred_gbt, acc_gbt, acc_cv_gbt, model = fit_ml_algo(GradientBoostingClassifier(), 
                                                                       X_train, 
                                                                       Y_train,
                                                                       10)
gbt_time = (time.time() - start_time)
print("Accuracy: %s" % acc_log)
print("Accuracy CV 10-Fold: %s" % acc_cv_log)
print("Running Time: %s" % datetime.timedelta(seconds=log_time))
preds = model.predict(X_test) 
test_pred_gbt = round(metrics.accuracy_score(Y_test, preds) * 100, 2)
print("Test accuracy: %s" % test_pred_gbt)

# =============================================================================
# Others:
#   · Catboost   
# =============================================================================

#%% Review results

models = pd.DataFrame({
    'Model': ['KNN', 'Logistic Regression', 'Naive Bayes', 
              'Stochastic Gradient Decent', 'Linear SVC', 
              'Decision Tree', 'Gradient Boosting Trees'],
    'Score': [
        acc_knn, 
        acc_log,  
        acc_gaussian, 
        acc_sgd, 
        acc_linear_svc, 
        acc_dt,
        acc_gbt
    ]})
print("---Reuglar Accuracy Scores---")
models.sort_values(by='Score', ascending=False)

cv_models = pd.DataFrame({
    'Model': ['KNN', 'Logistic Regression', 'Naive Bayes', 
              'Stochastic Gradient Decent', 'Linear SVC', 
              'Decision Tree', 'Gradient Boosting Trees'],
    'Score': [
        acc_cv_knn, 
        acc_cv_log,      
        acc_cv_gaussian, 
        acc_cv_sgd, 
        acc_cv_linear_svc, 
        acc_cv_dt,
        acc_cv_gbt
    ]})
print('---Cross-validation Accuracy Scores---')
cv_models.sort_values(by='Score', ascending=False)

cv_models = pd.DataFrame({
    'Model': ['KNN', 'Logistic Regression', 'Naive Bayes', 
              'Stochastic Gradient Decent', 'Linear SVC', 
              'Decision Tree', 'Gradient Boosting Trees'],
    'Score': [
        test_pred_knn, 
        test_pred_log,      
        test_pred_gaussian, 
        test_pred_sgd, 
        test_pred_svc, 
        test_pred_dt,
        test_pred_gbt
    ]})
print('---Test Accuracy Scores---')
cv_models.sort_values(by='Score', ascending=False)

#%% Showing predictions

data = X_train
names_freq = data.sum().sort_values(ascending=False).index
next_data = data
vector = pd.Series([0] * len(data.columns), index=data.columns)
asked = []
while len(names_freq) > 0:
    synthom_bool = int(input(f"{names_freq[0]}?"))
    vector.loc[names_freq[0]] = synthom_bool
    asked.append(names_freq[0])
    next_data = next_data[next_data[names_freq[0]] == synthom_bool] #index with answer
    next_data = next_data.loc[:, (next_data==1).any(axis=0)] #remove not possible cases (?)
    names_freq = [x for x in next_data.sum().sort_values(ascending=False).index if x not in asked]
    
preds = model.predict([vector])
preds = le.inverse_transform(preds)

print(f"Oh no! Your synthoms indicate {preds[0]}. You should consider visiting your local doctor")


