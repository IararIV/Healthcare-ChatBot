#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 21:42:02 2020

@author: gerard
"""

#%% Import dependencies

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

#%% Load models and data

print("Loading data...")
pkl_file = open("../models/log_model.pkl", 'rb')
model = pickle.load(pkl_file)
pkl_file.close()
pkl_file = open("../models/label_encoder.pkl", 'rb')
le = pickle.load(pkl_file)
pkl_file.close()
data = df = pd.read_csv("../data/Training.csv").drop("prognosis", axis=1)
print("Done!")

#%% Ask synthoms

print("I'm going to ask you about some synthoms, You must answer with yes/no.")
names_freq = data.sum().sort_values(ascending=False).index
next_data = data
vector = pd.Series([0] * len(data.columns), index=data.columns)
asked = []
while len(names_freq) > 0:
    synthom_bool = int(input(f"Is {names_freq[0]} one of you synthoms?"))
    vector.loc[names_freq[0]] = synthom_bool
    asked.append(names_freq[0])
    next_data = next_data[next_data[names_freq[0]] == synthom_bool] #index with answer
    next_data = next_data.loc[:, (next_data==1).any(axis=0)] #remove not possible cases (?)
    names_freq = [x for x in next_data.sum().sort_values(ascending=False).index if x not in asked]
    
#%% Predict
    
preds = model.predict([vector])
preds = le.inverse_transform(preds)

print(f"Oh no! Your synthoms indicate {preds[0]}. You should consider visiting your local doctor")
