#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 08:55:47 2020

@author: gerard

Datasets: https://medium.com/@ODSC/15-open-datasets-for-healthcare-830b19980d9
Healthcare datasets:
    https://data.medicare.gov/
    https://hcup-us.ahrq.gov/databases.jsp
Using now: https://github.com/vsharathchandra/AI-Healthcare-chatbot
"""

# =============================================================================
# We have to classify into one of the 41 possible diagnostics
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

df = pd.read_csv("./Training.csv")
label = df.prognosis
data = df.drop("prognosis", axis=1)

#%% Data Analysis
print(f"Sympthoms:\n {list(df.columns)[:-1]} - Total: {len(df.columns)-1}\n")
print(f"Prognosis:\n {df.prognosis.unique()} - N. categories: {len(df.prognosis.unique())} - Total: {len(df.prognosis)-1}\n")

plt.figure()
plt.title("Synthoms frequency")
plt.plot(data.sum())

vals = []
for index, row in data.iterrows():
    vals.append(np.sum(row.values))
plt.figure()
plt.title("Number of synthoms per diagnosticw")
plt.xlim(0, len(vals))
plt.ylim(0, max(vals))
plt.plot(vals)

print(f"Data NaN values: {data.isna().sum().sum()} - Label NaN values: {label.isna().sum()}")

#%% Example 1
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
    #print("Synthoms left:", len(names_freq))

print("Your synthoms are:", ', '.join(list(vector[vector == 1].dropna().index)))
print("Final diagnostic:")
for idx, row in data.iterrows():
    b = np.array_equal(row.values, vector.values)
    if b == True: 
        print(label.iloc[idx])
        break
    
#%% EXTRA: saving synthoms names as pickle

with open('sytnhoms.pkl', 'rb') as f:
    mynewlist = pickle.load(f)
with open('synthoms.pkl', 'wb') as f:
    pickle.dump(list(data.columns), f)
with open('synthoms.pkl', 'rb') as f:
    mynewlist = pickle.load(f)