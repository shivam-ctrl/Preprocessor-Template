# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing Dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
#print(X)
Y = dataset.iloc[:, -1].values
#print(Y)

# Missing data handling
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])
#print(X)

# Categorical Data Encoding
from sklearn.preprocessing import LabelEncoder, OneHotEncoder # for dummy encoding
from sklearn.compose import ColumnTransformer
# For a Perticular Column
# Simple Encoding
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
#print(X)
# Dummy Encoding
ct = ColumnTransformer([("Country", OneHotEncoder(), [0])], remainder = 'passthrough')
X = ct.fit_transform(X)
#print(X)
# column(Yes/No)
# In binary categorical data we can apply just simple encoding
from sklearn.preprocessing import LabelEncoder
labelencoder_Y = LabelEncoder()
Y[:] = labelencoder_Y.fit_transform(Y[:])
#print(Y)
# NOTE : Always handle the dummy variable trap in case of dummy encoding.

# Splitting dataset into train set and test set
from sklearn.model_selection import train_test_split # module for efficient splitting(we don't have to do it manually)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
#print(X_train, X_test, Y_train, Y_test)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
#print(X_train, X_test)
