#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 14:26:35 2020

@author: golnaznikmehr
"""
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.feature_selection import SelectFromModel
from sklearn import linear_model

### Load Data

data = pd.read_csv('data.csv',encoding='latin1')


### Function for converting Categorical Features to Numerical

def convertCategoricalToNumerical(inputData, NameOfColumn, column):
    uniq_itemOfColumn = set(list(inputData[NameOfColumn]))
    label_index = dict((c, i) for i, c in enumerate(uniq_itemOfColumn))
    
    Y = []
    for i in column:
        Y.append(label_index[i])
        
    return Y

### Choose some Features and convert them to numerical Features
    
columns = ['Episode', 'Station', 'Channel Type', 'Season', 'Year','Day of week',
           'Name of show','Genre','First time or rerun','# of episode in the season', 
           'Movie?','Game of the Canadiens during episode?']

convertedX = pd.DataFrame(columns=columns)

for c in columns:
    convertedX[c] = convertCategoricalToNumerical(data, c, data[c])

convertedX['Length'] = data['Length']
convertedX['MarketShare_total'] = data['MarketShare_total']

# Delete NAN records
convertedX = convertedX.dropna()

### Define our Dense Network

def baseline_model(train):
    # create model
    model = Sequential()
    model.add(Dense(512, input_dim=train.shape[1], activation='elu',kernel_initializer='he_uniform'))
    model.add(Dense(64, activation="elu"))
    model.add(Dense(1, activation='linear'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


### Make Train and Test from the Data

msk = np.random.rand(len(convertedX)) < 0.8
train = convertedX[msk]
test = convertedX[~msk]

train_y = train.MarketShare_total
train = train.drop('MarketShare_total', axis=1)

test_y = test.MarketShare_total
test = test.drop('MarketShare_total', axis=1)

### Feature Selection and Transform the Data

clf = linear_model.Lasso(alpha=0.1).fit(train, train_y)
model = SelectFromModel(clf, prefit=True)
X_new = model.transform(train)
print(X_new.shape)

test_new = model.transform(test)

### Make the Dense Network Model and Evaluation

model = baseline_model(train)
# fit model
history = model.fit(train, train_y, batch_size=100, validation_data=(test, test_y), epochs=100, verbose=1)

# evaluate the model
train_mse = model.evaluate(train, train_y, verbose=0)
test_mse = model.evaluate(test, test_y, verbose=0)
print('Train: %.3f' % (train_mse))
print('Test: %.3f' % (test_mse))

### Load Test Data and Predict

TaskTest = pd.read_csv('test.csv',encoding='latin1')

convertedTest = pd.DataFrame(columns=columns)

for cc in columns:
    convertedTest[cc] = convertCategoricalToNumerical(TaskTest, cc, TaskTest[cc])

convertedTest['Length'] = TaskTest['Length']
convertedTest = convertedTest.dropna()

test_Prediction = model.predict(convertedTest, verbose=0)

