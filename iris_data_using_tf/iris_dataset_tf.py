#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 10:44:13 2019

@author: omprakash
"""

import numpy as np
import pandas as pd

dataset = pd.read_csv("iris.csv")
dataset.columns
dataset.columns = ['s_length','s_width','p_length','p_widht','target']
dataset['target'] = dataset['target'].apply(int)

dataset.head()
dataset.describe()

X = dataset.drop('target',axis = 1)
Y = dataset['target']

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size =0.3)

import tensorflow as tf

feature_columns= []
for item in X.columns:
    feature_columns.append(tf.feature_column.numeric_column(item))

#create 2 input function 1 for training and other for testing
input_function = tf.estimator.inputs.pandas_input_fn(x=X_train,y=Y_train,
                                                     batch_size=10,
                                                     num_epochs=5,shuffle=True)

#hidden_units define how many neuron in each  layer
#3 classe as 3 species of flower in iris dataset
classifier = tf.estimator.DNNClassifier(hidden_units=[10, 20, 10],
                                        n_classes=3,
                                        feature_columns=feature_columns)
# train the data
classifier.train(input_fn = input_function,steps = 60)   

prediction_function = tf.estimator.inputs.pandas_input_fn(x=X_test,
                                                   batch_size=len(X_test),
                                                   shuffle=False)
#as predict method is a generatot so cast it to a list
prediction = list(classifier.predict(input_fn = prediction_function))
prediction[0]

#from list of data to take only prediction
pred_list = []
for pred_v in prediction:
    pred_list.append(pred_v['class_ids'][0])

pred_list

from sklearn.metrics import confusion_matrix,classification_report
print(confusion_matrix(Y_test,pred_list))

print(classification_report(Y_test,pred_list))






    
    
    
