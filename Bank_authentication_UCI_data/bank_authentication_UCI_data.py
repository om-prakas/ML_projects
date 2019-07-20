#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 13:15:57 2019

@author: omprakash
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

dataset = pd.read_csv("bank_note_data.csv")
dataset.head()
dataset.describe()

sns.countplot(x ='Class',data = dataset)

x_temp = dataset.drop('Class',axis = 1)
#Y_temp = dataset['Class']
#preprocess the data and stanardised it 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_scaler = sc.fit_transform(x_temp)

#X_scaler is a numpy.ndarray type.Columns is present in Dataframe. covert 1st
X=pd.DataFrame(X_scaler)
X.columns = ['Image.Var', 'Image.Skew', 'Image.Curt', 'Entropy']
Y = dataset['Class']

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size =0.2)

image_var = tf.feature_column.numeric_column("Image.Var")
image_skew = tf.feature_column.numeric_column('Image.Skew')
image_curt = tf.feature_column.numeric_column('Image.Curt')
entropy =tf.feature_column.numeric_column('Entropy')
feature_columns = [image_var,image_skew,image_curt,entropy]

'''
feature_columns = []
for col in X.columns:
    feature_columns.append(tf.feature_column.numeric_column(col))
'''

classifier = tf.estimator.DNNClassifier(hidden_units=[10, 20, 10],
                                        n_classes=2,
                                        feature_columns=feature_columns)

input_func = tf.estimator.inputs.pandas_input_fn(x=X_train,y=Y_train,
                                                     batch_size=15,
                                                     num_epochs=5,shuffle=True)

classifier.train(input_fn = input_func,steps = 450)   

prediction_func = tf.estimator.inputs.pandas_input_fn(x=X_test,
                                                         batch_size=len(X_test),
                                                         shuffle=False)

prediction = list(classifier.predict(input_fn = prediction_func))
prediction[0]

pred_list = []
for pred_v in prediction:
    pred_list.append(pred_v['class_ids'][0])

from sklearn.metrics import confusion_matrix,classification_report
print(confusion_matrix(Y_test,pred_list))

print(classification_report(Y_test,pred_list))








