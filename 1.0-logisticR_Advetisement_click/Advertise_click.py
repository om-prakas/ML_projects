# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#Instaling Theano
#(work in GPU)
#install Tensorflow
#install Keras

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv("advertising.csv")
dataset.head()
dataset.info()
dataset.describe()

dataset['Age'].hist(bins = 30)
plt.xlabel('Age')

X = dataset[['Daily Time Spent on Site', 'Age', 'Area Income','Daily Internet Usage', 'Male']]
y = dataset['Clicked on Ad']
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,y,test_size = 0.3,random_state= 30)

from sklearn.linear_model import LogisticRegression
LR = LogisticRegression()
LR.fit(X_train,Y_train)

Y_pred = LR.predict(X_test)

from sklearn.metrics import classification_report
print(classification_report(Y_test,Y_pred))


