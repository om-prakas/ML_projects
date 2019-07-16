# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 10:31:41 2019

@author: OmPrakash
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

dataset = pd.read_csv('loan_data.csv')
dataset.describe()
dataset.info()

plt.figure(figsize=(10,6))
dataset[dataset['credit.policy'] == 1]['fico'].hist(bins = 30,label='Credit.Policy=1')
dataset[dataset['credit.policy'] == 0]['fico'].hist(bins = 30,label='Credit.Policy=0')
plt.legend()
plt.xlabel('FICO')

final_data = pd.get_dummies(dataset,drop_first=True)

X = final_data.drop('not.fully.paid',axis=1)
Y = final_data['not.fully.paid']

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.3,random_state=30)

from sklearn.tree import DecisionTreeClassifier
DT = DecisionTreeClassifier()
DT.fit(X_train,Y_train)

Y_pred = DT.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(Y_test,Y_pred))
print('\n')
print (classification_report(Y_test,Y_pred))