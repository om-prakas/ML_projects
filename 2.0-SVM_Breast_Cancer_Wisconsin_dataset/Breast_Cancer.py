# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 12:18:09 2019

@author: OmPrakash
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.datasets import load_breast_cancer
cancer_data= load_breast_cancer()
cancer_data.keys()

cancer_data['data']
cancer_data['target']
cancer_data['target_names']
cancer_data['DESCR']
cancer_data['feature_names']


X_dataset =  pd.DataFrame(cancer_data['data'],columns=cancer_data['feature_names'])
Y_dataset = pd.DataFrame(cancer_data['target'],columns=['Cancer'])

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X_dataset,Y_dataset,test_size=0.2)

from sklearn.svm import SVC
svc = SVC()
svc.fit(X_train,Y_train)

Y_pred = svc.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(Y_test,Y_pred))
print('\n')
print(classification_report(Y_test,Y_pred))

#search for the best parameter using c and gamma value 
from sklearn.grid_search import GridSearchCV
parameter = [{'C': [0.1,1, 10, 100, 1000], 
             'gamma': [1,0.1,0.01,0.001,0.0001],
             'kernel': ['rbf']}]

grid_search = GridSearchCV(estimator = SVC(),param_grid=parameter,refit=True,verbose=3)
grid_search.fit(X_train,Y_train)


grid_search.best_params_
grid_predictions = grid_search.predict(X_test)

print(confusion_matrix(Y_test,grid_predictions))
print(classification_report(Y_test,grid_predictions))

