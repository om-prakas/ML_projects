# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Banking_data.csv')
X = dataset.iloc[:,3:13].values
Y = dataset.iloc[:,13].values

#encoding categorical data
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
le_x1 = LabelEncoder()
X[:,1]=le_x1.fit_transform(X[:,1])
le_x2 = LabelEncoder()
X[:,2] = le_x2.fit_transform(X[:,2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X= onehotencoder.fit_transform(X).toarray()
#dummy variable trap to reduce remove 1 variable
X= X[:,1:]  #France(0,0),spain(0,1),germany(1,0)

#train_test split
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2,random_state = 50)

#feature scaling 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train =  sc.fit_transform(X_train)
X_test =  sc.transform(X_test)

#keras built on theano and tensorflow library
import keras
#model used to initialise the NN
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout   # to decrease overfitting
se = Sequential()

#add input layer and output layer  11+1 /2 = 6 hidden layer 
#activation function rectifier function for hidden and sigmoid function output layer
# uniform function initialise the weight randomly
se.add(Dense(output_dim =6,activation = 'relu',kernel_initializer = 'uniform', 
             input_dim = 11))
se.add(Dropout(rate = 0.1))
# add 2nd hidden layer 
se.add(Dense(6,activation = 'relu',kernel_initializer = 'uniform'))

#adding output layer 
se.add(Dense(output_dim = 1,activation = 'sigmoid',kernel_initializer = 'uniform'))

#compiling  ANN(stocastic gradient discent )/categorical_crossentropy(>2 output variable)
se.compile(optimizer='adam',loss='binary_crossentropy',metrics = ['accuracy'])

# fit the data
se.fit(X_train,Y_train,batch_size = 10, epochs= 100)

#Predict value
Y_pred = se.predict(X_test)
# less than 0.5 make value 0
Y_pred = (Y_pred > 0.5)

#confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,Y_pred)
print(cm)

#-----------------------------------------------------
# Predicting a single new observation
"""Predict if the customer with the following informations will leave the bank:
Geography: Spain
Credit Score: 700
Gender: Female
Age: 45
Tenure: 4
Balance: 70000
Number of Products: 3
Has Credit Card: Yes
Is Active Member: Yes
Estimated Salary: 55000"""
#insert into ARRAY in same order HORIZONTALLY,so add 2d array [[]] 
#Apply the same scaller(sc) for stanardised the data
pred_customer= se.predict(sc.transform(np.array([[0.0, 1, 700, 0, 45, 4, 70000, 3, 1, 1, 55000]])))
pred_customer = (pred_customer > 0.5)

#---------------------------------------------------------------------------


# Evaluating the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense
#from keras.layers import Dropout
def build_classifier():
    se = Sequential()
    se.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    #se.add(Dropout(rate = 0.1)) 
    se.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    se.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    se.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return se

#se is local variable make 1st global varible to use outside it
se = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)
#cross-validation -cv -1 is all cpu we are using parallel processing
accuracies = cross_val_score(estimator=se,X = X_train,y=Y_train,cv=10,n_jobs=-1)
mean = accuracies.mean()
variance = accuracies.std()   #high variance = overfitting

#--------------------------------------------------------------------------

#parameter tuning find the best value of hyper parameter(best choice make)
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
#we pass parameter in function to choose the optimiser parameter 
def build_classifier(optimizer):
    se = Sequential()
    se.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    se.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    se.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    #instead od adam we will choose between adam and rmsprop. 
    se.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return se

se = KerasClassifier(build_fn = build_classifier)
#define the hyper parameter which you want to tune for best fit model
parameters = {'batch_size': [25, 32],
              'epochs': [100, 500],
              'optimizer': ['adam', 'rmsprop']}

grid_search = GridSearchCV(estimator = se,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
#fit grid search to training set
grid_search_new = grid_search.fit(X_train, Y_train)
best_parameters = grid_search_new.best_params_
best_accuracy = grid_search_new.best_score_










