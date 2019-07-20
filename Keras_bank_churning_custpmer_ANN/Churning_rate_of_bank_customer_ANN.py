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
X= X[:,1:]

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
se = Sequential()

#add input layer and output layer  11+1 /2 = 6 hidden layer 
#activation function rectifier function for hidden and sigmoid function output layer
# uniform function initialise the weight randomly
se.add(Dense(output_dim =6,activation = 'relu',kernel_initializer = 'uniform', 
             input_dim = 11))

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
















