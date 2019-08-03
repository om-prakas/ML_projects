# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import keras as ke

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

dataset_train = pd.read_csv('xyz_Stock_Price_Train.csv')
#only numpy array can be input of NN in Keras. 
training_set = dataset_train.iloc[:,1:2].values

#Normilasation of data(feature scaling)
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))  
sc_train = sc.fit_transform(training_set)

#read privious 60 days timestamp and predict next 1 value
X_train = []   
Y_train = []
# start from 60 and go upto range of your dataframe i.e. 1258
for item in range(60,1258):                 
    X_train.append(sc_train[item-60:item,0])   #from 0 to 59 row, 1 column
    Y_train.append(sc_train[item,0])           # 60 value count

#from list to convert numpy array
X_train = np.array(X_train)
Y_train = np.array(Y_train)

#reshape (batch size )
X_train.shape[0]  #1198 - row
X_train.shape[1]  #60 -column
#no of indicator =1  stock price
X_train_3d = np.reshape(X_train,(1198,60,1))  #shape (batch_size, timesteps,input_dim)

#After preprocess now create model 

seq = Sequential()
#to make a stacked LSTM, return_sequence should be true,input shape- last 2 parameter
seq.add(LSTM(units= 50,return_sequences= True,input_shape = (X_train.shape[1],1)))
seq.add(Dropout(0.2))

#no need to give input_shape  as we are define units in privious layer
seq.add(LSTM(units= 50, return_sequences= True))
seq.add(Dropout(0.2))

seq.add(LSTM(units= 50,return_sequences= True))
seq.add(Dropout(0.2))

seq.add(LSTM(units = 50, return_sequences= False))
seq.add(Dropout(0.2))

#output layer is fully connected class,units means no of neurons in layer
seq.add(Dense(units= 1))

#compile  with optimiser and loss function -MSE (not binary cross entropy) as it's regressor
seq.compile(optimizer= 'adam', loss = 'mean_squared_error')

#fit the datset
seq.fit(X_train_3d,Y_train,epochs= 100, batch_size=32)


dataset_test = pd.read_csv('xyz_Stock_Price_Test.csv')
testing_set = dataset_test.iloc[:,1:2].values

#test data 

#we need privious 60 days data,so concact both original data  
#don't concact train scaled with unscaled test,combine both unscaled data
dataset_all = pd.concat((dataset_train['Open'], dataset_test['Open']),axis = 0)
#we require train last 60 element to test last element (lowerbound:uper bound)
input_value = dataset_all[len(dataset_train)- 60:].values
#inputs = dataset_all[len(dataset_all) - len(dataset_test) - 60:].values
input_value = input_value.reshape(-1,1)
#scale your data
#sc obj already fitted to training set,no need to fit.use privious scaling
input_value = sc.transform(input_value)

X_test = []
for i in range(60,80):       #60+20(test data) = 80
    X_test.append(input_value[i-60:i,0])
X_test = np.array(X_test)

X_test.shape[0]  #20
X_test.shape[1]  #60
X_test_3d = np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))

predicting_set = seq.predict(X_test_3d)
#value we got as scaled. We need original value. Do inverse transform
predicting_set = sc.inverse_transform(predicting_set)

#visualize the data
plt.plot(testing_set,color = 'Red',label= 'Real stock price graph')
plt.plot(predicting_set,color = 'Blue', label = 'Prediction graph')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

#for regression parameter tuning is same as classifier only scoring change
#for grid search use scoring = 'neg_mean_squared_error'  instead of scoring = 'accuracy'

import math
from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(testing_set, predicting_set))


















