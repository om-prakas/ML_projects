# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#Instaling Theano
#(work in GPU)
#install Tensorflow
#install Keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


dataset = pd.read_csv("Ecommerce Customers")
dataset.describe()
dataset.info()

X = dataset[['Avg. Session Length', 'Time on App','Time on Website', 'Length of Membership']]
#X = dataset.iloc[:,3:7].values
Y = dataset['Yearly Amount Spent']
#Y= dataset.iloc[:,7].values

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3,random_state= 30)

from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X_train,Y_train)

print(LR.coef_)

Y_pred = LR.predict(X_test)

plt.scatter(Y_test,Y_pred)
plt.xlabel('Y test')
plt.ylabel('Y pred')


#evaluating the model
from sklearn import metrics
print('MSE:',metrics.mean_squared_error(Y_test,Y_pred))
print('MAE:',metrics.mean_absolute_error(Y_test,Y_pred))
print('RMSE:',np.sqrt(metrics.mean_squared_error(Y_test,Y_pred)))

sns.distplot(Y_test-Y_pred)

#calculate coefficient to check 1 unit change how much affected
cofficient = pd.DataFrame(LR.coef_,X.columns)
cofficient.columns = ['Coeffecient']
cofficient


