# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#read data
dataset = pd.read_csv("USA_Housing.csv")
dataset.info()
dataset.describe()

#plot the data to see the price value
sns.distplot(dataset['Price'])
sns.heatmap(dataset.corr(),annot= True)

X = dataset.iloc[:,0:5].values
Y = dataset.iloc[:,5].values

#train test split
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test =train_test_split(X,Y,test_size = 0.2,random_state = 50)

#model selection
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train,Y_train)

print(lr.intercept_)
#coifficient 
# unit change in 1 parametre how much affect(keeping other fixe-d)
print(lr.coef_)

Y_pred = lr.predict(X_test)

plt.scatter(Y_test,Y_pred)
sns.distplot((Y_test-Y_pred),bins=50);


from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(Y_test, Y_pred))
print('MSE:', metrics.mean_squared_error(Y_test, Y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(Y_test, Y_pred)))




