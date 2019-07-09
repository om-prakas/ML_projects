# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 09:21:45 2019

@author: OmPrakash
"""



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv('titanic_train.csv')
dataset.describe()
dataset.info()

#create a heatmap or from info find the field missing
sns.heatmap(dataset.isnull(),yticklabels=False,cbar=False,cmap='viridis')
#classification problem count who survive(0-not survived,1-survived)
#change hue parameter to check on different parametr or x parametr
sns.countplot(x='SibSp',data=dataset)
sns.countplot(x='Survived',data=dataset,hue='Sex',palette='RdBu_r')
dataset['Age'].hist(bins=30,color='darkred',alpha=0.7)

# data cleaning
#find a relation between passenger class and age to make adjust value for Age
plt.figure(figsize=(12, 7))
sns.boxplot(x='Pclass',y='Age',data=dataset,palette='winter')


def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):

        if Pclass == 1:
            return 37

        elif Pclass == 2:
            return 29

        else:
            return 24

    else:
        return Age

dataset['Age'] = dataset[['Age','Pclass']].apply(impute_age,axis=1)
#drop cabin column as so many missing value
dataset.drop('Cabin',axis=1,inplace=True)
dataset.dropna(inplace=True) # any row missing value drop


#handle categorical value ,dummy variable trp to avoid drop 1 column
#pd.get_dummies(dataset['Sex'])
sex = pd.get_dummies(dataset['Sex'],drop_first=True)
embark = pd.get_dummies(dataset['Embarked'],drop_first=True)  
    
#delete the row which not required
dataset.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
#add the converted categorical varible 
dataset = pd.concat([dataset,sex,embark],axis=1)

X=dataset.drop('Survived',axis=1)
Y=dataset['Survived']
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, 
                                                    random_state=101)
from sklearn.linear_model import LogisticRegression
LR = LogisticRegression()
LR.fit(X_train,Y_train)

Y_pred = LR.predict(X_test)

from sklearn.metrics import classification_report
print(classification_report(Y_test,Y_pred))