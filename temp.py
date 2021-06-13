# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.

"""


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
train.isnull().sum()/train.shape[0]*100
train.info()
train.isnull().sum().sort_values(ascending = False)
combine = [train, test]
train['Credit_History'] = train['Credit_History'].fillna(train['Credit_History'].mean())
train['Self_Employed'] = train['Self_Employed'].fillna(train['Self_Employed'].mode()[0])
train['LoanAmount'] = train['LoanAmount'].fillna(train['LoanAmount'].mean())
train['Loan_Amount_Term'] = train['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mean())
train['Gender'] = train['Gender'].fillna(train['Gender'].mode()[0])
train['Married'] = train['Married'].fillna(train['Married'].mode()[0])
train['Self_Employed'].unique()
train['Dependents'].unique()
train['Married'].unique()
train['Gender'].unique()
train['Dependents'] = train['Dependents'].replace('3+', 3)
train['Dependents'] = train['Dependents'].replace('2', 2)
train['Dependents'] = train['Dependents'].replace('1', 1)
train['Dependents'] = train['Dependents'].replace('0', 0)
train['Dependents'].unique()
train['Dependents'].dtype
train['Dependents'] = train['Dependents'].fillna(train['Dependents'].mean())
train.isnull().sum()
train.drop(['Loan_ID'], axis = 1, inplace = True)
obj = train.select_dtypes(include = 'O')
num = train.select_dtypes(exclude = 'O')
obj['Gender'].unique()
train['Gender'] = le.fit_transform(train['Gender'])
train['Gender'] = le.fit_transform(train['Gender'])
train['Married'] = le.fit_transform(train['Married'])
train['Education'].unique()
obj.columns
train['Education'] = le.fit_transform(train['Education'])
train['Self_Employed'] = le.fit_transform(train['Self_Employed'])
train['Property_Area'].unique()
#As it is not 
train
Property_Area = pd.get_dummies(train['Property_Area'])
train = pd.concat([Property_Area, train], axis =1)
train.drop(['Property_Area'], axis = 1, inplace  = True)
train.columns
train['Loan_Status'] = le.fit_transform(train['Loan_Status'])
corrrection = train.corr()
sns.heatmap(corrrection)
#Let us remove highly corrected varibles
correction = train.corr(method = 'pearson')*100
train.isnull().sum()
train
num.columns
sns.boxplot(train['ApplicantIncome'])
len(train['ApplicantIncome'].unique())
from scipy import stats
z_score = stats.zscore(train['ApplicantIncome'])
train.info()
train.isnull().sum()
train['Dependents'] = train['Dependents'].fillna(train['Dependents'].mean())
train.info()
# Now let us find for the outlier
train.columns
sns.scatterplot(train['Rural'])
train['Rural'].unique()
sns.scatterplot(train['Semiurban'])
train['ApplicantIncome'].unique()
train['Education'].dtype
sns.boxplot(train['ApplicantIncome'])
describe = train.describe()
sns.boxplot(train['ApplicantIncome'])
train['ApplicantIncome_Zscore'] = stats.zscore(train['ApplicantIncome'])
new_train = train.loc[train['ApplicantIncome_Zscore'].abs()<=3]
sns.boxplot(new_train['ApplicantIncome_Zscore'])
new_train.columns
#sns.boxplot(new_train1['CoapplicantIncome'])
new_train['CoapplicantIncome_zscore'] = stats.zscore(new_train['CoapplicantIncome'])
new_train1 = new_train.loc[new_train['CoapplicantIncome_zscore'].abs()<=3]
new_train1.columns
train = new_train1
train.columns
sns.boxplot(train['LoanAmount']) #There are outliers
sns.boxplot(train['Loan_Amount_Term']) 
sns.boxplot(train['Credit_History']) 
sns.boxplot(train['Loan_Status']) 
sns.boxplot(train['LoanAmount']) 
sns.boxplot(train['LoanAmount']) 
train['Credit_History'].unique()
train['LoanAmount_z'] = stats.zscore(train['LoanAmount'])
train = train.loc[train['LoanAmount_z'].abs()<=3]
sns.boxplot(train['Credit_History'])
train.columns
train.drop(['ApplicantIncome_Zscore', 'CoapplicantIncome_zscore', 'LoanAmount_z'], axis = 1, inplace = True)
train.columns
for i in train.columns:
    print(i)

print(len(train.columns))
#Now let us check the correlation between the output and other variables
train.columns
sns.scatterplot(x = 'Credit_History', y = 'Loan_Status', data = train)
train.corr()
sns.heatmap(train.corr())
#As semi utben has high correlation we shall remove those features
train.columns
train.drop(['Semiurban'], inplace =True, axis = 1)
train.drop(['Urban'], inplace =True, axis = 1)
print(len(train.columns))
#Now let us split the data in to train and test

X = train.drop(['Loan_Status'], axis = 1)
y = train['Loan_Status']
sns.distplot(X['LoanAmount'])
sns.distplot(X['Loan_Amount_Term'])
sns.distplot(X['Credit_History'])

from sklearn import preprocessing
#X_norm = preprocessing.scale(X)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(X)
#data_scaled.columns

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.25)
#As this is a binary classification, we shall go with Logistic regression first
from sklearn.linear_model import LogisticRegression
LR = LogisticRegression()
LR_model = LR.fit(X_train, y_train)
prediction = LR_model.predict(X_test)
from sklearn.metrics import accuracy_score
Logistic_accuracy = accuracy_score(y_test, prediction)
print(round(Logistic_accuracy*100))

from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier()
RF_model = RF.fit(X_train, y_train)
RF_pred = RF_model.predict(X_test)
print(RF_pred)
RF_acc = accuracy_score(RF_pred,y_test)
print(round(RF_acc))
print(RF_acc*100)



