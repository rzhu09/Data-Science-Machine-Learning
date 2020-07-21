# Predict if a particular internet user clicked on an Advertisement or not

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.offline as py

ad_data = pd.read_csv('advertising.csv')

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

#print(ad_data.head())

#print(ad_data.info())
# we can see that there are 1000 entries here with 10 coluns

#print(ad_data.describe())

#plt.show(ad_data['Age'].hist(bins=30))
# histogram of age ^

#plt.show(sns.jointplot(x='Age', y='Area Income',data=ad_data))
# joint plot of age vs the area of income

#plt.show(sns.jointplot(x=ad_data['Age'], y=ad_data['Daily Time Spent on Site'],kind='kde'))
# joint plot that shows the kde distributions of time spent on site vs age

#plt.show(sns.jointplot(x='Daily Internet Usage', y='Daily Time Spent on Site', data=ad_data))
# joint plot that shows the daily time spent on site vs their daily internet usage

#plt.show(sns.pairplot(hue='Clicked on Ad', data=ad_data))
# with a pair plot we can see which factors are related and unrelated

# Logistic Regression:
from sklearn.model_selection import  train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

X = ad_data[['Daily Time Spent on Site', 'Age', 'Area Income','Daily Internet Usage', 'Male']]
y = ad_data['Clicked on Ad']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)

logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)

predictions = logmodel.predict(X_test)

print(classification_report(y_test, predictions))