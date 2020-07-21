# using data from lending club from 2007-2010
# I will try to predict whether or not
# the person who borrowed money paid their loan back in full

import pandas as pd
import numpy as np
import  matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('loan_data.csv')

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# Some information regarding the columns information
# credit policy is whether or not the customer meets the criteria to borrow (1 = okay, 0 = not okay)
# installment is the monthly installments owed by the customer if loan was funded
# dti is the debt to income ratio of the customer (amound of debt diveded by the annual income)
# fico is the credit score the the customer
# revol. bal is the customers revolving balance, the amount unpaid at the end of the credit billing cycle
# inq.last.6mths is the number of inquireies the customer had in the last 6 months
# delinq.2yrs is the number of times the customers had been 30+ days late on a payment in the past 2 years
# pub.rec: the number of derogatory public records the customer has (bankruptcy fillings, tax liens, judgments)

print(df.info())
print(df.head())
print(df.describe())

plt.figure(figsize=(10,6))

plt.hist(df[df['credit.policy']==0]['fico'], alpha=0.5, bins=30, color = 'red', label='credit is 0') #does not meet credit policy
plt.hist(df[df['credit.policy']==1]['fico'], alpha=0.5, bins=30, color='blue', label='credit is 1') #meets credit policy
plt.legend()
plt.xlabel("fico score")
plt.show()

# ^ there are more people that meet the credit policy than do not

plt.hist(df[df['not.fully.paid']==0]['fico'], alpha=0.5, bins=30, color = 'red', label='not fully paid is 0') # not paid fully
plt.hist(df[df['not.fully.paid']==1]['fico'], alpha=0.5, bins=30, color='blue', label='not fully paid is 1') # paid fully
plt.legend()
plt.xlabel("fico score")
plt.show()

plt.figure(figsize=(11,7))

plt.show(sns.countplot(x='purpose',hue='not.fully.paid',data=df, palette='Set1'))
# ^ group by customer purpose for loan and see how many paid back fully and how many didn't

plt.show(sns.jointplot(x='fico', y='int.rate', data=df, color='blue'))
# ^ we can see that as the interests rates go down, the customers fico score goes up

plt.show(sns.lmplot(x='fico', y='int.rate', data=df, hue='credit.policy',col='not.fully.paid',palette='Set1'))
# ^ we can see that there are more customers that meet the credit policy than those that do not

cat_feats = ['purpose']

final_data = pd.get_dummies(df, columns=cat_feats, drop_first=True)

print(final_data.info())

from sklearn.model_selection import train_test_split

X = final_data.drop('not.fully.paid',axis=1)
y = final_data['not.fully.paid']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)

from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)

predictions = dtree.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_test, predictions))

print(confusion_matrix(y_test, predictions))

#Random Forest Model:

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=600)
rfc.fit(X_train, y_train)

predictions2 = rfc.predict(X_test)
print("Random Forest Tree Results: ")
print(classification_report(y_test, predictions2))
print(confusion_matrix(y_test,predictions2))

# overall the random forest tree was able to predict better than the decision tree by a few percent