import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('Ecommerce Customers')

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

#print(df.describe())
#print(df.info())
#print(df.head())

#plt.show(sns.jointplot(x='Time on Website', y='Yearly Amount Spent',data=df ))
# ^ more time on the website means more money spent
#plt.show(sns.jointplot(x='Time on App', y='Yearly Amount Spent', data=df))
# ^ more time on the app means more money spent

#plt.show(sns.jointplot(x='Time on App',y='Length of Membership',kind='hex',data=df))

#plt.show(sns.pairplot(df))
# ^ shows relationships across the entire data set
# we can see that the longer the membership is the more yearly spent

#plt.show(sns.lmplot(x='Length of Membership',y='Yearly Amount Spent',data=df))

# Split data into training and testing sets

y = df['Yearly Amount Spent']
X = df[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


lm = LinearRegression()

lm.fit(X_train,y_train)

#print('Coefficient: ', lm.coef_)

predictions = lm.predict(X_test)

plt.xlabel('Y test')
plt.ylabel('Predicted Y')
#plt.show(plt.scatter(y_test,predictions))


print('Mean Absolute Error: ', metrics.mean_absolute_error(y_test, predictions))
print('Mean Squared Error: ', metrics.mean_squared_error(y_test,predictions))
print('Root Mean Squared Error: ', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

plt.show(sns.distplot((y_test-predictions),bins = 50))
# showing the residual (normally distributed)

coeff = pd.DataFrame(lm.coef_,X.columns)
coeff.columns = ['Coeffecient']
print(coeff)

# ^ we can see that the more time spent the more money spent ($25.98  per unit of time)
# ^ we can also see that the more time spent on the APP the more money spent ($38 per unit of time)
# ^ we can also see that the more time on the Website spent only $0.19 per unit of time
# ^ and then the longer the membership the more money spent ($62 per unit of time)

# Conclude that the app is doing better than that of their website