import pandas as pd
import numpy as np
import  matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('KNN_Project_Data')

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

print(df.head())

#plt.show(sns.pairplot(hue='TARGET CLASS',data=df))

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(df.drop('TARGET CLASS', axis=1))

scaled_features = scaler.transform(df.drop('TARGET CLASS', axis=1))
df2 = pd.DataFrame(scaled_features, columns=df.columns[:-1])

#print(df2.head())

from sklearn.model_selection import train_test_split
X = df2
y = df['TARGET CLASS']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

from sklearn.neighbors import  KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train,y_train)

pred = knn.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix

print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))

error_rate = []
for i in range(1,30):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test)) # error rate where predictions were not eqaul to the actual values

plt.figure(figsize=(10,6))
plt.title('Error Rate vs K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
plt.show(plt.plot(range(1,30), error_rate, color='blue', linestyle ='dashed', marker='o', markerfacecolor='red', markersize=10))

# retrain with better k value to lower the error rate

knn = KNeighborsClassifier(n_neighbors=24)

knn.fit(X_train,y_train)

pred = knn.predict(X_test)

print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))


