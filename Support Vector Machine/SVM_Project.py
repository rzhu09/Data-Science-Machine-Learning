import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# famous iris dataset, predict which is which

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

iris = sns.load_dataset('iris')

plt.show(sns.pairplot(iris, hue='species'))
setosa = iris[iris['species'] == 'setosa']
plt.show(sns.kdeplot(setosa['sepal_width'], setosa['sepal_length'])) # kde plot of sepal length vs sepal width for setosa flower

from sklearn.model_selection import train_test_split

X = iris.drop('species',axis=1)
y = iris['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)

from sklearn.svm import SVC

model = SVC()
model.fit(X_train, y_train)

preditctions = model.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix

print(confusion_matrix(y_test,preditctions))
print("\n")
print(classification_report(y_test,preditctions))

from sklearn.model_selection import GridSearchCV

param_grid = {'C': [0.1,1, 10, 100, 1000], 'gamma': [1,0.1,0.01,0.001,0.0001]}
grid = GridSearchCV(SVC(), param_grid, verbose=3)
grid.fit(X_train, y_train)

grid_predictions = grid.predict(X_test)

print(confusion_matrix(y_test, grid_predictions))
print("\n")
print(classification_report(y_test,grid_predictions))
