# natural language processing
# based on yelp review comments, predict how helpful that review was to other customers 


import pandas as pd
import numpy as np
import  matplotlib.pyplot as plt
import seaborn as sns
import string
from nltk.corpus import stopwords

yelp = pd.read_csv('yelp.csv')

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

#print(yelp.head())
#print(yelp.info())
#print(yelp.describe())

yelp['text length'] = yelp['text'].apply(len)
# ^ adding a new column here to see how long the review comment was

print(yelp.head())

grid = sns.FacetGrid(yelp, col='stars')
grid.map(plt.hist,'text length')

plt.show(grid)
# ^ we can see that the majority people who left 5 star ratings also left a comment between lengths 0-1000
# also notice that there are more 3/4/5 star reviews than 1/2 star reviews

plt.show(sns.boxplot(x='stars',y='text length', data=yelp))

plt.show(sns.countplot(x='stars',data=yelp))

star_mean = yelp.groupby('stars').mean()

print(star_mean)
print(star_mean.corr())
# we can see there is a high correlation where as the text length increase, so does the funny score

plt.show(sns.heatmap(star_mean.corr(), annot=True))

# NLP

yelp_df = yelp[(yelp.stars == 1) | (yelp.stars==5)]

X = yelp_df['text']
y = yelp_df['stars']

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()

X = cv.fit_transform(X)

from sklearn.model_selection import  train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3,random_state=101)

from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()

nb.fit(X_train,y_train)

predictions = nb.predict(X_test)

from sklearn.metrics import confusion_matrix,classification_report

print(confusion_matrix(y_test,predictions))
print('\n')
print(classification_report(y_test,predictions))

# we can then do everything ^ using the pipeline method and use TFID transform to weigh the scores

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('count_vector', CountVectorizer()),
    ('Tfid', TfidfTransformer()), # can remove for better results in this case
    ('Navies', MultinomialNB())
])

X = yelp_df['text']
y = yelp_df['stars']
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3,random_state=101)

pipeline.fit(X_train,y_train)
prediction_2 = pipeline.predict(X_test)

print(confusion_matrix(y_test,prediction_2))
print('\n')
print(classification_report(y_test,prediction_2))

# TFID made things worse for some reason
# get better results if we just take out the TFID from the pipeline
