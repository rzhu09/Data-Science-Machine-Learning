# Item similarity based recommender system

import pandas as pd
import numpy as np
import  matplotlib.pyplot as plt
import seaborn as sns

columns_names = ['user_id','item_id','rating','timestamp']

df = pd.read_csv('u.data',sep='\t', names=columns_names)

print(df.head())

movie_titles = pd.read_csv('Movie_Id_Titles')

print(movie_titles.head())

df = pd.merge(df, movie_titles, on='item_id') # merge the movie titles to df so we can see the titles along side, not just the item_id

print(df.head())

sns.set_style('white')
print(df.groupby('title')['rating'].mean().sort_values(ascending=False).head()) # find top rated movies

# ^ problem with this is that, if a movie was only viewed and reviewed by one person (5/5) then that movie would get recommended ....

print(df.groupby('title')['rating'].count().sort_values(ascending=False).head())

# ^ find the movies that are most rated, most popular is Star Wars 1977 in this case

ratings = pd.DataFrame(df.groupby('title')['rating'].mean())
print(ratings.head())

ratings['num of ratings'] = pd.DataFrame(df.groupby('title')['rating'].count())
print(ratings.head())

#plt.show(ratings['num of ratings'].hist(bins=70))
# ^ most movies have 0-1 ratings

#plt.show(ratings['rating'].hist(bins=70))
# most movies are rated 3-4 stars

#plt.show(sns.jointplot(x='rating',y='num of ratings', data=ratings, alpha=0.5))
# ^ from this we can see that as the num of reviews go up the rated score actually goes up as well, so the more people who review a movie, the higher the movie will be rated

moviemat = df.pivot_table(index='user_id',columns='title',values='rating')

print(moviemat.head())

starwars_ratings = moviemat['Star Wars (1977)']
liar_ratings = moviemat['Liar Liar (1997)']

print(starwars_ratings.head())
# we can see who rated the movie and who didn't as well as the rated score given

similar_to_star = moviemat.corrwith(starwars_ratings)
# movies with a high correlation value is similar to starwars, in this example
# if a user watched star wars and wanted to watch another movie, 'Til there was you' has a correlation value of 0.87 so that'd get recommended

similar_to_liar = moviemat.corrwith(liar_ratings)

corr_star = pd.DataFrame(similar_to_star,columns=['Correlation'])
corr_star.dropna(inplace=True)
print(corr_star.head())

print(corr_star.sort_values('Correlation',ascending=False).head(10))
# ^ data is incorrect, since the movie suggested might be an outliar

corr_star = corr_star.join(ratings['num of ratings'])

print(corr_star.head())
# ^ now next to each movie there is a number of reviews column

#Based on the number of reviews, we filter out those with less than 100 ratings
print(corr_star[corr_star['num of ratings']>100].sort_values('Correlation', ascending=False).head())
# ^ the data here makes more sense

# Now i do the same for the movie Liar Liar

corr_liar = pd.DataFrame(similar_to_liar, columns=['Correlation'])
corr_liar.dropna(inplace=True)

corr_liar = corr_liar.join(ratings['num of ratings'])

print(corr_liar.head())
print(corr_liar[corr_liar['num of ratings']>100].sort_values('Correlation', ascending=False).head())




