### Item Similarity Recommnder System

Using movie Data from IMDB

Data Looks Like:

![1_](https://user-images.githubusercontent.com/60201899/89479596-3ccba500-d761-11ea-914d-7354a9e50aae.PNG)

Movie Title Data Used:

![2_](https://user-images.githubusercontent.com/60201899/89479598-3d643b80-d761-11ea-9768-b09959360b1c.PNG)

Recommending a movie based off the correlation between movie rating 

![3_](https://user-images.githubusercontent.com/60201899/89479599-3d643b80-d761-11ea-9bf7-6e1ad5247df8.PNG)

![5_](https://user-images.githubusercontent.com/60201899/89479603-3d643b80-d761-11ea-8255-76381ae0c205.PNG)

Given that the user enjoys ' Star Wars (1977) then the system will recommend:

 - Empire Strikes Back (1980)
 - Return of the Jedi
 - Raiders of the Lost Ark
 .... etc.

![6_](https://user-images.githubusercontent.com/60201899/89479812-caa79000-d761-11ea-9249-5ea72a079ec6.PNG)


Given that the user enjoys the movie 'Liar Liar' then the system will recommend:
 - Batman Forever
 - Mask, The
 - Down Periscope
 .... etc. 

![4_](https://user-images.githubusercontent.com/60201899/89479601-3d643b80-d761-11ea-90af-8ea52774af2c.PNG)


Conclusion: 

Given a list of user_id's who have rated a list of movies, we can find similar movies given a movie title
based on what other user_id's rated other movies. i.e. user 0 rates star wars as 5/5, they also rate Empire Strikes Back
with a 5/5 rating, hence if someone just finished star wars and wanted another movie suggestions, then Empire Strikes Back movie would get recommended because user 0 has seen both and given both a good ratings. This is done on a larger scale and only the top rated movies will be recommended. 
Correlation (star wars) -> Returns 'Empire Strikes Back' 1.00

We can observe that finding similar movies using this correlation is relatively efficient.

