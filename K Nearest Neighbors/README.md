### Using artificially created "classified" data

Using a K nearest neighbors classifier to predict the given target class

Initially the data looks like, Data Frame 1:

![lg7](https://user-images.githubusercontent.com/60201899/88076240-d4b57600-cb47-11ea-95f9-014c5e684840.PNG)

After scaling down the data values taking out the target class, Data Frame 2:

![lg8](https://user-images.githubusercontent.com/60201899/88076317-ed259080-cb47-11ea-88e7-f7a0fe6ff8d7.PNG)

We can then use Data Frame 2 to train test split and predict Data Frame 1's target class

![lg9](https://user-images.githubusercontent.com/60201899/88076606-4988b000-cb48-11ea-9b80-fb36112bb9a1.PNG)

I have set the K value to 1 here ^, however, to find out which K value is best, I looked at the error rates

K = 1:

![lg10](https://user-images.githubusercontent.com/60201899/88077062-d5024100-cb48-11ea-9c68-d7c2742e8fc2.PNG)

![lg6](https://user-images.githubusercontent.com/60201899/88076823-88b70100-cb48-11ea-84e6-2ef03157647d.PNG)

From this, we can see that K values of 24/25 would yield a lower ereror rate so hence we can adjust the K value for better results

K = 24:

![lg11](https://user-images.githubusercontent.com/60201899/88077147-ee0af200-cb48-11ea-96a9-b2a13e540afb.PNG)

### Conclusion: K = 24 returns better results
