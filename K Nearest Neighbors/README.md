### Using artificially created "classified" data

Using a K nearest neighbors classifier to predict the given target class

Initially the data looks like, Data Frame 1:

![lg7](https://user-images.githubusercontent.com/60201899/88076240-d4b57600-cb47-11ea-95f9-014c5e684840.PNG)

After scaling down the data values taking out the target class, Data Frame 2:

![lg8](https://user-images.githubusercontent.com/60201899/88076317-ed259080-cb47-11ea-88e7-f7a0fe6ff8d7.PNG)

We can then use Data Frame 2 to train test split and predict Data Frame 1's target class

![lg9](https://user-images.githubusercontent.com/60201899/88076606-4988b000-cb48-11ea-9b80-fb36112bb9a1.PNG)

I have set the K value to 1 here ^, however, to find out which K value is best, I looked at the error rates


