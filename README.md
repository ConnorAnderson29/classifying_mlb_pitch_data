# Classifying MLB In-Game Pitches
## By Werlindo Mangrobang and Connor Anderson

## Business Case:
We created a pitch recognition model that can correctly classify pitches given data recorded by Major League Baseball. 
This model was designed to help enhance the broadcast of ROOT sports. ROOT Sports is a regional channel for the Pacific Northwest
and is the home of the Seattle Mariners. 

The ROOT Sports broadcast is falling behind in complexity and technology in comparison with national brands like FOX and ESPN.
Incorporating our pitch classification model to the broadcast would add accuracy and engagement as well as creating a stepping stone
to adding more features to the broadcast. 

## DATA Understanding:

The Data were worked was collected using a MLB Stats API. We were able to pull in all game data housed on the MLB site.
For our purposes we needed pitch-by-pitch data, so we filtered our data to include all information for every single pitch. 

## DATA Preparation:

In order to use our model, we needed to coerce the data we collected into a Pandas DataFrame. Our data gave us the complete game
situation for each pitch in a row. This made it very easy to feed our data into our model later. 
To create a smoother modeling process, we created a data pipeline. The pipeline took in our dataframe data, and fit it to each model.
It one-hot encoded our categorical data and fed our data through a model.

## Modeling

We tried various classification models to try and determine the most successful model. This process was fairly simple once we had 
created the data pipeline. Using the pipeline allowed us to try 4 different models at once and compare them. The model that 
produced the best output was a Gradient Boosted Trees model. We were able to predict pitches with a cross valdidated 85% accuracy. 

## Evaluation

Our Gradient Boosted Trees model produced the best output for us. We trained it on test data and a validation set. It was cross
validated and returned an 85% accuracy score. Random guess in the MLB is around 55% accurate, and that is only classifying if it is 
a fastball or not. Our model correctly predicted over 6 pitches at that accuracy. The mistakes our model did make were on pitches with
very similar movement patterns(i.e sinkers and 2-seam fastballs). It rarely classified pitches that were not similar incorrectly. 



