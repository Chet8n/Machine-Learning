# importing libraries

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# get the data set from the respective directory
# columnNames is the names of column in our data
# note that non csv files can also be read using read_csv but just specify the seperator

columnNames = ["userID","itemID","rating", "timeStamp"]
df = pd.read_csv("E:/python/MovieRecommendation Practice/ml-100k/u.data", sep = "\t", names = columnNames)

# df.head() - displays the first 5 lines of your Data

# df.shape - dimensions of your data e.g. (99999,4) in this data

# number of unique users
df["userID"].nunique()

# number of unique movies
df["itemID"].nunique()

# this file contains mappings of itemID to movie name
movieTitles = pd.read_csv("E:/python/MovieRecommendation Practice/ml-100k/u.item",sep = "\|", header = None)

movieTitles.shape

# consider only frst and sec column
movieTitles = movieTitles[[0,1]]
movieTitles.columns = ["itemID", "title"]
movieTitles.head()

# merge these two data frames:
df = pd.merge(df, movieTitles, on = "itemID")

df.head()


# Exploratory Data Analysis


import matplotlib.pyplot as plot
import seaborn as sns
sns.set_style("white")

# get the mean rating for all movies and sort
df.groupby("title").mean()["rating"].sort_values(ascending=False)

# table with movie name and number of times its rated
df.groupby("title").count()

ratings = pd.DataFrame(df.groupby("title").mean()["rating"])
ratings["numOfRatings"] = pd.DataFrame(df.groupby("title").count()["rating"])
ratings.head()

ratings.sort_values(by = "rating", ascending=False)

# histogram of number of ratings
plot.figure(figsize=(10,6))
plot.hist(ratings["numOfRatings"], bins = 70)
plot.show()

plot.hist(ratings["rating"], bins = 70)
plot.show()

sns.jointplot( x = "rating" ,y = "numOfRatings",data = ratings, alpha = 0.5)


# Creating Movie Recommendation   

movieMat = df.pivot_table(index="userID",columns="title", values = "rating")
movieMat.head()

ratings.sort_values("numOfRatings",ascending=False)


# for any movie how to get all its rating by users
starWarsUserRatings = movieMat["Star Wars (1977)"]
starWarsUserRatings.head()

# get the correlation between the movie and all other movies
similarToStarWars = movieMat.corrwith(starWarsUserRatings)
similarToStarWars.head()
corrStarWars = pd.DataFrame(data = similarToStarWars, columns = ["correlation"])

# get rid of all NaN values
corrStarWars.dropna(inplace=True)
corrStarWars.head()

# higher the correlation the more the movie is similar or rated similar
corrStarWars.sort_values("correlation", ascending=False)

# a movie rated 5 stars by just 1 viewer doesnt make much sense so keep a threshold for number of ratings
corrStarWars =  corrStarWars.join(ratings["numOfRatings"])
corrStarWars.head()
# threshold = 100
corrStarWars[corrStarWars["numOfRatings"] > 100].sort_values("correlation", ascending = False)


# finally that we have understood the concept lets write a prediction function

def predict_movies(movieName):
    movieUserRatings = movieMat[movieName]
    similarToMovie = movieMat.corrwith(movieUserRatings)
    corrToMovie = pd.DataFrame(similarToMovie, columns = ["correlation"])
    corrToMovie.dropna(inplace = True)
    corrToMovie = corrToMovie.join(ratings["numOfRatings"])
    predictions = corrToMovie[corrToMovie["numOfRatings"] > 100].sort_values("correlation", ascending = False)
    
    return predictions
    

p = predict_movies("Raiders of the Lost Ark (1981)")
print(p)
