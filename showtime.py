import warnings
import pandas as pd
import numpy as np
from pandas.io.parsers import FixedWidthReader
import scipy as sc
import matplotlib.pyplot as plt
import seaborn as sns

#-------------- Settings --------------------------------------------
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 50)
warnings.filterwarnings('ignore')

#-------------- Users ---------------------------------------------
users_header = ['user_id', 'gender',  'age', 'occupation', 'zipcode', 'age_desc', 'occ_desc']

# We create a users dataframe and then remove the 1st column of the original dataset (which is unecessary)
users = pd.read_csv('users.csv', sep='\t', names=users_header, usecols=[1, 2, 3, 4, 5, 6, 7], encoding="ISO-8859-1", low_memory=False, header=0)
users.to_csv("fixed_users.csv", index=None)

# After we save the new dataframe to the fixed csv we re-read the file in order to get our dataframe with the new delimiter and correct columns
users = pd.read_csv('fixed_users.csv', sep=',')

# print how many unique users are in our data
print(f"There are {len(users.user_id.unique())} unique user ids in the data")

#-------------- Ratings ----------------------------------------------
ratings_header = ['user_id', 'movie_id', 'rating', 'timestamp']

# We do the exact same thing as before in order to get the fixed_ratings dataset
ratings = pd.read_csv('ratings.csv', sep=';', names=ratings_header, usecols=[1, 2, 3, 4], encoding="ISO-8859-1", low_memory=False, header=0)
ratings.to_csv("fixed_ratings.csv", index=None)

ratings = pd.read_csv('fixed_ratings.csv', sep=',')

# Commands in case we want to check our Data

# print(ratings.head(), '\n')
# print(ratings['rating'].value_counts(), '\n')
# print(ratings['rating'].describe())
# ratings['rating'].hist(bins=np.arange(11) - 0.5)

#-------------- 1st plot: Ratings Frequency ----------------------------------------------

# Here we plot the frequency of the 5 possible ratings by all users for all the movie ratings that occured
plt.hist(ratings['rating'], rwidth=2, 
         bins=np.arange(11) - 0.5,
         edgecolor='black', linewidth=1.2)

# Plot settings
plt.style.use('seaborn')
plt.xlabel('Movie Rating') 
plt.ylabel('Number of Movies') 
plt.title('Ratings frequency (all users)\n\n', 
          fontweight ="bold")

plt.rwidth = 2
plt.grid(False)
plt.xticks(range(6))
plt.xlim([0, 6])

plt.show() 

#-------------- 2nd plot: Rated Movies----------------------------------------------

# Smart console comment
print('\nBeep boop :) \nI am loading data...')

# We create a movies dataframe and then remove the 1st column of the original dataset (which is unecessary)
movies_header = ['movie_id', 'title', 'genres']
movies = pd.read_csv('movies.csv', sep='\t', names=movies_header, usecols=[1, 2, 3], encoding="ISO-8859-1", low_memory=False, header=0)

# After we save the new dataframe to the fixed csv we re-read the file in order to get our dataframe with the new delimiter and correct columns
movies.to_csv("fixed_movies.csv", index=None)

# Inner join movie with ratings dataset -> brings the movies along only if they are rated
movies_rating = (ratings.set_index("movie_id").join(movies.set_index("movie_id"), how="left"))

# Check outcome
# print(movies_rating.head(2))

# Create dummies to quantify the genres
dummies = movies_rating['genres'].str.get_dummies() 

# Link the dummies to the original movies_rating dataframe
tidy_movie_ratings = (pd.concat([movies_rating, dummies], axis=1)
                       .drop(["timestamp", "genres"], axis=1))

# Separate the production year info into a new column
tidy_movie_ratings["production_year"] = tidy_movie_ratings["title"].str[-5:-1]
tidy_movie_ratings["title"] = tidy_movie_ratings["title"].str[:-7]
tidy_movie_ratings.reset_index(inplace=True)

# Check outcome
# print(tidy_movie_ratings.head(2))

# Condition results for movies before 2001
condition = tidy_movie_ratings["production_year"].astype(int) < 2001
# Now we will count the total number of productions for each year and plot it
prodcount = (tidy_movie_ratings[condition][["production_year", "movie_id"]]
             .groupby("production_year")
             .count())

# Prints how many movie ratings happened in the last 5 years
# print(prodcount.tail())

# Chart a 5 year moving average of the total productions
(prodcount
 .rolling(5).mean().rename(columns={"movie_id":"count"})
 .plot(figsize=(15,5),
       title="Total number of Rated Movies in the dataset - by production year"))

plt.style.use('seaborn')
plt.show()

#-------------- Plot 3: Genre Popularity by Rating Frequency ----------------------------------------------

# Τop 5 genres by the total number of movies
top5_genre = (tidy_movie_ratings.iloc[:, 4:-1] # get the genre columns only
              .sum() # sum them up
              .sort_values(ascending=False) # sort descending
              .head(5) # get the first 5
              .index.values # get the genre names
              )

print('\nTop 5 Genres:', top5_genre)

# Get the ratings for these genres (not average score, but number or ratings)
genre_groups = (tidy_movie_ratings.iloc[:, 4:]
                .groupby("production_year")
                .sum()
               ).loc["1970":"2000", top5_genre] # 1970-2000

# print('\nTheir Ratings:', genre_groups)
genre_groups.rolling(2).mean().plot(figsize=(15,5),
                                    title="Genre popularity (based on rating frequency)")
plt.show()
print('\n')

# -------------------------------------------------------------------------------------------------

# Plot 4: Pie Chart of which genres are most watched by the users (based on ratings)

# Sum the columns of the top 5 genres to find number of total ratings for each
# Equals the sum of all ratings of a genre for all the given years
sum_column = genre_groups.sum(axis=0)
print(sum_column, "\n")

# data = {'genres':['Others'], 'Ratings':[10000]}
# others = pd.DataFrame(data)
# sum_column = pd.concat([sum_column,others], ignore_index=True)

# Data to plot
labels = top5_genre # Top 5 genres we found before
sizes = sum_column # Sum of columns

# Plot settings
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue', 'gray']
explode = (0.1, 0, 0, 0, 0)  # explode 1st slice

plt.pie(sizes, explode=explode, labels=labels, colors=colors,
autopct='%1.1f%%', shadow=True, startangle=140)

plt.title('Genre Popularity (Pie Chart)')
plt.axis('equal')
plt.show()

#-------------- Console Output: Top Rated Action Movies per Decade ----------------------------------------------

# Initialize cols and filter the results for before 2000
cols = ["title", "rating", "production_year", "Action", "movie_id"]
condition0 = tidy_movie_ratings["production_year"].astype(int) < 2000
condition1 = tidy_movie_ratings["Action"] == 1

# Build an "action" table containing the columns we need 
action = (tidy_movie_ratings
         [cols]
         [condition0 & condition1]
         .drop("Action", axis=1)
        )

# Create a new column called decade
action["decade"] = action['production_year'].astype(int)//10*10
# Check outcome
action.head()

# Find how many times a movie was rated
count_group = action.groupby("movie_id").count()["rating"]

# Make a movie list that will contain specific movies according to our needs
# Our movie list need to contain only movies with more than 100 ratings
# This is in order to be fair towards popular movies that have greater impact
movie_list = count_group[count_group > 100].index.values
movie_list[:5]

# The movie_list now contains those movies that have been rated more than 100 times
# Filter action table using the movie_list
condition = action["movie_id"].isin(movie_list)
columns = ["title", "decade", "rating"]
action_filtered = action[condition][columns]

# Build the metrics to get the top_rated movies of the century
top_rate_by_decade = (action_filtered
                     .groupby(["decade", "title"])                             # Group the action_filtered dataframe by decade and title
                     .mean()
                     .sort_values(["decade", "rating"], ascending=False)       # Sort by decade and rating values
                     .groupby(level=0, as_index=False)
                     .apply(lambda x: x.head() if len(x) >= 5 else x.head(1))  # Loop throught the decade groups and show only the top 5
                     .reset_index(level=0, drop=True)
                    ).round(2)                                                 # Round to 2 decimal points

# Slice dataframe option (if we want to remove decades)
# top_rate_by_decade.loc[1950:]

print(top_rate_by_decade)