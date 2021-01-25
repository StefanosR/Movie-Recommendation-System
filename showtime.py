import warnings

import pandas as pd
import numpy as np
from pandas.io.parsers import FixedWidthReader
import scipy as sc

import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 50)
warnings.filterwarnings('ignore')

#--------------users---------------------------------------------
users_header = ['user_id', 'gender',  'age', 'occupation', 
                'zipcode', 'age_desc', 'occ_desc']

users = pd.read_csv('users.csv', sep='\t', names=users_header, usecols=[1, 2, 3, 4, 5, 6, 7], encoding="ISO-8859-1", low_memory=False, header=0)
users.to_csv("fixed_users.csv", index=None)

users = pd.read_csv('fixed_users.csv', sep=',')

# print(users.head())

print(f"There are {len(users.user_id.unique())} unique user ids in the data")
#--------------ratings----------------------------------------------
ratings_header = ['user_id', 'movie_id', 'rating', 'timestamp']

ratings = pd.read_csv('ratings.csv', sep=';', names=ratings_header, usecols=[1, 2, 3, 4], encoding="ISO-8859-1", low_memory=False, header=0)
ratings.to_csv("fixed_ratings.csv", index=None)

ratings = pd.read_csv('fixed_ratings.csv', sep=',')

# print(ratings.head(), '\n')

# print(ratings['rating'].value_counts(), '\n')

# print(ratings['rating'].describe())

# ratings['rating'].hist(bins=np.arange(11) - 0.5)

plt.hist(ratings['rating'], rwidth=2, 
         bins=np.arange(11) - 0.5,
         edgecolor='black', linewidth=1.2)

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
# Add number of total movie in each bar
#--------------movies----------------------------------------------

print('\nBeep boop :) \nI am loading data...')

# Inner join movie with ratings dataset -> brings the movies only if they are rated
movies = pd.read_csv('1_titles.csv', sep=',')

movies_rating = (ratings
                  .set_index("movie_id")
                  .join(movies.set_index("movie_id"),
                        how="left")
                 )

# print(movies_rating.head(2))

dummies = movies_rating['genres'].str.get_dummies()
tidy_movie_ratings = (pd.concat([movies_rating, dummies], axis=1)
                       .drop(["timestamp", "genres"], axis=1)
                        )

tidy_movie_ratings["production_year"] = tidy_movie_ratings["title"].str[-5:-1]
tidy_movie_ratings["title"] = tidy_movie_ratings["title"].str[:-7]
tidy_movie_ratings.reset_index(inplace=True)

# print(tidy_movie_ratings.head(2))

condition = tidy_movie_ratings["production_year"].astype(int) < 2021

prodcount = (tidy_movie_ratings[condition][["production_year", "movie_id"]]
             .groupby("production_year")
             .count()
            )

# print(prodcount.tail())

(prodcount
 .rolling(5).mean().rename(columns={"movie_id":"count"})
 .plot(figsize=(15,5),
       title="Total number of Rated Movies in the dataset - by production year")
)
plt.style.use('seaborn')
plt.show()

#--------------genres----------------------------------------------

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
                                    title="Genre popularity according to rating frequency)")
plt.show()

#--------------Top Rated Action Movies per Decade----------------------------------------------

cols = ["title", "rating", "production_year", "Action", "movie_id"]
condition0 = tidy_movie_ratings["production_year"].astype(int) < 2000
condition1 = tidy_movie_ratings["Action"] == 1

action = (tidy_movie_ratings
         [cols]
         [condition0 & condition1]
         .drop("Action", axis=1)
        )

action["decade"] = action['production_year'].astype(int)//10*10

action.head()

# Our movie list need to contain only movies with more than 100 ratings
# This is in order to be fair towards popular movies
count_group = action.groupby("movie_id").count()["rating"]
movie_list = count_group[count_group > 100].index.values
movie_list[:5]

# Filter action table using the movie_list
condition = action["movie_id"].isin(movie_list)
columns = ["title", "decade", "rating"]
action_filtered = action[condition][columns]

top_rate_by_decade = (action_filtered
                     .groupby(["decade", "title"])
                     .mean()
                     .sort_values(["decade", "rating"],
                                                ascending=False)
                     .groupby(level=0, as_index=False)
                     .apply(lambda x: x.head() if len(x) >= 5 else x.head(1))
                     .reset_index(level=0, drop=True)
                    ).round(2)

# Slice dataframe option (remove decades)
# top_rate_by_decade.loc[1990:]

print(top_rate_by_decade)

# Since I have dummied the genres make "Ποια είδη βλέπονται πιο πολύ από τους χρήστες (pie chart)"
# -------------------------------------------------------------------------------------------------