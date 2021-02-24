import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sc
import seaborn as sns
from pandas.core.dtypes.dtypes import str_type
from pandas.io.parsers import FixedWidthReader

# Inner join movie with ratings dataset -> brings the movies only if they are rated
movies = pd.read_csv('1_titles.csv', sep=',')
ratings = pd.read_csv('fixed_ratings.csv', sep=',')
users = pd.read_csv('fixed_users.csv', sep=',')

users_rating = (ratings
                  .set_index("user_id")
                  .join(users.set_index("user_id"),
                        how="left")
                 )

# print(users_rating.head(5))

movies_users_rating = (users_rating
                    .set_index("movie_id")
                    .join(movies.set_index("movie_id"),
                        how="left")
                    )
                       
# print(movies_users_rating.head(5))

movies_users_rating.to_csv("unified.csv", index=None)

# Try to find out which genre <30 prefer
dummies = movies_users_rating['genres'].str.get_dummies()

tidy_movie_ratings = (pd.concat([movies_users_rating, dummies], axis=1)
                       .drop(["timestamp"], axis=1)
                        )
condition = tidy_movie_ratings["age"].astype(int) < 30

prodcount = (tidy_movie_ratings[condition][["age", "genres"]]
             .groupby("age")
             .count()
            )
print(tidy_movie_ratings.head(5),"\n")

print(prodcount.tail())

# https://youtu.be/KLkJQUYFSlA
# https://medium.com/technofunnel/data-categorization-using-scikit-onehotencoder-python-f7686ef43650

# https://docs.google.com/document/d/1SJlskDMNPujvzKN_AnZQwXMwkzn17xTogB0fLBbcAs0/edit#

# a. Ποια είδη βλέπονται πιο πολύ από τους χρήστες (pie chart)  DONE
# b. Ποια είδη βλέπουνε πιο πολύ συγκεκριμένοι χρήστες (ανάλογα με κατηγορία ηλικίας ή φύλο) 
# c. Ποια ταινία είναι πιο δημοφιλής στις τάδες ηλικίες/επαγγέλματα
# d. Ranking ταινιών ανάλογα με τις αξιολογήσεις όπως το imbd
# e. Πόσες ταινίες βλέπει μια κατηγορία ατόμων (πχ οι γυναικες 100 ταινιες, οι αντρες 200)
