import pandas as pd
import json
from datetime import datetime 
import time
from scipy import spatial
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.neighbors import NearestNeighbors
import operator

##import excel with ratings
ratings = pd.read_csv('ratings.csv',sep=';',index_col=[0])


##import excel with user details
users_info = pd.read_csv('users.csv',sep='\t',index_col=[0])

##import excel with movie details
movies_info = pd.read_csv('movies.csv', sep='\t', encoding='latin-1',index_col=[0])

## create one merged DataFrame
movie_ratings = pd.merge(movies_info, ratings)
merged = pd.merge(movie_ratings, users_info)

## timestamp to date
merged['timestamp'] = [time.strftime(' %d-%m-%Y', time.localtime(x)) for x in merged['timestamp']]

#average rating diference between genders
pivoted = merged.pivot_table(index=['movie_id', 'title'],
                           columns=['gender'],
                           values='rating',
                           fill_value=0)

pivoted['diff'] = pivoted.M - pivoted.F

pivoted.reset_index('movie_id', inplace=True)
disagreements = pivoted[ (pivoted.movie_id > 50 ) & (pivoted.movie_id <= 100)]['diff']
disagreements.sort_values().plot(kind='barh', figsize=[9, 15])
plt.title('Male vs. Female Avg. Ratings\n(Difference > 0 = Favored by Men)')
plt.ylabel('Title')
plt.xlabel('Average Rating Difference');
plt.show()


##find popular movie depending on age desc
age_dataframe = merged[['movie_id','title','user_id','rating','age_desc']]
#by_age =age_dataframe.groupby(['movie_id','title','age_desc']).size()
by_age = age_dataframe.groupby(['title','age_desc'])['rating'].mean().unstack(1).fillna(0)
#by_age['average'] = age_dataframe.groupby(['movie_id','title','age_desc'])['rating'].mean() 
print(by_age)

##find popular movie depending on occupation desc
occ_dataframe = merged[['movie_id','title','user_id','rating','occ_desc']]
by_occ = occ_dataframe.groupby(['title','occ_desc'])['rating'].mean().unstack(1).fillna(0)
print(by_occ)
