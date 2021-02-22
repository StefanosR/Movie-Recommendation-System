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
merged['timestamp'] = [time.strftime('%Y', time.localtime(x)) for x in merged['timestamp']]
# %d-%m-

##average rating diference between genders
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

by_age = merged.pivot_table(index=['movie_id', 'title'],
                           columns=['age_desc'],
                           values='rating',
                           fill_value=0)

age_df = (merged.groupby(['age_desc','title'])['rating'].size().sort_values(ascending=False) 
   .reset_index(name='counts') 
   .drop_duplicates(subset='age_desc'))
print(age_df)

maxValues_age = by_age.max() 
maxValueIndex_age = by_age.idxmax()
#print( maxValueIndex_age)

##find popular movie depending on occupation desc

by_occ = merged.pivot_table(index=['movie_id', 'title'],
                           columns=['occ_desc'],
                           values='rating',
                           fill_value=0)
occ_df = (merged.groupby(['occ_desc','title'])['rating'].size().sort_values(ascending=False) 
   .reset_index(name='counts') 
   .drop_duplicates(subset='occ_desc'))
print(occ_df)

maxValues_occ = by_occ.max() 
maxValueIndex_occ = by_occ.idxmax()
#print( maxValueIndex_occ)

##find popular movie depending on zipcode
zip_df = (merged.groupby(['zipcode','title'])['rating'].size().sort_values(ascending=False) 
   .reset_index(name='counts') 
   .drop_duplicates(subset='zipcode'))
print(zip_df)

##find popular movie depending on year

year_df = (merged.groupby(['timestamp','title'])['rating'].size().sort_values(ascending=False) 
   .reset_index(name='counts') 
   .drop_duplicates(subset='timestamp'))
print(year_df)