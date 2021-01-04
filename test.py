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

##vazoyme to excel me tis vathmologies
ratings = pd.read_csv('ratings.csv',sep=';',index_col=[0])


##vazoyme to excel me ta details twn users
users_info = pd.read_csv('users.csv',sep='\t',index_col=[0])

##vazoyme to excel me ta movies descriptions k titles
movies_info = pd.read_csv('movies.csv', sep='\t', encoding='latin-1',index_col=[0])

##enonoyme ta 3 excel arxeia se 1 pinaka
test = pd.merge(ratings,movies_info, on='movie_id') #enonoyme tous pinakes ratings kai movies info
merged = pd.merge(test,users_info, on = 'user_id') #enonoyme ton pinaka me ratings kai movies info me ton users_info
print(merged)
#vazoume to eidos( h eidh ) kathe tenias se enan pinaka kai ta xorizoyme me ,
merged['genres'] = merged['genres'].str.strip('[]').str.replace(' ','').str.replace("'",'')
merged['genres'] = merged['genres'].str.split('|')
#merged['age_desc'] = merged['age_desc'].str.strip('[]').str.replace(' ','').str.replace("'",'')
#merged['age_desc'] = merged['age_desc'].str.split(' ')


#timestamp se hmerominia
merged['timestamp'] = [time.strftime(' %d-%m-%Y', time.localtime(x)) for x in merged['timestamp']]

#merged['gender'] = merged['gender'].map({'F': 1, 'M': 0})# metatropis ton filon apo M k F se 0 kai 1 (male = 0 female = 1)
print(merged)

male_user_rating =merged.loc[merged['gender'] == 'M'] #an alakso to 1 se 2
female_user_rating =merged.loc[merged['gender'] == 'F']
underaged_user_rating=merged.loc[merged['age_desc']=='Under 18']

print(underaged_user_rating)

ratings_matrix = female_user_rating.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)
#genre_matrix = pd.DataFrame(merged['genres_bin'].to_list(), columns =['Drama', 'Animation', 'Childrens', 'Musical', 'Romance', 'Comedy', 'Action', 'Adventure', 'Fantasy', 'Sci-Fi', 'War', 'Thriller', 'Crime', 'Mystery', 'Western', 'Horror', 'Film-Noir', 'Documentary'],index=merged.movie_id)
#gender_matrix = pd.DataFrame(merged['gender_bin'].to_list(),index=merged.user_id, columns=['F','M'])
#ages_matrix = pd.DataFrame(merged['ages_bin'].to_list(),index=merged.user_id, columns=['Under18', '56+', '25-34', '50-55', '18-24', '45-49', '35-44'])
#genre_matrix = genre_matrix[~genre_matrix.index.duplicated()]

features = csr_matrix(ratings_matrix.values)
#features = csr_matrix(genre_matrix.values)
#features = csr_matrix(gender_matrix.values)
#features = csr_matrix(ages_matrix.values)

knnModel = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=10)
knnModel.fit(features)

distances, indexes = knnModel.kneighbors([ratings_matrix.loc[1, :]])

# recommend 5 movies to the user that he has not already seen

users = ratings_matrix.reindex(indexes[0] + 1)
moviesRec = np.sum(users, 0)

for i in range(1, len(moviesRec)):
    try:
        if ratings_matrix.loc[1, i] != 0:
            moviesRec[i] = 0
    except KeyError:
        pass

print('You should also watch', female_user_rating[['title']].loc[female_user_rating['movie_id'] == moviesRec.idxmax()])