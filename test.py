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
grafikidf = merged.drop([ 'movie_id','timestamp','age','occupation','zipcode','age_desc','occ_desc','genres'], axis=1)
grafiki2 = merged.drop([ 'movie_id','timestamp','age','occupation','zipcode','gender','occ_desc','genres'], axis=1)

#vazoume to eidos( h eidh ) kathe tenias se enan pinaka kai ta xorizoyme me ,
merged['genres'] = merged['genres'].str.strip('[]').str.replace(' ','').str.replace("'",'')
merged['genres'] = merged['genres'].str.split('|')
#merged['age_desc'] = merged['age_desc'].str.strip('[]').str.replace(' ','').str.replace("'",'')
#merged['age_desc'] = merged['age_desc'].str.split(' ')



#timestamp se hmerominia
merged['timestamp'] = [time.strftime(' %d-%m-%Y', time.localtime(x)) for x in merged['timestamp']]

#merged['gender'] = merged['gender'].map({'F': 1, 'M': 0})# metatropis ton filon apo M k F se 0 kai 1 (male = 0 female = 1)


male_user_rating =merged.loc[merged['gender'] == 'M'] #an alakso to 1 se 2
female_user_rating =merged.loc[merged['gender'] == 'F']
underaged_user_rating=merged.loc[merged['age_desc']=='Under 18']



ratings_matrix = female_user_rating.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)
#genre_matrix = pd.DataFrame(merged['genres_bin'].to_list(), columns =['Drama', 'Animation', 'Childrens', 'Musical', 'Romance', 'Comedy', 'Action', 'Adventure', 'Fantasy', 'Sci-Fi', 'War', 'Thriller', 'Crime', 'Mystery', 'Western', 'Horror', 'Film-Noir', 'Documentary'],index=merged.movie_id)
#gender_matrix = pd.DataFrame(merged['gender_bin'].to_list(),index=merged.user_id, columns=['F','M'])
#ages_matrix = pd.DataFrame(merged['ages_bin'].to_list(),index=merged.user_id, columns=['Under18', '56+', '25-34', '50-55', '18-24', '45-49', '35-44'])
#genre_matrix = genre_matrix[~genre_matrix.index.duplicated()]

features = csr_matrix(ratings_matrix.values)


knnModel = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=11)
knnModel.fit(features)

distances, indexes = knnModel.kneighbors([ratings_matrix.loc[1, :]])
users = ratings_matrix.reindex(indexes[0][1:11] + 1)

# recommend 5 movies to the user that he has not already seen
# the opinion of the closest user matters more than the opinion of the second closest user etc

weights = np.array([0.15, 0.14, 0.13, 0.12, 0.11, 0.09, 0.08, 0.07, 0.06, 0.05])
usersSize = users.shape
moviesRec = np.zeros(usersSize[1])
moviesRecId = np.array(users.columns.values.tolist())

for i in range(0, usersSize[1]):
    try:
        vec = users.loc[:, i+1].to_numpy()
        moviesRec[i] = sum(np.multiply(weights, vec))
    except KeyError:
        pass

indOfMoviesToRecommend = moviesRec.argsort()[-10:][::-1] + 1
moviesToRecommend = []

for i in range(10):
    moviesToRecommend.append(female_user_rating.loc[female_user_rating['movie_id'] == indOfMoviesToRecommend[i], 'title'])



#print('You should also watch these movies!')
print(moviesToRecommend)

##grafikes parastaseis
grafikidf= grafikidf[grafikidf['gender'] != 0]


grafikidf.groupby('gender')['title'].nunique().plot(kind='bar')
plt.show()

grafiki2=grafiki2[grafiki2['rating'] == 5]

#
grafiki2 =grafiki2.groupby('age_desc').filter(lambda g: len(g) > 1).groupby(['age_desc', 'title']).head(1)
print(grafiki2)
under18_results =grafiki2['title'].value_counts()[grafiki2['title'].value_counts() == grafiki2['title'].value_counts().max()]

print(under18_results)
