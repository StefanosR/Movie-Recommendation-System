import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors


# read data with panda, only the columns that are needed

r_cols = ['user_id', 'movie_id', 'rating']
ratings = pd.read_csv('ratings.csv', sep=';', names=r_cols, usecols=[1, 2, 3], encoding="ISO-8859-1", low_memory=False, header=0)

m_cols = ['movie_id', 'title']
movies = pd.read_csv('movies.csv', sep='\t', names=m_cols, usecols=[1, 2], encoding="ISO-8859-1", low_memory=False, header=0)


# create m*n matrix where m is the number of the users and n the number of the movies,
# each cell contains the rating of user i for the movie j

matrix = ratings.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)


# convert dataframe of movie features to scipy sparse matrix for efficiency

features = csr_matrix(matrix.values)


# train knn model, using cosine as similarity metric

knnModel = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=11)
knnModel.fit(features)


# get the 10 nearest neighbors of user

distances, indexes = knnModel.kneighbors([matrix.loc[1, :]])
users = matrix.reindex(indexes[0][1:11] + 1)

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
        print('')

indOfMoviesToRecommend = moviesRec.argsort()[-10:][::-1] + 1
moviesToRecommend = []

for i in range(10):
    moviesToRecommend.append(movies.loc[movies['movie_id'] == indOfMoviesToRecommend[i], 'title'])


print('You should also watch these movies!')
print(moviesToRecommend)


