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
print(matrix)

# convert dataframe of movie features to scipy sparse matrix for efficiency

features = csr_matrix(matrix.values)


# train knn model, using cosine as similarity metric

knnModel = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=10)
knnModel.fit(features)


# get the 10 nearest neighbors of user

distances, indexes = knnModel.kneighbors([matrix.loc[1, :]])


# recommend 5 movies to the user that he has not already seen

users = matrix.reindex(indexes[0] + 1)
moviesRec = np.sum(users, 0)

for i in range(1, len(moviesRec)):
    try:
        if matrix.loc[1, i] != 0:
            moviesRec[i] = 0
    except KeyError:
        print('')

print('You should also watch', movies[['title']].loc[movies['movie_id'] == moviesRec.idxmax()])













