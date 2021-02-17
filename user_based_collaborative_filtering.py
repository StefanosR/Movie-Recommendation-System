import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
import math
import matplotlib.pyplot as plt
from operator import itemgetter
import random


def evaluateModel(kValue, train, test):
    # convert dataframe of movie features to scipy sparse matrix for efficiency
    featuresTrain = csr_matrix(train.values)
    featuresTest = csr_matrix(test.values)

    # train knn model, using cosine as similarity metric
    knnModel = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=kValue)
    knnModel.fit(featuresTrain)

    # for each user in the test set do the following:
    # - find the k-nearest neighbors according to the model trained
    # - predict the ratings for the movies
    # - calculate the RMSE(Root Mean Square Error) between the predicted ratings and the actual ratings
    RMSE = []

    for index, row in test.iterrows():
        # the indexes returned must be used with iloc not loc
        distances, indexes = knnModel.kneighbors([test.loc[index, :]])
        nearestUsers = train.iloc[indexes[0]]
        predictedRatings = {}

        # predict the ratings for the movies
        for column in nearestUsers:
            predictedRatings[column] = sum(nearestUsers[column]) / 10

        # calculate the RMSE error only for the movies which the user has already rated
        rmse = 0
        counter = 0
        for column in test:
            if test.loc[index, column] != 0:
                rmse = pow(predictedRatings[column] - test.loc[index, column], 2)
                counter += 1

        RMSE.append(math.sqrt(rmse / counter))

    return sum(RMSE)/604, knnModel


def recommendMovies(user, knnModel, kValue, N):
    distances, indexes = knnModel.kneighbors([user])
    nearestUsers = train.iloc[indexes[0]]
    predictedRatings = {}

    # predict the ratings for the movies
    for column in nearestUsers:
        predictedRatings[column] = sum(nearestUsers[column]) / kValue

    for movie in user:
        if movie != 0:
            predictedRatings[movie] = 0

    res = dict(sorted(predictedRatings.items(), key=itemgetter(1), reverse=True)[:N])
    return res

# -----//-----  MAIN STARTS HERE -----//------

# read data with panda, only the columns that are needed
r_cols = ['user_id', 'movie_id', 'rating']
ratings = pd.read_csv('ratings.csv', sep=';', names=r_cols, usecols=[1, 2, 3], encoding="ISO-8859-1", low_memory=False, header=0)

m_cols = ['movie_id', 'title']
movies = pd.read_csv('movies.csv', sep='\t', names=m_cols, usecols=[1, 2], encoding="ISO-8859-1", low_memory=False, header=0)


# create m*n matrix where m is the number of the users and n the number of the movies,
# each cell contains the rating of user i for the movie j
matrix = ratings.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)


# split the dataset into training an test set 90%-10%
train, test = train_test_split(matrix, test_size=0.1)


# test the model for different number of nearest neighbors
rmse1, mdl5 = evaluateModel(5, train, test)
rmse2, mdl10 = evaluateModel(10, train, test)
rmse3, mdl20 = evaluateModel(20, train, test)
rmse4, mdl30 = evaluateModel(30, train, test)
rmse5, mdl40 = evaluateModel(40, train, test)


# plot the RMSEs
plt.plot([5, 10, 20, 30, 40], [rmse1, rmse2, rmse3, rmse4, rmse5])
bestModel = mdl20
kValue = 20

# we choose the number of the nearest neighbors of the model according to the RMSE metric
# and we recommend for a random user of the test set N movies that he has not already seen
# and we predict that he is going to enjoy
rNum = random.randrange(1, 604, 1)
user = test.iloc[rNum]
N = 10
moviesID = recommendMovies(user, bestModel, kValue, N)

recMovies = []
for ident in moviesID:
    recMovies.append(movies.loc[ident]['title'])

for movie in recMovies:
    print(movie)




































# recommend 5 movies to the user that he has not already seen
# the opinion of the closest user matters more than the opinion of the second closest user etc

# weights = np.array([0.15, 0.14, 0.13, 0.12, 0.11, 0.09, 0.08, 0.07, 0.06, 0.05])
# usersSize = users.shape
# moviesRec = np.zeros(usersSize[1])
# moviesRecId = np.array(users.columns.values.tolist())
#
# for i in range(0, usersSize[1]):
#     try:
#         vec = users.loc[:, i+1].to_numpy()
#         moviesRec[i] = sum(np.multiply(weights, vec))
#     except KeyError:
#         print('')
#
# indOfMoviesToRecommend = moviesRec.argsort()[-10:][::-1] + 1
# moviesToRecommend = []
#
# for i in range(10):
#     moviesToRecommend.append(movies.loc[movies['movie_id'] == indOfMoviesToRecommend[i], 'title'])
#
#
# print('You should also watch these movies!')
# print(moviesToRecommend)


