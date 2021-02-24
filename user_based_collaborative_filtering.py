import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
import math
import matplotlib.pyplot as plt
from operator import itemgetter
import random

def trainModel(kValue, train, test):
    # convert dataframe of movie features to scipy sparse matrix for efficiency
    featuresTrain = csr_matrix(train.values)
    featuresTest = csr_matrix(test.values)

    # train knn model, using cosine as similarity metric
    knnModel = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=kValue)
    knnModel.fit(featuresTrain)

    # for each user in the test set do the following:
    # - find the k-nearest neighbors according to the model trained
    # - predict the ratings for the movies
    # - calculate the MAE(Mean Absolute Error) between the predicted ratings and the actual ratings
    MAE = []

    for index, row in test.iterrows():
        # the indexes returned must be used with iloc not loc
        distances, indexes = knnModel.kneighbors([test.loc[index, :]])
        nearestUsers = train.iloc[indexes[0]]
        predictedRatings = {}

        # predict the ratings for the movies
        for column in nearestUsers:
            nonZero = np.count_nonzero(nearestUsers[column])
            if nonZero != 0:
                predictedRatings[column] = sum(nearestUsers[column]) / nonZero
            else:
                predictedRatings[column] = 0

        # calculate the MAE error only for the movies which the user has already rated
        mae = 0
        counter = 0
        for column in test:
            if test.loc[index, column] != 0:
                mae += abs(predictedRatings[column] - test.loc[index, column])
                counter += 1

        MAE.append(mae / counter)

    # return the MAE and the model trained
    return sum(MAE) / 604, knnModel


def recommendMovies(user, knnModel, N):
    # find the nearest neighbors of user
    distances, indexes = knnModel.kneighbors([user])
    nearestUsers = train.iloc[indexes[0]]
    predictedRatings = {}

    # predict the ratings for the movies
    for column in nearestUsers:
        nonZero = np.count_nonzero(nearestUsers[column])
        if nonZero != 0:
            predictedRatings[column] = sum(nearestUsers[column]) / nonZero
        else:
            predictedRatings[column] = 0

    # if the user has already seen a movie make the predicted rating 0
    # so the movie doesn't get recommended to the user (he has already seen it)
    for movie in user:
        if movie != 0:
            predictedRatings[movie] = 0

    # find the N movies with the best rating
    res = dict(sorted(predictedRatings.items(), key=itemgetter(1), reverse=True)[:N])
    return res


def noRatings(newUser, kValue, N):
    # from user.csv keep only gender and age
    onlyUsers = users[['gender', 'age']]
    # subtract from every user's age the age of the user that we want to recommend movies
    onlyUsers[:]['age'] = abs(onlyUsers[:]['age'] - newUser['age'])

    # train knn model, using cosine as similarity metric
    knnModel = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=kValue)
    knnModel.fit(onlyUsers)

    # find the nearest neigbors of user
    distances, indexes = knnModel.kneighbors(newUser[['gender', 'age']])
    nearestUsers = matrix.iloc[indexes[0]]
    predictedRatings = {}

    # predict the ratings for the movies
    for column in nearestUsers:
        predictedRatings[column] = sum(nearestUsers[column]) / kValue

    # find the N movies with the best rating
    res = dict(sorted(predictedRatings.items(), key=itemgetter(1), reverse=True)[:N])
    return res


# -----//-----  MAIN STARTS HERE -----//------

# read data with panda, only the columns that are needed
# ratings.csv
r_cols = ['user_id', 'movie_id', 'rating']
ratings = pd.read_csv('ratings.csv', sep=';', names=r_cols, usecols=[1, 2, 3], encoding="ISO-8859-1", low_memory=False,
                      header=0)

# movies.csv
m_cols = ['movie_id', 'title']
movies = pd.read_csv('movies.csv', sep='\t', names=m_cols, usecols=[1, 2], encoding="ISO-8859-1", low_memory=False,
                     header=0)

# users.csv
u_cols = ['user_id', 'gender', 'age']
users = pd.read_csv('users.csv', sep='\t', names=u_cols, usecols=[1, 2, 3], encoding="ISO-8859-1", low_memory=False,
                    header=0)

# - replace the F with 0 and the M with 1 in the gender column of users
users = users.replace({'gender': {'F': 0, 'M': 1}})

# create m*n matrix where m is the number of the users and n the number of the movies,
# each cell contains the rating of user i for the movie j
matrix = ratings.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)

# split the dataset into training an test set 90%-10%
train, test = train_test_split(matrix, test_size=0.1)

# test the model for different number of nearest neighbors
# rmse1, mdl5 = trainModel(50, train, test)
# rmse2, mdl10 = trainModel(100, train, test)
# rmse3, mdl20 = trainModel(150, train, test)
# rmse4, mdl30 = trainModel(200, train, test)
# rmse5, mdl40 = trainModel(250, train, test)
rmse6, mdl50 = trainModel(300, train, test)
# rmse7, mdl100 = trainModel(350, train, test)
# rmse8, mdl200 = trainModel(400, train, test)
# rmse9, mdl400 = trainModel(450, train, test)
# rmse10, mdl500 = trainModel(500, train, test)
# rmse11, mdl1000 = trainModel(550, train, test)
# rmse12, mdl2000 = trainModel(600, train, test)

# plot the RMSEs
# plt.plot([50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600],
#          [rmse1, rmse2, rmse3, rmse4, rmse5, rmse6, rmse7, rmse8, rmse9, rmse10, rmse11, rmse12])
# plt.plot([50, 600], [rmse1, rmse12])


# we choose the number of the nearest neighbors of the model according to the MAE metric
# and we recommend for a random user of the test set N movies that he has not already seen
# and we predict that he is going to enjoy
bestModel = mdl50
kValue = 300
user = test.iloc[random.randrange(1, 604, 1)]
N = 10
moviesID = recommendMovies(user, bestModel, N)

recMovies = []
for ident in moviesID:
    recMovies.append(movies.loc[ident]['title'])

for movie in recMovies:
    print(movie)

print('*********')
# # for a new user that has not rated any movies, we use his age and gender to find
# # other similar users and recommend him some movies to get him started
# # (the dataset did not have users of this kind, se we created one)
newUser = pd.DataFrame({'user_id': [6041], 'gender': ['0'], 'age': [23]})
mID = noRatings(newUser, kValue, N)

recMovies2 = []
for ident in mID:
    recMovies2.append(movies.loc[ident]['title'])

for movie in recMovies2:
    print(movie)
