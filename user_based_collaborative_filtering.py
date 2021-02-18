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
            nonZero = np.count_nonzero(nearestUsers[column])
            if nonZero != 0:
                predictedRatings[column] = sum(nearestUsers[column]) / nonZero
            else:
                predictedRatings[column] = 0

        # calculate the RMSE error only for the movies which the user has already rated
        rmse = 0
        counter = 0
        for column in test:
            if test.loc[index, column] != 0:
                rmse += pow(predictedRatings[column] - test.loc[index, column], 2)
                counter += 1

        RMSE.append(math.sqrt(rmse / counter))

    return sum(RMSE) / 604, knnModel


def recommendMovies(user, knnModel, kValue, N):
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

    for movie in user:
        if movie != 0:
            predictedRatings[movie] = 0

    # find the N movies with the best rating
    res = dict(sorted(predictedRatings.items(), key=itemgetter(1), reverse=True)[:N])
    return res, predictedRatings, nearestUsers


def noRatings(newUser, users, kValue, N, matrix):
    onlyUsers = users[['gender', 'age']]
    onlyUsers[:]['age'] = abs(onlyUsers[:]['age'] - newUser['age'])

    knnModel = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=kValue)
    knnModel.fit(onlyUsers)

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
rmse1, mdl5 = trainModel(25, train, test)
rmse2, mdl10 = trainModel(50, train, test)
rmse3, mdl20 = trainModel(75, train, test)
rmse4, mdl30 = trainModel(100, train, test)
rmse5, mdl40 = trainModel(125, train, test)
rmse6, mdl50 = trainModel(150, train, test)
rmse7, mdl100 = trainModel(175, train, test)
rmse8, mdl200 = trainModel(200, train, test)
rmse9, mdl400 = trainModel(225, train, test)
rmse10, mdl500 = trainModel(250, train, test)
rmse11, mdl1000 = trainModel(275, train, test)
rmse12, mdl2000 = trainModel(300, train, test)

# plot the RMSEs
plt.plot([25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300],
         [rmse1, rmse2, rmse3, rmse4, rmse5, rmse6, rmse7, rmse8, rmse9, rmse10,
          rmse11, rmse12])

# we choose the number of the nearest neighbors of the model according to the RMSE metric
# and we recommend for a random user of the test set N movies that he has not already seen
# and we predict that he is going to enjoy
bestModel = mdl20
kValue = 20
user = test.iloc[random.randrange(1, 604, 1)]
N = 10
moviesID, pr, nu = recommendMovies(user, bestModel, kValue, N)

recMovies = []
for ident in moviesID:
    recMovies.append(movies.loc[ident]['title'])

for movie in recMovies:
    print(movie)

print('*********')
# for a new user that has not rated any movies, we use his age and gender to find
# other similar users and recommend him some movies to get him started
# (tha dataset did not have users of this kind, se we created one)
newUser = pd.DataFrame({'user_id': [6041], 'gender': ['0'], 'age': [23]})
mID = noRatings(newUser, users, kValue, N, matrix)

recMovies2 = []
for ident in mID:
    recMovies2.append(movies.loc[ident]['title'])

for movie in recMovies2:
    print(movie)
