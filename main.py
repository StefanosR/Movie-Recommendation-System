import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

# the next 2 lines (temporary), change the working directory while executing the file, so that VS Code understands csv file names 
# without needing the full path of the datasets.. Reminder to fix this issue internally (or include full paths) and delete this code 

# import os
# os.chdir(r'C:\Users\Stefanos\OneDrive\Υπολογιστής\Fast Projects\Decision Theory\Decision-Theory')

# ---------------------------------------------------------------------------------------------------------------------------------------------------

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

# save matrix to a csv file in case we want to manually check the data
# matrix.to_csv("matrix.csv", header=None, index=None)

# Output information about our data

# print(ratings.head(), "\n") # check the head of the data
# print(movies.head(), "\n") # check the head of the data
# print(ratings.describe(), "\n") # rating statistics
# print(movies.describe(), "\n") # movie statistics

# should we merge the datasets?
# change timestamps to datetime!

# ---------------------------------------------------------------------------------------------------------------------------------------------------
# Data training του ζούλφου

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
        # do nothing
        pass
    
# reminder to fix the output format

print('You should also watch:\n', movies[['title']].loc[movies['movie_id'] == moviesRec.idxmax()])

# ---------------------------------------------------------------------------------------------------------------------------------------------------

# Task: Content based filtering -> Πρόταση ταινιών βάση του ιστορικού του χρήστη (είδη ταινιών που έχει δει)
# Να αντιστοιχίσω τα είδη ταινιών σε αριθμούς και να κάνω knn (δλδ συγγενικά είδη θα βρίσκονται κοντά - εαν γίνεται)
# Να συνδυαστούνε τα είδη ταινιών με τα στοιχεία χρηστών όπως επάγγελμα και ταχυδρομικό κώδικα -> Κατηγοριοποίηση μοτίβων

# 1. Φτιάχνω το titles.csv όπου κρατάω id, τίτλο και είδος ταινίας για να χρησιμοποιηθεί 
# στο μέλλον για πρόταση ταινιών που σχετίζονται με τα είδη ταινιών που βλέπει ο χρήστης

# Διαβάζω το movies.csv για να πάρω όλα τα είδη ταινιών και δημιουργία του titles
titles_header = ['movie_id', 'title', 'genres']
titles = pd.read_csv('movies.csv', sep='\t', names=titles_header, usecols=[1, 2, 3], encoding="ISO-8859-1", low_memory=False, header=0)

# Έλεγχος δεδομένων
# print(titles.head(), "\n")

# Αποθήκευση σε csv 
titles.to_csv("1_titles.csv", index=None)

# Δημιουργία νέου dataframe ids που περιέχει μόνο τα movie id των ταινιών και τα είδη που τους αντιστοιχούν
ids_header = ["movie id", "genres"]
ids = pd.read_csv("1_titles.csv",  sep=',', names=ids_header, usecols=[0, 2], encoding="ISO-8859-1", low_memory=False, header=0)

# Έλεγχος δεδομένων
# print(ids.head())

# Αποθήκευση σε csv 
ids.to_csv("2_ids.csv", index=None)

# 2. Δημιουργία του dataframe genres που περιέχει μόνο τα είδη των ταινιών του αρχικού dataset 
# προκειμένου να φιλτραριστούνε και να χρησιμοποιηθούνε τα συγκεκριμένα δεδομένα
genres_header = ["genres"]
genres = pd.read_csv("2_ids.csv", sep=',', names=genres_header, usecols=[1], encoding="ISO-8859-1", low_memory=False, header=0)

# Έλεγχος δεδομένων
# print(genres.head())

# Αποθήκευση σε csv 
genres.to_csv("3_genres.csv", index=None)

# Παίρνουμε τα είδη από το genres και φτιάχνουμε λίστα με όλα τα είδη ταινιών
df = pd.read_csv('3_genres.csv')
list(set(df.genres))

# Με τις συναρτήσεις unique & sorted αφαιρούμε όλες τις επαναλήψεις ειδών και ταξινομούμε αλφαβητικά τη λίστα μας
real_genres = list(df['genres'].unique())
sorted_genres = sorted(real_genres)

# Αποθηκεύουμε τα μοναδικά είδη ταινιών στο df Dataframe και στο sorted_genres csv
df = pd.DataFrame(sorted_genres, columns=["Sorted Genres:"])
df.to_csv("4_sorted_genres.csv", index=False)

# 3. Αντιστοίχιση των ειδών σε αριθμούς και δημιουργία σχέσεων μεταξύ τους
# 




