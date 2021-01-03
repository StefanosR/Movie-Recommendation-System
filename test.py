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

#vazoume to eidos( h eidh ) kathe tenias se enan pinaka kai ta xorizoyme me ,
merged['genres'] = merged['genres'].str.strip('[]').str.replace(' ','').str.replace("'",'')
merged['genres'] = merged['genres'].str.split('|')
merged['age_desc'] = merged['age_desc'].str.strip('[]').str.replace(' ','').str.replace("'",'')
merged['age_desc'] = merged['age_desc'].str.split(' ')


#timestamp se hmerominia
merged['timestamp'] = [time.strftime(' %d-%m-%Y', time.localtime(x)) for x in merged['timestamp']]

#merged['gender'] = merged['gender'].map({'F': 1, 'M': 0})# metatropis ton filon apo M k F se 0 kai 1 (male = 0 female = 1)
print(merged.tail())
genderList = ['F','M'] #lista me ta 2 fila 
def binary(gender_List):
    binaryList = []
    
    for gender in genderList:
        if gender in gender_List:
            binaryList.append(1)
        else:
            binaryList.append(0)
    
    return binaryList
merged['gender_bin'] = merged['gender'].apply(lambda x: binary(x))

###################################################################################################

#ftiaxnoyme mia lista me ta monadika eidh tenion
genreList = []
for index, row in merged.iterrows():
    genres = row["genres"]
    
    for genre in genres:
        if genre not in genreList:
            genreList.append(genre)
genreList[:10] #now we have a list with unique genres


#vazoume ton arithmo 1 otan to genre mias tenias einai idio me thn lista apo to unique genres poy ftiaksame prin kai 0 sta ipolipa
def binary(genre_list):
    binaryList = []
    
    for genre in genreList:
        if genre in genre_list:
            binaryList.append(1)
        else:
            binaryList.append(0)
    
    return binaryList
merged['genres_bin'] = merged['genres'].apply(lambda x: binary(x))


ratings_matrix = merged.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)
genre_matrix = pd.DataFrame(merged['genres_bin'].to_list(), columns =['Drama', 'Animation', 'Childrens', 'Musical', 'Romance', 'Comedy', 'Action', 'Adventure', 'Fantasy', 'Sci-Fi', 'War', 'Thriller', 'Crime', 'Mystery', 'Western', 'Horror', 'Film-Noir', 'Documentary'],index=merged.movie_id)
gender_matrix = pd.DataFrame(merged['gender_bin'].to_list(),index=merged.user_id, columns=['F','M'])
print(gender_matrix)

