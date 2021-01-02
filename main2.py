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

#merged['gender'] = merged['gender'].map({'F': 1, 'M': 0})
print(merged.head())
gendreList = ['M','F'] #lista me ta 2 fila 


#sinartisi metatropis ton filon apo M k F se 0 kai 1 (male = 0 female = 1)
def binary(gendre_List):
    binaryList = []
    
    for gender in gendreList:
        if gender in gendre_List:
            binaryList.append(1)
        else:
            binaryList.append(0)
    
    return binaryList
merged['gender_bin'] = merged['gender'].apply(lambda x: binary(x))
##################################################################################################
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

###################################################################################################
#ftiaxnoyme mia lista me ta monadika eidh age_desc
ageList = []
for index, row in merged.iterrows():
    ages = row["age_desc"]
    
    for age in ages:
        if age not in ageList:
            ageList.append(age)
ageList[:10] #now we have a list with unique ages_desc


def binary(ageList):
    binaryList = []
    
    for age in ageList:
        if age in ageList:
            binaryList.append(1)
        else:
            binaryList.append(0)
    
    return binaryList
merged['ages_bin'] = merged['age_desc'].apply(lambda x: binary(x))


def Similarity(movieId1, movieId2):
    
    a = merged.iloc[movieId1]
    b = merged.iloc[movieId2]
    
    genresA = a['genres_bin']
    genresB = b['genres_bin']
    
    genreDistance = spatial.distance.cosine(genresA, genresB)
    
    scoreA = a['gender_bin']
    scoreB = b['gender_bin']
    scoreDistance = spatial.distance.cosine(scoreA, scoreB)
    
    directA = a['ages_bin']
    directB = b['ages_bin']
    directDistance = spatial.distance.cosine(directA, directB)
    
    return genreDistance + directDistance + scoreDistance 

print(Similarity(3,160))


def predict_score():
    name = input('Enter a movie title: ')
    new_movie = merged[merged['title'].str.contains(name)].iloc[0].to_frame().T
    print('Selected Movie: ',new_movie.title.values[0])
    def getNeighbors(baseMovie, K):
        distances = []
    
        for index, movie in merged.iterrows():
            if movie['new_id'] != baseMovie['new_id'].values[0]:
                dist = Similarity(baseMovie['new_id'].values[0], movie['new_id'])
                distances.append((movie['new_id'], dist))
    
        distances.sort(key=operator.itemgetter(1))
        neighbors = []
    
        for x in range(K):
            neighbors.append(distances[x])
        return neighbors
    
    K = 10
    avgRating = 0
    neighbors = getNeighbors(new_movie, K)
    print('\nRecommended Movies: \n')
    for neighbor in neighbors:
        avgRating = avgRating+merged.iloc[neighbor[0]][2]  
        print( merged.iloc[neighbor[0]][0]+" | Genres: "+str(merged.iloc[neighbor[0]][1]).strip('[]').replace(' ','')+" | Rating: "+str(merged.iloc[neighbor[0]][2]))
    
    print('\n')
    avgRating = avgRating/K
    print('The predicted rating for %s is: %f' %(new_movie['original_title'].values[0],avgRating))
    print('The actual rating for %s is %f' %(new_movie['original_title'].values[0],new_movie['vote_average']))

predict_score()
##grafikes parastaseis
#grafiki parastash Top Eidh tenion
plt.subplots(figsize=(12,10))
list1 = []
for i in merged['genres']:
    list1.extend(i)
ax = pd.Series(list1).value_counts()[:10].sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('hls',10))
for i, v in enumerate(pd.Series(list1).value_counts()[:10].sort_values(ascending=True).values): 
    ax.text(.8, i, v,fontsize=12,color='white',weight='bold')
plt.title('Top Genres')
#plt.show()