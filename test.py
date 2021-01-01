import pandas as pd
import json
from datetime import datetime 
import time


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


#timestamp se hmerominia
merged['timestamp'] = [time.strftime(' %d-%m-%Y', time.localtime(x)) for x in merged['timestamp']]


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

#ftiaxnoyme mia lista me ta monadika eidh tenion
genreList = []
for index, row in merged.iterrows():
    genres = row["genres"]
    
    for genre in genres:
        if genre not in genreList:
            genreList.append(genre)
genreList[:10] #now we have a list with unique genres
#print(genreList)


def binary(genre_list):
    binaryList = []
    
    for genre in genreList:
        if genre in genre_list:
            binaryList.append(1)
        else:
            binaryList.append(0)
    
    return binaryList
merged['genres_bin'] = merged['genres'].apply(lambda x: binary(x))
print(merged['genres_bin'].head())











##grafikes parastaseis
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
plt.subplots(figsize=(12,10))
list1 = []
for i in merged['genres']:
    list1.extend(i)
ax = pd.Series(list1).value_counts()[:10].sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('hls',10))
for i, v in enumerate(pd.Series(list1).value_counts()[:10].sort_values(ascending=True).values): 
    ax.text(.8, i, v,fontsize=12,color='white',weight='bold')
plt.title('Top Genres')
#plt.show()
