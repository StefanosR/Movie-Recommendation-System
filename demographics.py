import pandas as pd
import json
from datetime import datetime 
import time
from scipy import spatial
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import operator

##import excel with ratings
ratings = pd.read_csv('ratings.csv',sep=';',index_col=[0])


##import excel with user details
users_info = pd.read_csv('users.csv',sep='\t',index_col=[0])

##import excel with movie details
movies_info = pd.read_csv('movies.csv', sep='\t', encoding='latin-1',index_col=[0])

## create one merged DataFrame
movie_ratings = pd.merge(movies_info, ratings)
merged = pd.merge(movie_ratings, users_info)

## timestamp to date
merged['timestamp'] = [time.strftime('%Y', time.localtime(x)) for x in merged['timestamp']]
# %d-%m-

## difference in female/male preferences on movie genres
pivoted = merged.pivot_table(index=['genres'],
                           columns=['gender'],
                           values='rating',
                           fill_value=0)

dic1 = {} #dic1 = dictionary for women
count1 = {} #counter for owmen
dic2 = {} #dic2 = dictionary for men
count2 = {} #counter for men
for genre in pivoted.iterrows():
   if genre[0] not in dic1:
      dic1[genre[0]] = genre[1][0]
      count1[genre[0]] = 1
   else:
      dic1[genre[0]] += genre[1][0]
      count1[genre[0]] += 1

   if genre[0] not in dic2:
      dic2[genre[0]] = genre[1][1]
      count2[genre[0]] = 1
   else:
      dic2[genre[0]] += genre[1][1]
      count2[genre[0]] += 1

for genre in dic1:
   dic1[genre] = dic1[genre]/count1[genre]

for genre in dic2:
   dic2[genre] = dic2[genre]/count2[genre]

dic3 = {} #dic3 = dictionary of diference female - male

for genre in dic1:
   if genre in dic2:
      dic3[genre] = dic1[genre] - dic2[genre]
   else:
      dic3[genre] = dic1[genre] - 0

lists = sorted(dic3.items()) # sorted by key, return a list of tuples

x, y = zip(*lists) # unpack a list of pairs into two tuples


plt.barh(x[0:49],y[0:49])
plt.title('Male vs. Female Avg. Ratings\n(Difference > 0 = Favored by women)')

plt.show()
##
plt.barh(x[50:100],y[50:100])
plt.title('Male vs. Female Avg. Ratings\n(Difference > 0 = Favored by women)')
plt.show()
##
plt.barh(x[101:151],y[101:151])
plt.title('Male vs. Female Avg. Ratings\n(Difference > 0 = Favored by women)')
plt.show()
##
plt.barh(x[152:202],y[152:202])
plt.title('Male vs. Female Avg. Ratings\n(Difference > 0 = Favored by women)')
plt.show()
##
plt.barh(x[203:253],y[203:253])
plt.title('Male vs. Female Avg. Ratings\n(Difference > 0 = Favored by women)')
plt.show()
##
plt.barh(x[254:301],y[254:301])
plt.title('Male vs. Female Avg. Ratings\n(Difference > 0 = Favored by women)')
plt.show()

##find popular movie depending on age desc

by_age = merged.pivot_table(index=['movie_id', 'title'],
                           columns=['age_desc'],
                           values='rating',
                           fill_value=0)

age_df = (merged.groupby(['age_desc','title'])['rating'].size().sort_values(ascending=False) 
   .reset_index(name='counts') 
   .drop_duplicates(subset='age_desc'))
print(age_df)
# Pie chart, where the slices will be ordered and plotted counter-clockwise:
labels = age_df['age_desc']
sizes =  age_df['counts']

fig1, ax1 = plt.subplots()
ax1.pie(sizes,  labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()

maxValues_age = by_age.max() 
maxValueIndex_age = by_age.idxmax()
#print( maxValueIndex_age)

##find popular movie depending on occupation desc

by_occ = merged.pivot_table(index=['movie_id', 'title'],
                           columns=['occ_desc'],
                           values='rating',
                           fill_value=0)
occ_df = (merged.groupby(['occ_desc','title'])['rating'].size().sort_values(ascending=False) 
   .reset_index(name='counts') 
   .drop_duplicates(subset='occ_desc'))
print(occ_df)

maxValues_occ = by_occ.max() 
maxValueIndex_occ = by_occ.idxmax()
#print( maxValueIndex_occ)

##find popular movie depending on zipcode
zip_df = (merged.groupby(['zipcode','title'])['rating'].size().sort_values(ascending=False) 
   .reset_index(name='counts') 
   .drop_duplicates(subset='zipcode'))
print(zip_df)

##find popular movie depending on year

year_df = (merged.groupby(['timestamp','title'])['rating'].size().sort_values(ascending=False) 
   .reset_index(name='counts') 
   .drop_duplicates(subset='timestamp'))
print(year_df)