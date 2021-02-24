from typing import get_args, get_type_hints
import numpy as np
from numpy.lib.arraysetops import unique
from numpy.lib.shape_base import split
import pandas as pd
from collections import Counter
import csv 
from itertools import combinations
import sys

#------------------------------ Περιγραφή --------------------------------------------

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
titles.to_csv("content-based filtering/1_titles.csv", index=None)

# Δημιουργία νέου dataframe ids που περιέχει μόνο τα movie id των ταινιών και τα είδη που τους αντιστοιχούν
ids_header = ["movie id", "genres"]
ids = pd.read_csv("content-based filtering/1_titles.csv",  sep=',', names=ids_header, usecols=[0, 2], encoding="ISO-8859-1", low_memory=False, header=0)

# Έλεγχος δεδομένων
# print(ids.head())

# Αποθήκευση σε csv 
ids.to_csv("content-based filtering/2_ids.csv", index=None)

# 2. Δημιουργία του dataframe genres που περιέχει μόνο τα είδη των ταινιών του αρχικού dataset 
# προκειμένου να φιλτραριστούνε και να χρησιμοποιηθούνε τα συγκεκριμένα δεδομένα. Συν τα CSV files με διαφορετικά δεδομένα
genres_header = ["genres"]
genres = pd.read_csv("content-based filtering/2_ids.csv", sep=',', names=genres_header, usecols=[1], encoding="ISO-8859-1", low_memory=False, header=0)

# Έλεγχος δεδομένων
# print(genres.head())

# Αποθήκευση σε csv 
genres.to_csv("content-based filtering/3_genres.csv", index=None)

# Παίρνουμε τα είδη από το genres και φτιάχνουμε λίστα με όλα τα είδη ταινιών
df = pd.read_csv('content-based filtering/3_genres.csv')
list(set(df.genres))

# Με τις συναρτήσεις unique & sorted αφαιρούμε όλες τις επαναλήψεις ειδών και ταξινομούμε αλφαβητικά τη λίστα μας
real_genres = list(df['genres'].unique())
sorted_genres = sorted(real_genres)

# Αποθηκεύουμε τα είδη ταινιών (ξεχωριστά μεταξύ τους αλλά όχι μοναδικά) στο df Dataframe και στο sorted_genres csv
df = pd.DataFrame(sorted_genres, columns=["Sorted_Genres"])
df.to_csv("content-based filtering/4_sorted_genres.csv", index=False)

# Ορίζουμε το μέγιστο αριθμό στηλών/ειδών που μπορεί να έχει η κάθε ταινία, διαβάζουμε όλες τις γραμές και κρατάμε σε dataframe τα είδη
columns=["col1", "col2", "col3", "col4", "col5", "col6"]
unique_genres = pd.read_csv("content-based filtering/4_sorted_genres.csv",  sep='|', skiprows=[0], names=columns, encoding="ISO-8859-1", low_memory=False, header=None)

# Έλεγχος του πίνακα με τα είδη ανά ταινία
# print(unique_genres.head())

# Δημιουργούμε λίστα για τα δεδομένα αυτά και δημιουργούμε νέο dataframe unique_genres2 που θα κρατάει από τη λίστα τις μοναδικές λέξεις/είδη
list(set(unique_genres.col1))
unique_genres2 = list(unique_genres['col1'].unique())

# List of the unique genres
# print(unique_genres2)

# Μεταφέρουμε τη νέα λίστα μας στο df (ή σε df2 αν θέλουμε) και αποθηκεύουμε σε csv
df = pd.DataFrame(unique_genres2, columns=["Unique Genres:"])
df.to_csv("content-based filtering/5_unique_genres.csv", index=False)

# 3. Αντιστοίχιση των ειδών σε αριθμούς και δημιουργία σχέσεων μεταξύ τους

# Για την δημιουργία των σχέσεων θα μετρήσουμε πόσες φορές το κάθε είδος μια ταινίας συνυπάρχει με ένα άλλο
# Όσο μεγαλύτερη η συχνότητα που εμφανίζονται 2 ή περισσότερα είδη μαζί τόσο πιο δυνατή η σχέση τους
# Επομένως μπορεί να γίνει πρόταση ταινίας με βάση κάποιο δυνατά συσχετισμένο συνδυασμό ειδών με το ιστορικό ταινιών του χρήστη

# Αρχικά θέτουμε έναν αριθμό σε κάθε είδος και ελέγχουμε το αποτέλεσμα αποθηκεύοντας σε csv
df.insert(0, 'No.', df.index + 1, allow_duplicates = False)
df.to_csv("content-based filtering/5_unique_genres.csv", index=False)

#--------------------------------IMPLEMENTATION_1----------------------------------------------------------------

'''
# Περιγραφή μεθόδου: Χρήση της συνάρτησης Counter για να μετρήσει τα unique set από 2 είδη ταινιών (import combinations)
# Πρόβλημα με αυτό το implementation: Χάνονται κάποιες γραμμές ή και προστίθενται παραπάνω στο 6_combinations.csv

counter = Counter()
unique = set()
with open('3_genres.csv') as csvfile:
  next(csvfile) # αγνοεί το header 
  reader = csv.reader(csvfile, delimiter='|')
  for line in reader:
    unique.update(line)
    counter.update(combinations(line, 2))
counter.update({entry: 0 for entry in combinations(unique, 2)})
print(counter)

# O counter είναι λεξικό -> Counter({('Action', 'Adventure'): 39, ('Action', 'Thriller'): 30, ...})
# Αποθηκεύουμε τον counter σε pandas dataframe και αφού μετονομάζουμε τις στήλες αποθηκεύουμε σε csv

comb = pd.DataFrame.from_dict(counter, orient='index').reset_index()
comb = comb.rename(columns={'index':'Relationship', 0:'Count'}) 
comb.to_csv("content-based filtering/6_combinations.csv", index=False)
'''

#--------------------------------IMPLEMENTATION_2----------------------------------------------------------------

# Νέο pandas dataframe που διαβάζει το csv
df = pd.read_csv('content-based filtering/3_genres.csv')

# Check Results
print(df.head(5))

# Κωδικοποίηση των ειδών σε dummies και αντιστοίχιση της ύπαρξης τους σε στήλες με 0 & 1
d = df['genres'].str.get_dummies('|')

#    data1  data2  data3  data4  data5  data6  data7
# 0      1      1      1      0      0      0      0
# 1      1      1      0      0      0      0      0
# 2      1      0      0      1      0      1      0
# 3      0      1      1      0      0      0      0 

# Όπου data1, data2 κλπ είναι τα είδη του dataset και 0-3 τυχαία παραδείγματα από σειρές του 3_genres.csv

# Δημιουργία λεξικού dct για αποθήκευση των δεδομένων που θα παραχθούνε
dct = {}
# Όταν δυο είδη συνυπάρχουνε κρατάμε τον συνδυασμό # Check την for ότι παίρνει σωστά τα δεδομένα
for x, y in combinations(d, r=2):
    try:
      dct[f'{x}:{y}'] = d[[x, y]].eq(1).all(1).sum()
    except KeyError:
      continue
    
# Έλεγχος του αποτελέσματος για τους μοναδικούς συνδυασμούς 
print(dct, "\n")

# Αποθήκευση του αποτελέσματος στο csv     
with open('content-based filtering/6_combinations.csv', 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    writer = csv.writer(csvfile)
    writer.writerow(["Combination", "Number"])
    for row in dct.items():
        csvwriter.writerow(row)
 
# Αποθήκευση του dictionary με τα αποτελέσματα στο comb dataframe (για να φύγουνε τα empty lines)
comb = pd.read_csv('content-based filtering/6_combinations.csv')
comb.to_csv('content-based filtering/6_combinations.csv', index=False) # Μπορούμε να βάλουμε και indexes -> Should we?       

#--------------------------------Sorting_of_the_Combinations----------------------------------------------------------------

# Ονομασία των στηλών για να χρησιμοποιηθεί ο sort_values 
comb.columns = ['Combination','Number']
comb = comb.sort_values(by = ['Number'], kind='quicksort', ascending=False)

# Αποθήκευση και έλεγχος
comb.to_csv("content-based filtering/7_sorted_combs.csv", index=False) # Μπορούμε να βάλουμε και indexes 
print(comb.head(5))

# Έλεγχος του αθροίσματος των συνδυασμών
total_combs = comb['Number'].sum()
print (total_combs)

#--------------------------------Relationship_Strength_Measurement----------------------------------------------------------------

# 4. Μέτρηση των σχέσεων που έχουμε μετρήσει ανάμεσα στα διαφορετικά είδη ταινιών

# Είδη ταξινομήθηκε η λίστα μας, οπότε έχουμε οπτική των πιο δυνατών και πιο αδύναμων σχέσεων
# Combination Counter overall results: 0-226 max relationships

# Power Levels (Relevance of 2 movie genres)
# Zero: <10
# Low: 10-29
# Medium: 30-64
# High: 65-100
# Very High: >100

list = ['V','H','M','L','Z']

# for i in range(150):
#     if comb.columns 