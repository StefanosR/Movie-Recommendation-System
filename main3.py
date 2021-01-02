import numpy as np
from numpy.lib.arraysetops import unique
from numpy.lib.shape_base import split
import pandas as pd

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
# προκειμένου να φιλτραριστούνε και να χρησιμοποιηθούνε τα συγκεκριμένα δεδομένα. Συν τα CSV files με διαφορετικά δεδομένα
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

# Αποθηκεύουμε τα είδη ταινιών (ξεχωριστά μεταξύ τους αλλά όχι μοναδικά) στο df Dataframe και στο sorted_genres csv
df = pd.DataFrame(sorted_genres, columns=["Sorted Genres:"])
df.to_csv("4_sorted_genres.csv", index=False)

# Ορίζουμε το μέγιστο αριθμό στηλών/ειδών που μπορεί να έχει η κάθε ταινία, διαβάζουμε όλες τις γραμές και κρατάμε σε dataframe τα είδη
columns=["col1", "col2", "col3", "col4", "col5", "col6"]
unique_genres = pd.read_csv("4_sorted_genres.csv",  sep='|', skiprows=[0], names=columns, encoding="ISO-8859-1", low_memory=False, header=None)

# Έλεγχος του πίνακα με τα είδη ανά ταινία
# print(unique_genres.head())

# Δημιουργούμε λίστα για τα δεδομένα αυτά και δημιουργούμε νέο dataframe unique_genres2 που θα κρατάει από τη λίστα τις μοναδικές λέξεις/είδη
list(set(unique_genres.col1))
unique_genres2 = list(unique_genres['col1'].unique())

# List of the unique genres
# print(unique_genres2)

# Μεταφέρουμε τη νέα λίστα μας στο df (ή σε df2 αν θέλουμε) και αποθηκεύουμε σε csv
df = pd.DataFrame(unique_genres2, columns=["Unique Genres:"])
df.to_csv("5_unique_genres.csv", index=False)

# 3. Αντιστοίχιση των ειδών σε αριθμούς και δημιουργία σχέσεων μεταξύ τους
# Pandas insert in dataframe -> https://www.geeksforgeeks.org/python-pandas-dataframe-insert/
# Αρχικά θέτουμε έναν αριθμό σε κάθε είδος και ελέγχουμε το αποτέλεσμα

df.insert(0, 'No.', df.index, allow_duplicates = False)
print(df)