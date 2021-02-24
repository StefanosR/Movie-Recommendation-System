# Decision-Theory (Movie Recommendation System)

Ο σκοπός αυτής της άσκησης είναι η δημιουργία ενός συστήματος που να προτείνει ταινίες στους χρήστες σύμφωνα με τις ταινίες που έχουν
παρακολουθήσει και τις αξιολογήσεις που έχουν κάνει. Επιπλέον γίνεται ανάλυση δεδομένων από την οποία θα προκύψει εξαγωγή πληροφοριών 
και οπτικοποίηση αυτών.

Αρχικά Datasets: users.csv, movies.csv, ratings.csv

Τα νέα αρχεία dataset fixed_ratings, fixed_users, fixed_movies κατασκευάστηκαν για την εξυπηρέτηση της ανάλυσης δεδομένων που γίνεται στο showtime.py

Περιγραφή αρχείων του Project:

user_based_collaborative_filtering -> σύστημα πρότασης ταινιών (2 σεναρίων)

demographics -> επεξεργασία δεδομένων των datasets για εξαγωγή πληροφοριών

showtime -> ανάλυση δεδομένων για οπτικοποίηση δεδομένων και παραγωγή στατιστικών

Προτεινόμενη σειρά εκτέλεσης: 

1. user_based_colabborative_filtering
2. demographics
3. showtime

Υπάρχει εκτενής σχολιασμός σε όλα τα αρχεία κώδικα. 

Βιβλιοθήκες που χρησιμοποιούνται στο Project:

sys
csv
random
warnings
pandas 
numpy 
pandas.io.parsers, FixedWidthReader
scipy, csr_matrix
matplotlib.pyplot
seaborn
numpy.lib.arraysetops, unique
numpy.lib.shape_base, split
collections, Counter
operator, itemgetter
itertools, combinations
sklearn.neighbors, NearestNeighbors
sklearn.model_selection, train_test_split
