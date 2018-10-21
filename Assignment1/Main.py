import numpy as np
from sklearn import neighbors, datasets
import ReadData as r

print(np.__version__)

users = {}
books = {}

bookRatings = r.ReadBookRatings("/home/hakanmint/Desktop/oKuL/409/ASSignment 1/Assignment1/data/BX-Book-Ratings-Train.csv", 5000)

r.ReadUsers("/home/hakanmint/Desktop/oKuL/409/ASSignment 1/Assignment1/data/BX-Users.csv", users)
r.ReadBooks("/home/hakanmint/Desktop/oKuL/409/ASSignment 1/Assignment1/data/BX-Books.csv", books)

bookRatings = np.array(bookRatings[1:])

r.FilterRatings(bookRatings, users, books)

##
#TODO:1. Matrix oluştur, her satır bir user her stun bir kitap
#TODO:2. Satırlar arasında similarity bul.
#TODO:3. Verilen dataya en yakın k elemanı bul(similarity)
#
##
