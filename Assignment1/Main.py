import numpy as np
from sklearn import neighbors, datasets
import ReadData as r

print(np.__version__)

users = {}
books = {}

r.ReadUsers("/home/hakanmint/Desktop/oKuL/409/ASSignment 1/Assignment1/data/BX-Users.csv", users)
#bookRatings = r.ReadBookRatings("/home/hakanmint/Desktop/oKuL/409/ASSignment 1/Assignment1/data/BX-Book-Ratings-Train.csv")

print(users.__len__())

#bookRatings = np.array(bookRatings[1:])

#print(bookRatings)

##
#1. Matrix oluştur, her satır bir user her stun bir kitap
#2. Satırlar arasında similarity bul.
#
#
##
