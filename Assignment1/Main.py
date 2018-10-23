import numpy as np
from sklearn import neighbors
import ReadData as r
import kNN as knn
import time

dataset = [[0, 2, 3, 1], [6, 2, 3, 6], [6, 6, 6, 6], [2, 5, 1, 0], [0, 2, 2, 4]]
testdata = [0, 2, 3, 0]

a = knn.GetNeighbours(dataset, testdata, 2)

print(a)

#r.PandaReader("/home/hakanmint/Desktop/oKuL/409/ASSignment 1/Assignment1/data/BX-Book-Ratings-Train.csv","/home/hakanmint/Desktop/oKuL/409/ASSignment 1/Assignment1/data/BX-Users.csv","/home/hakanmint/Desktop/oKuL/409/ASSignment 1/Assignment1/data/BX-Books.csv")

start = time.time()
users = {}
books = {}

bookRatings = r.ReadBookRatings("/home/hakanmint/Desktop/oKuL/409/ASSignment 1/Assignment1/data/BX-Book-Ratings-Train.csv")

r.ReadUsers("/home/hakanmint/Desktop/oKuL/409/ASSignment 1/Assignment1/data/BX-Users.csv", users)
r.ReadBooks("/home/hakanmint/Desktop/oKuL/409/ASSignment 1/Assignment1/data/BX-Books.csv", books)

bookRatings = np.array(bookRatings[1:])


end = time.time()
print(end - start)
r.FilterRatings(bookRatings, users, books)
end = time.time()
print(end - start)
##
#TODO:1. Matrix oluştur, her satır bir user her stun bir kitap
#TODO:2. Satırlar arasında similarity bul.
#TODO:3. Verilen dataya en yakın k elemanı bul(similarity)
#
##
