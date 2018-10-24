import numpy as np
from sklearn import neighbors
import ReadData as r
import kNN as knn
import time

dataSplit = 900/1000

#r.PandaReader("/home/hakanmint/Desktop/oKuL/409/ASSignment 1/Assignment1/data/BX-Book-Ratings-Train.csv","/home/hakanmint/Desktop/oKuL/409/ASSignment 1/Assignment1/data/BX-Users.csv","/home/hakanmint/Desktop/oKuL/409/ASSignment 1/Assignment1/data/BX-Books.csv")

start = time.time()
users = {}
books = {}

allData = r.ReadBookRatings("/home/hakanmint/Desktop/oKuL/409/ASSignment 1/Assignment1/data/BX-Book-Ratings-Train.csv")

r.ReadUsers("/home/hakanmint/Desktop/oKuL/409/ASSignment 1/Assignment1/data/BX-Users.csv", users)
r.ReadBooks("/home/hakanmint/Desktop/oKuL/409/ASSignment 1/Assignment1/data/BX-Books.csv", books)

end = time.time()
print("Read data time: ", end - start)


start = time.time()

tempUsers, tempBooks = r.FilterRatings(allData, users, books)

allData = np.array(allData)

mainMatrix, bookIndices = knn.ConstructTrainMatrix(tempUsers, tempBooks, allData)

testData = np.array(mainMatrix[int(dataSplit * len(mainMatrix)):])
trainData = np.array(mainMatrix[0: int(dataSplit * len(mainMatrix))])

end = time.time()
print("train data matrix time: ", end - start)


print(testData, "\n", trainData)


##
#TODO:1. Matrix oluştur, her satır bir user her stun bir kitap
#TODO:2. Satırlar arasında similarity bul.
#TODO:3. Verilen dataya en yakın k elemanı bul(similarity)
#
##
