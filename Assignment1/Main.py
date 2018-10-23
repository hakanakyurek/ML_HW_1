import numpy as np
from sklearn import neighbors
import ReadData as r
import kNN as knn
import time

dataSplit = 990/1000

dataset = [[0, 2, 3, 1], [6, 2, 3, 6], [6, 6, 6, 6], [2, 5, 1, 0], [0, 2, 2, 4]]
testdata = [0, 2, 3, 0]

a = knn.GetNeighbours(dataset, testdata, 2)

print(a)

#r.PandaReader("/home/hakanmint/Desktop/oKuL/409/ASSignment 1/Assignment1/data/BX-Book-Ratings-Train.csv","/home/hakanmint/Desktop/oKuL/409/ASSignment 1/Assignment1/data/BX-Users.csv","/home/hakanmint/Desktop/oKuL/409/ASSignment 1/Assignment1/data/BX-Books.csv")

start = time.time()
users = {}
books = {}

trainData = r.ReadBookRatings("/home/hakanmint/Desktop/oKuL/409/ASSignment 1/Assignment1/data/BX-Book-Ratings-Train.csv")

r.ReadUsers("/home/hakanmint/Desktop/oKuL/409/ASSignment 1/Assignment1/data/BX-Users.csv", users)
r.ReadBooks("/home/hakanmint/Desktop/oKuL/409/ASSignment 1/Assignment1/data/BX-Books.csv", books)

testData = np.array(trainData[int(dataSplit * len(trainData)):])
trainData = np.array(trainData[1: int(dataSplit * len(trainData))])

end = time.time()
print("Read data time: ", end - start)


start = time.time()

tempUsers, tempBooks = r.FilterRatings(trainData, users, books)
trainMatrix = knn.ConstructTrainMatrix(tempUsers, tempBooks, trainData)

testDictionary = knn.ConstructTestMatrix(testData)

end = time.time()
print("train data matrix time: ", end - start)



##
#TODO:1. Matrix oluştur, her satır bir user her stun bir kitap
#TODO:2. Satırlar arasında similarity bul.
#TODO:3. Verilen dataya en yakın k elemanı bul(similarity)
#
##
