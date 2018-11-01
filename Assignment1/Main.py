import numpy as np
from sklearn import neighbors
import ReadData as r
import kNN as knn
import time

timer = time.time()

testRatings = r.ReadTest("./data/Test-User_Rating0.csv")

ratings = r.PandaReader("./data/BX-Book-Ratings-Train.csv", "./data/BX-Users.csv", "./data/BX-Books.csv")


print("Read data time: ", time.time() - timer)

timer = time.time()

userRatingMap, bookRatingMap = knn.ConstructTrainModel(ratings)
userRatingTestMap, bookRatingTestMap = knn.ConstructTrainModel(testRatings)

print("matrix creation time: ", time.time() - timer)

timer = time.time()

function = 'ACos'
f = open("knn" + function + "txt", "a")

#sim = knn.ValidateData(userRatingMap, bookRatingMap, split=2000, k=3, function=function, threshold=5, weighted=False)
for k in range(1, 50, 2):
    timer = time.time()
    print("K = ", k)
    sim = knn.TestData(userRatingMap, userRatingTestMap, bookRatingMap, k=k, function=function, threshold=50, weighted=True)
    print("Test k time: ", time.time() - timer)
#print("sim dict: ", sim)
print("Validation time: ", time.time() - timer)

##
#TODO:1. Matrix oluştur, her satır bir user her stun bir kitap
#TODO:2. Satırlar arasında similarity bul.
#TODO:3. Verilen dataya en yakın k elemanı bul(similarity)
#
##
