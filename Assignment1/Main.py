import numpy as np
from sklearn import neighbors
import ReadData as r
import kNN as knn
import time
import threading


dataSplit = 4000/12779
timer = time.time()

testRatings = r.ReadTest("/home/hakanmint/Desktop/oKuL/409/ASSignment 1/Assignment1/data/Test-User_Rating0.csv")

ratings = r.PandaReader("/home/hakanmint/Desktop/oKuL/409/ASSignment 1/Assignment1/data/BX-Book-Ratings-Train.csv",
                        "/home/hakanmint/Desktop/oKuL/409/ASSignment 1/Assignment1/data/BX-Users.csv",
                        "/home/hakanmint/Desktop/oKuL/409/ASSignment 1/Assignment1/data/BX-Books.csv")


print("Read data time: ", time.time() - timer)

timer = time.time()

userRatingMap, bookRatingMap = knn.ConstructTrainModel(ratings)
userRatingTestMap, bookRatingTestMap = knn.ConstructTrainModel(testRatings)

print("matrix creation time: ", time.time() - timer)

timer = time.time()

sim = knn.ValidateData(userRatingMap, bookRatingMap, split=3000, k=5, function='Cos', threshold=24)
#sim = knn.TestData(userRatingMap, userRatingTestMap, bookRatingMap, k=9, function='Cos', threshold=50, weighted=True)
#print("sim dict: ", sim)
print("Validation time: ", time.time() - timer)

##
#TODO:1. Matrix oluştur, her satır bir user her stun bir kitap
#TODO:2. Satırlar arasında similarity bul.
#TODO:3. Verilen dataya en yakın k elemanı bul(similarity)
#
##
