import numpy as np
from sklearn import neighbors
import ReadData as r
import kNN as knn
import time

timer = time.time()

ratings, users, books = r.PandaReader("./data/BX-Book-Ratings-Train.csv", "./data/BX-Users.csv", "./data/BX-Books.csv")

testRatings = r.ReadTest("./data/BXBookRatingsTest.csv", users, books)

print("Read data time: ", time.time() - timer)

timer = time.time()

userRatingMap, bookRatingMap = knn.ConstructTrainModel(ratings)
userRatingTestMap, bookRatingTestMap = knn.ConstructTrainModel(testRatings)

print("matrix creation time: ", time.time() - timer)

timer = time.time()

function = 'Cor'

'''
f = open("knn" + function + "txt", "w")
f.write("k" + " threshold" + " mae" + " time" + "\n")

min = [0, 0]

for k in range(1, 50, 2):
    print("K = ", k)

    for split in range(0, 5):
        print("split = ", split)
        timer = time.time()

        sim, mae = knn.ValidateData(userRatingMap, bookRatingMap,
                                    split_1=split * int(len(userRatingMap) / 5), split_2=(split + 1) * int(len(userRatingMap) / 5),
                                    k=k, function=function, threshold=8, weighted=False)
        min[0] += mae
        min[1] += time.time() - timer

        print("Test k time: ", time.time() - timer)

    min[0] /= 5
    min[1] /= 5
    f.write("%d %d %.2f %.2f %d\n" % (k, 8, min[0], min[1], 0))
    min[0] = min[1] = 0
    
'''
sim, mae = knn.TestData(userRatingMap, userRatingTestMap, bookRatingMap, k=3, function=function, threshold=0,
                            weighted=False)
print("Validation time: ", time.time() - timer)