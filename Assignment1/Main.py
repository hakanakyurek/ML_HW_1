import numpy as np
from sklearn import neighbors
import ReadData as r
import kNN as knn
import time

timer = time.time()

testRatings = r.ReadTest("./data/BXBookRatingsTest.csv")

ratings = r.PandaReader("./data/BX-Book-Ratings-Train.csv", "./data/BX-Users.csv", "./data/BX-Books.csv")


print("Read data time: ", time.time() - timer)

timer = time.time()

userRatingMap, bookRatingMap = knn.ConstructTrainModel(ratings)
userRatingTestMap, bookRatingTestMap = knn.ConstructTrainModel(testRatings)

print("matrix creation time: ", time.time() - timer)

timer = time.time()

function = 'Cor'
f = open("knn" + function + "txt", "a")

min = [1000, 0]
for split in range(0, 5):
    print("split = ", split)
    for k in range(1, 50, 2):
        timer = time.time()
        print("K = ", k)
        sim, mae = knn.ValidateData(userRatingMap, bookRatingMap,
                                    split_1=split * int(len(userRatingMap) / 5), split_2=(split + 1) * int(len(userRatingMap) / 5),
                                    k=k, function=function, threshold=8, weighted=False)
        if mae < min[0]:
            min[0] = mae
            min[1] = k

        print("Test k time: ", time.time() - timer)
        print(min)
'''
for k in range(1, 50, 2):
    timer = time.time()
    print("K = ", k)
    sim = knn.TestData(userRatingMap, userRatingTestMap, bookRatingMap, k=k, function=function, threshold=0, weighted=False)
    print("Test k time: ", time.time() - timer)
#print("sim dict: ", sim)
'''
#sim = knn.TestData(userRatingMap, userRatingTestMap, bookRatingMap, k=5, function=function, threshold=8, weighted=False)
print("Validation time: ", time.time() - timer)