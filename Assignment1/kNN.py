import numpy as np
from numpy import dot
from numpy.linalg import norm
from pympler import asizeof
import pandas as pd
import math

def ConstructTrainMatrix(tempUsers, tempBooks, ratings):

    array = np.zeros((len(tempUsers), len(tempBooks)), dtype=int)

    tempUsers = list(tempUsers)
    tempBooks = list(tempBooks)

    bookIndices = {}
    userIndices = {}
    userRatingMap = {}
    bookRatingMap = {}

    for i in range(0, len(tempBooks)):
        bookIndices[tempBooks[i]] = i

    for i in range(0, len(tempUsers)):
        userIndices[tempUsers[i]] = i

    for rat in ratings:
        try:
            array[userIndices[rat[0]]][bookIndices[rat[1]]] = rat[2]
            '''
            userRatingMap.setdefault(rat[0], [])
            userRatingMap[rat[0]].append([rat[1], rat[2]])

            bookRatingMap.setdefault(rat[1], [])
            bookRatingMap[rat[1]].append([rat[0], rat[2]])
            '''
        except KeyError:
            continue

    print("Train matrix size: ", asizeof.asizeof(array))


    return array, bookIndices

def ConstructTrainModel(filteredData):

    filteredData.set_index("User-ID", drop=False, inplace=True)
    userRatingMap = filteredData.to_dict(orient="index")

    filteredData.set_index("ISBN", drop=True, inplace=True)
    bookRatingMap = filteredData.to_dict(orient="index")

    for rat in userRatingMap.values():
        del rat['User-ID']
        print(rat)

#Get from Matrix
def GetNeighbours(dataset, testData, k):

    neightbours = []

    for test in testData:

        for data in dataset:
            sim = CosineBasedSimilarity(data, test)

            neightbours.append([data, sim])



    return neightbours
#Using Matrix
def CosineBasedSimilarity(var1, var2):
    #return 1 - spatial.distance.cosine(var1, var2)
    x = norm(var1)
    y = norm(var2)

    if(x == 0 or y == 0):
        return 0

    result = dot(var1, var2) / x * y
    #print("result: ", result)
    return result

