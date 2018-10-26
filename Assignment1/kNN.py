import numpy as np
from numpy import dot
from numpy.linalg import norm
from pympler import asizeof
import pandas as pd
import math
from collections import defaultdict

def ConstructTrainModel(filteredData):

    users = filteredData['User-ID'].tolist()
    books = filteredData['ISBN'].tolist()
    ratings = filteredData['Book-Rating'].tolist()

    userRatingMap = {}
    bookRatingMap = {}

    for x in range (len(users)):

        user = users[x]
        book = books[x]
        rating = ratings[x]

        if user not in userRatingMap:
            userRatingMap[user] = {}
        if book not in bookRatingMap:
            bookRatingMap[book] = {}

        userRatingMap[user][book] = rating
        bookRatingMap[book][user] = rating

    return userRatingMap, bookRatingMap

def ValidateData(userRatingMap, bookRatingMap, function = "Cos", split = 1):

    trainingData = {k: userRatingMap[k] for k in list(userRatingMap)[split:]}
    testData = {k: userRatingMap[k] for k in list(userRatingMap)[:split]}

    simData = {}

    for test in testData:

        #print("test ", test, testData[test].keys(), testData[test].values())
        simData[test] = defaultdict(float)

        for book in testData[test].keys():

            #print("book ", book)

            for user in bookRatingMap[book]:

                if(user in trainingData):

                    #print("user ", bookRatingMap[book][user])
                    simData[test][user] += np.multiply(bookRatingMap[book][user], testData[test][book])

                else:
                    print("User not found ", user)

    CosineSimiarity(simData, trainingData, testData)

    return simData

def CosineSimiarity(simData, trainingData, testData):
    for test in simData:

        testNorm = np.array(list(testData[test].values()))
        testNorm = np.linalg.norm(testNorm)

        for sim in simData[test]:

            simNorm = np.array(list(trainingData[sim].values()))
            simNorm = np.linalg.norm(simNorm)

            if(simNorm != 0):
                simData[test][sim] /= simNorm * testNorm
            else:
                simData[test][sim] = 0.0

            print(sim, simData[test][sim])


def AdjCosineSimilarity():
    print("empty")

def CorrolationSimilarity():
    print("empty")