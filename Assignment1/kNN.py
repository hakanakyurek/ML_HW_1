import numpy as np
from heapq import nlargest
from collections import Counter
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

def ValidateData(userRatingMap, bookRatingMap, function = "Cos", split = 1, k = 1):

    trainingData = {k: userRatingMap[k] for k in list(userRatingMap)[split:]}
    testData = {k: userRatingMap[k] for k in list(userRatingMap)[:split]}

    simData = {}

    for test in testData:

        #print("test ", test, testData[test].keys(), testData[test].values())
        simData[test] = {}#defaultdict(float)

        for book in testData[test].keys():

            #print("book ", book)
            simData[test][book] = {}#defaultdict(float)

            for user in bookRatingMap[book]:

                if(user in trainingData):

                    #print("user ", user,  bookRatingMap[book][user])
                    simData[test][book][user] = np.multiply(bookRatingMap[book][user], testData[test][book])#+=

                else:
                   print("User not found ", user)

    CosineSimiarity(simData, trainingData, testData)
    FindK(simData, userRatingMap, bookRatingMap, k)
    return simData

def FindK(simData, userRatingMap, bookRatingMap, k):

    for user in simData:

        for book in simData[user]:
            print(book)
            mostSimilars = Counter(simData[user][book]).most_common(k)
            ratingSum = 0

            for simUser in mostSimilars:
                ratingSum += userRatingMap[simUser[0]][book]

                print(ratingSum)

            simData[user][book] = ratingSum / k
            simData[user][book] = userRatingMap[user][book] - simData[user][book]
    print(simData)


def CosineSimiarity(simData, trainingData, testData):
    for user in simData:

        testNorm = np.array(list(testData[user].values()))
        testNorm = np.linalg.norm(testNorm)
        for book in simData[user]:

            for sim in simData[user][book]:

                simNorm = np.array(list(trainingData[sim].values()))
                simNorm = np.linalg.norm(simNorm)

                if(simNorm != 0 and testNorm != 0):
                    simData[user][book][sim] /= np.multiply(simNorm, testNorm)
                else:
                    simData[user][book][sim] = 0.0

                #print(sim, simData[test][sim])


def AdjCosineSimilarity():
    print("empty")

def CorrolationSimilarity():
    print("empty")