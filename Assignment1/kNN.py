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

def ValidateData(userRatingMap, bookRatingMap, function = "Cos", split = 1, k = 1, threshold = 1):

    trainingData = {k: userRatingMap[k] for k in list(userRatingMap)[split:]}
    testData = {k: userRatingMap[k] for k in list(userRatingMap)[:split]}

    simData = {}
    if(function == 'Cos'):
        for test in testData:

            #print("test ", test, testData[test].keys(), testData[test].values())
            simData[test] = defaultdict(float)

            for book in testData[test].keys():

                #print("book ", book)
                #simData[test][book] = {}#defaultdict(float)

                for user in bookRatingMap[book]:

                    if(user in trainingData):

                        #print("user ", user,  bookRatingMap[book][user])
                        simData[test][user] += np.multiply(bookRatingMap[book][user], testData[test][book])#+=

        CosineSimiarity(simData, trainingData, testData)

        PredictRating(simData, userRatingMap, bookRatingMap, testData, k, threshold)

        MAE(simData)

    elif(function == 'ACos'):

        print((function))

    elif(function == 'Cor'):

        print(function)
        
    return simData

def PredictRating(simData, userRatingMap, bookRatingMap, testData, k, threshold):

    for user in testData:

        mostSimilars = Counter(simData[user]).most_common(k)
        simData[user] = {}

        for book in testData[user]:
            #print(book)

            ratingSum = 0

            if (len(bookRatingMap[book]) >= threshold):

                for simUser in mostSimilars:

                    if(book in userRatingMap[simUser[0]]):

                         ratingSum += userRatingMap[simUser[0]][book]

                    #print(ratingSum)

                simData[user][book] = ratingSum / k
                simData[user][book] = simData[user][book] - userRatingMap[user][book]


    #print(simData)

def MAE(simData):

    temp = {}
    errorCount = 0

    for user in simData:

        count = 0


        for book in simData[user]:

            count += math.fabs(simData[user][book])
            errorCount += 1

        if(len(simData[user]) != 0):

            temp[user] = count / len(simData[user])
        count = 0



    #print(temp)
    print("MAE = ", sum(temp.values()) / errorCount)

def CosineSimiarity(simData, trainingData, testData):
    for user in simData:

        testNorm = np.array(list(testData[user].values()))
        testNorm = np.linalg.norm(testNorm)

        for sim in simData[user]:

            simNorm = np.array(list(trainingData[sim].values()))
            simNorm = np.linalg.norm(simNorm)

            if(simNorm != 0 and testNorm != 0):
                simData[user][sim] /= np.multiply(simNorm, testNorm)
            else:
                simData[user][sim] = 0.0

            #print(sim, simData[test][sim])


def AdjCosineSimilarity():
    print("empty")

def CorrolationSimilarity():
    print("empty")