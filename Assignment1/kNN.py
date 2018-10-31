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


def TestData(userRatingMap, userRatingTestMap, bookRatingMap, function='Cos', k=1, threshold = 0, weighted = False):

    trainingData = {k: userRatingMap[k] for k in list(userRatingMap)}
    testData = {k: userRatingTestMap[k] for k in list(userRatingTestMap)}

    simData = {}
    if (function == 'Cos'):
        for test in testData:

            # print("test ", test, testData[test].keys(), testData[test].values())
            simData[test] = defaultdict(float)

            for book in testData[test].keys():

                # print("book ", book)
                # simData[test][book] = {}#defaultdict(float)

                for user in bookRatingMap[book]:

                    if (user in trainingData):
                        # print("user ", user,  bookRatingMap[book][user])
                        simData[test][user] += np.multiply(bookRatingMap[book][user], testData[test][book])  # +=

        CosineSimiarity(simData, trainingData, testData)

        PredictRating(simData, userRatingMap, bookRatingMap, testData, k, threshold, weighted)

        MAE(simData)

    return simData


def ValidateData(userRatingMap, bookRatingMap, function = "Cos", split = 1, k = 1, threshold = 0, weighted = False):

    trainingData = {k: userRatingMap[k] for k in list(userRatingMap)[split:]}
    testData = {k: userRatingMap[k] for k in list(userRatingMap)[:split]}

    simData = {}
    corValue = {}
    aCosValue = {}

    for test in testData:

        #print("test ", test, testData[test].keys(), testData[test].values())
        simData[test] = defaultdict(float)
        aCosValue[test] = {}

        for book in testData[test].keys():

            #print("book ", book)
            #simData[test][book] = {}#defaultdict(float)

            for user in bookRatingMap[book]:

                if(user in trainingData):

                    if (function == 'Cos'):
                        #print("user ", user,  bookRatingMap[book][user])
                        simData[test][user] += np.multiply(bookRatingMap[book][user], testData[test][book])#+=

                    elif (function == 'Cor'):
                        bookavg = Average(bookRatingMap[book].values())

                        x = testData[test][book] - bookavg
                        y = bookRatingMap[book][user] - bookavg

                        #print("user ", user,  bookRatingMap[book][user])
                        simData[test][user] += x * y

                    elif (function == 'ACos'):
                        testavg = Average(testData[test].values())
                        usravg = Average(trainingData[user].values())

                        x = testData[test][book] - testavg
                        y = bookRatingMap[book][user] - usravg


                        aCosValue[test][user] = [x, y]

                        #print("user ", user,  bookRatingMap[book][user])
                        simData[test][user] += x * y


    if(function == 'Cos'):

        CosineSimiarity(simData, trainingData, testData)

    elif (function == 'Cor'):

        CorrolationSimilarity(simData, trainingData, testData)

    elif (function == 'ACos'):

        AdjCosineSimilarity(simData, trainingData, testData, aCosValue)

    PredictRating(simData, userRatingMap, bookRatingMap, testData, k, threshold, weighted)

    MAE(simData)

    return simData


def PredictRating(simData, userRatingMap, bookRatingMap, testData, k, threshold, weighted = False):

    tempsimData = simData.copy()
    count = 0

    for user in testData:

        mostSimilars = Counter(simData[user]).most_common(k)
        simData[user] = {}

        for book in testData[user]:
            #print(book)

            ratingSum = 0

            if (len(bookRatingMap[book]) >= threshold):

                for simUser in mostSimilars:

                    if(book in userRatingMap[simUser[0]]):

                        if not weighted:

                            ratingSum += userRatingMap[simUser[0]][book]

                        else:

                            ratingSum += userRatingMap[simUser[0]][book] * (1 / ((1 - tempsimData[user][simUser[0]]) ** 2 + 0.001))

                    #print(ratingSum)

                simData[user][book] = ratingSum / k
                prediction = simData[user][book]
                prediction = 10 if prediction > 10 else prediction

                simData[user][book] = prediction - testData[user][book]
                count += 1
    print("prediction number: ", count)
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

            temp[user] = count

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


def AdjCosineSimilarity(simData, trainingData, testData, aCosValue):

    for user in simData:

        val1, val2 = 0, 0

        for sim in simData[user]:

            val1 += aCosValue[user][sim][0] * aCosValue[user][sim][0]
            val2 += aCosValue[user][sim][1] * aCosValue[user][sim][1]

            simData[user][sim] = math.sqrt(val1) * math.sqrt(val2)


def CorrolationSimilarity(simData, trainingData, testData):
    print("empty")


def Average(lst):
    return sum(lst) / len(lst)