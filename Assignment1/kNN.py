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


def UseData(userRatingMap, testData, trainingData, bookRatingMap, function='Cos', k=1, threshold = 0, weighted = False):

    simData = {}
    corValue = {}
    aCosValue = {}

    for test in testData:

        # print("test ", test, testData[test].keys(), testData[test].values())
        simData[test] = defaultdict(float)
        corValue[test] = [0, 0]

        for book in testData[test].keys():

            # print("book ", book)
            # simData[test][book] = {}#defaultdict(float)

            for user in bookRatingMap[book]:

                if (user in trainingData):

                    if (function == 'Cos'):
                        # print("user ", user,  bookRatingMap[book][user])
                        simData[test][user] += np.multiply(bookRatingMap[book][user], testData[test][book])  # +=

                    elif (function == 'Cor'):
                        testavg = Average(testData[test].values())
                        usravg = Average(trainingData[user].values())

                        x = testData[test][book] - testavg
                        y = bookRatingMap[book][user] - usravg

                        corValue[test] = [corValue[test][0] + (x ** 2), corValue[test][1] + (y ** 2)]

                        # print("user ", user,  bookRatingMap[book][user])
                        simData[test][user] += x * y

                    elif (function == 'ACos'):
                        bookavg = Average(bookRatingMap[book].values())

                        x = testData[test][book] - bookavg
                        y = bookRatingMap[book][user] - bookavg

                        # print("user ", user,  bookRatingMap[book][user])
                        simData[test][user] += x * y


    if (function == 'Cos'):

        CosineSimiarity(simData, trainingData, testData)

    elif (function == 'Cor'):

        CorrolationSimilarity(simData, trainingData, testData, corValue)

    elif (function == 'ACos'):

        AdjCosineSimilarity(simData, trainingData, testData, aCosValue)

    PredictRating(simData, userRatingMap, bookRatingMap, testData, k, threshold, weighted)

    MAE(simData)

    return simData


def TestData(userRatingMap, userRatingTestMap, bookRatingMap, function='Cos', k=1, threshold = 0, weighted = False):

    trainingData = {k: userRatingMap[k] for k in list(userRatingMap)}
    testData = {k: userRatingTestMap[k] for k in list(userRatingTestMap)}

    simData = UseData(userRatingMap, testData, trainingData, bookRatingMap, function, k, threshold, weighted)

    return simData


def ValidateData(userRatingMap, bookRatingMap, function = "Cos", split = 1, k = 1, threshold = 0, weighted = False):

    trainingData = {k: userRatingMap[k] for k in list(userRatingMap)[split:]}
    testData = {k: userRatingMap[k] for k in list(userRatingMap)[:split]}

    simData = UseData(userRatingMap, testData, trainingData, bookRatingMap, function, k, threshold, weighted)

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
    print("empty")

def CorrolationSimilarity(simData, trainingData, testData, corValue):

    for user in simData:

        corValue[user] = [math.sqrt(corValue[user][0]), math.sqrt(corValue[user][1])]

        for sim in simData[user]:

            if(corValue[user].count(0) == 0):
                simData[user][sim] /= corValue[user][0] * corValue[user][1]
            else:
                simData[user][sim] = 0.0


def Average(lst):
    return sum(lst) / len(lst)