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
    values = {}

    for test in testData:

        simData[test] = defaultdict(float)
        values[test] = [0, 0]

        for book in testData[test].keys():


            for user in bookRatingMap[book]:

                if (user in trainingData):

                    if (function == 'Cos'):

                        simData[test][user] += np.multiply(bookRatingMap[book][user], testData[test][book])

                    elif (function == 'Cor'):#kitap
                        bookavg = Average(bookRatingMap[book].values())

                        x = testData[test][book] - bookavg
                        y = bookRatingMap[book][user] - bookavg

                        values[test] = [values[test][0] + (x ** 2), values[test][1] + (y ** 2)]

                        simData[test][user] += x * y

                    elif (function == 'ACos'):#user
                        #TODO: ortak kitaplara verilen oyların ortalaması alınacak
                        testavg = Average(testData[test].values())
                        usravg = Average(trainingData[user].values())

                        x = testData[test][book] - testavg
                        y = bookRatingMap[book][user] - usravg

                        values[test] = [values[test][0] + (x ** 2), values[test][1] + (y ** 2)]

                        simData[test][user] += x * y

    if (function == 'Cos'):

        CosineSimiarity(simData, trainingData, testData)

    else:

        COR_ACOS_Similarity(simData, trainingData, testData, values)

    PredictRating(simData, userRatingMap, bookRatingMap, testData, k, threshold, weighted)

    mae = MAE(simData)

    return simData, mae


def TestData(userRatingMap, userRatingTestMap, bookRatingMap, function='Cos', k=1, threshold = 0, weighted = False):

    trainingData = {k: userRatingMap[k] for k in list(userRatingMap)}
    testData = {k: userRatingTestMap[k] for k in list(userRatingTestMap)}

    simData, mae = UseData(userRatingMap, testData, trainingData, bookRatingMap, function, k, threshold, weighted)

    return simData, mae


def ValidateData(userRatingMap, bookRatingMap, function = "Cos", split = 1, k = 1, threshold = 0, weighted = False):

    trainingData = {k: userRatingMap[k] for k in list(userRatingMap)[split:]}
    testData = {k: userRatingMap[k] for k in list(userRatingMap)[:split]}

    simData, mae = UseData(userRatingMap, testData, trainingData, bookRatingMap, function, k, threshold, weighted)

    return simData, mae


def PredictRating(simData, userRatingMap, bookRatingMap, testData, k, threshold, weighted = False):

    tempsimData = simData.copy()
    count = 0

    for user in testData:

        mostSimilars = Counter(simData[user]).most_common(k)
        simData[user] = {}

        for book in testData[user]:
            ratingSum = 0

            if (len(bookRatingMap[book]) >= threshold):

                for simUser in mostSimilars:

                    if(book in userRatingMap[simUser[0]]):

                        if not weighted:

                            ratingSum += userRatingMap[simUser[0]][book]

                        else:

                            ratingSum += userRatingMap[simUser[0]][book] * (1 / ((1 - tempsimData[user][simUser[0]]) ** 2 + 0.001))

                simData[user][book] = ratingSum / k
                prediction = simData[user][book]
                prediction = 10 if prediction > 10 else prediction

                simData[user][book] = prediction - testData[user][book]
                count += 1
    print("prediction number: ", count)


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

    mae = sum(temp.values()) / errorCount
    #print(temp)
    print("MAE = ", mae)
    return mae


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


def COR_ACOS_Similarity(simData, trainingData, testData, values):
    for user in simData:

        values[user] = [math.sqrt(values[user][0]), math.sqrt(values[user][1])]

        for sim in simData[user]:

            if(values[user].count(0) == 0):
                simData[user][sim] /= (values[user][0] * values[user][1])
            else:
                simData[user][sim] = 0.0


def Average(lst):
    return sum(lst) / len(lst)
