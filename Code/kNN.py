import numpy as np
from collections import Counter
import math
from collections import defaultdict

errorNumber = 0


def ConstructTrainModel(filteredData):

    users = filteredData['User-ID'].tolist()
    books = filteredData['ISBN'].tolist()
    ratings = filteredData['Book-Rating'].tolist()

    userRatingMap = {}
    bookRatingMap = {}

    # Constructing both user and book dictionaries
    for x in range (len(users)):

        user = users[x]
        book = books[x]
        try:
            rating = int(ratings[x])
        except ValueError:
            continue

        if user not in userRatingMap:
            userRatingMap[user] = {}
        if book not in bookRatingMap:
            bookRatingMap[book] = {}

        userRatingMap[user][book] = rating
        bookRatingMap[book][user] = rating

    return userRatingMap, bookRatingMap


def UseData(userRatingMap, testData, trainingData, bookRatingMap, function='Cos', k=1, threshold = 0, weighted = False):

    simData = {} # holds similarities then errors.
    values = {} # holds x and y values in cor-based and adj-cos-based similarities.

    # To hold averages in order to avoid recalculating.
    userAvgDict = {}
    bookAvgDict = {}
    testUserAvgDict = {}

    for test in testData:

        simData[test] = defaultdict(float)
        values[test] = [0, 0]

        for book in testData[test]:

            # if we do not have any info on that book, no one will be similar over to test user because of it.
            if(book in bookRatingMap):

                for user in bookRatingMap[book]:

                    # if the user read the book is in training data, just in case.
                    if (user in trainingData):

                        if (function == 'Cos'):

                            if(bookRatingMap[book][user] == 0 or testData[test][book] == 0):
                                simData[test][user] += 0
                                continue

                            simData[test][user] += np.multiply(bookRatingMap[book][user], testData[test][book])

                        elif (function == 'Cor'):

                            # Cache the averages to avoid recalculating
                            if(book not in bookAvgDict):
                                bookavg = Average(bookRatingMap[book].values())
                                bookAvgDict[book] = bookavg

                            bookavg = bookAvgDict[book]

                            x = testData[test][book] - bookavg
                            y = bookRatingMap[book][user] - bookavg

                            # calculate sums inside square roots. Will be used later
                            values[test] = [values[test][0] + (x ** 2), values[test][1] + (y ** 2)]

                            simData[test][user] += x * y

                        elif (function == 'ACos'):

                            # Cache the averages to avoid recalculating
                            if(test not in testUserAvgDict):
                                testavg = Average(testData[test].values())
                                testUserAvgDict[test] = testavg

                            if(user not in userAvgDict):
                                usravg = Average(trainingData[user].values())
                                userAvgDict[user] = usravg

                            x = testData[test][book] - testUserAvgDict[test]
                            y = bookRatingMap[book][user] - userAvgDict[user]

                            # calculate sums inside square roots. Will be used later
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
    # Create test data and training data for test
    trainingData = {k: userRatingMap[k] for k in list(userRatingMap)}
    testData = {k: userRatingTestMap[k] for k in list(userRatingTestMap)}

    simData, mae = UseData(userRatingMap, testData, trainingData, bookRatingMap, function, k, threshold, weighted)

    return simData, mae


def ValidateData(userRatingMap, bookRatingMap, function = "Cos", split_1 = 1, split_2 = 1, k = 1, threshold = 0, weighted = False):
    # Create test data and training data for validation
    trainingData = {k: userRatingMap[k] for k in list(userRatingMap)[split_2:] + list(userRatingMap)[:split_1]}
    testData = {k: userRatingMap[k] for k in list(userRatingMap)[split_1:split_2]}

    simData, mae = UseData(userRatingMap, testData, trainingData, bookRatingMap, function, k, threshold, weighted)

    return simData, mae


def PredictRating(simData, userRatingMap, bookRatingMap, testData, k, threshold, weighted = False):

    tempsimData = simData.copy()
    predictCount = 0

    for user in testData:

        mostSimilars = Counter(simData[user]).most_common(k)
        simData[user] = {}
        avgUser = Average(testData[user].values())

        for book in testData[user]:
            ratingSum = 0
            # Check if the book is in training dataset
            if(book in bookRatingMap):
                # Check if the book passes the threshold given
                if (len(bookRatingMap[book]) >= threshold):

                    count = 0

                    for simUser in mostSimilars:
                        # Check if most similar users read the book or not
                        if(book in userRatingMap[simUser[0]]):

                            if not weighted:
                                # Just sum the ratings given buy most similar users
                                ratingSum += userRatingMap[simUser[0]][book]

                            else:
                                # Need to apply weight in weighted-knn
                                ratingSum += userRatingMap[simUser[0]][book] * (1 / ((1 - tempsimData[user][simUser[0]]) ** 2 + 0.001))
                    # If prediction passes 10, which happened before but i did not check recently, lower it to 10
                    prediction = ratingSum / k
                    prediction = 10 if prediction > 10 else prediction
                    # Calculate the error
                    simData[user][book] = prediction - testData[user][book]
                    predictCount += 1

                else: # in case book cannot pass threshold

                    simData[user][book] = testData[user][book] - Average(bookRatingMap[book].values())
                    predictCount += 1
            else: # in case book is not recorded in training data
                simData[user][book] = testData[user][book] - avgUser
                predictCount += 1

    global errorNumber
    errorNumber = predictCount
    print("prediction number: ", predictCount)


def MAE(simData):

    temp = defaultdict(float)
    errorCount = 0

    for user in simData:

        count = 0

        for book in simData[user]:
            # increase count n times by each books error value
            count += math.fabs(simData[user][book])
            errorCount += 1

        if(len(simData[user]) != 0):
            # store it in dictionary
            temp[user] = count

        count = 0
    # sum all count values and divide it by total number of errors
    mae = sum(temp.values()) / errorCount

    print("MAE = ", mae)
    return mae


def CosineSimiarity(simData, trainingData, testData):

    simNormDict = {}

    for user in simData:
        # Calculate test user's norm
        testNorm = np.array(list(testData[user].values()))
        testNorm = np.linalg.norm(testNorm)

        for sim in simData[user]:
            # Cache the norms of the users to avoid recalculating, actually doubles the speed in test data
            if(sim not in simNormDict):
                simNorm = np.array(list(trainingData[sim].values()))
                simNorm = np.linalg.norm(simNorm)
                simNormDict[sim] = simNorm

            simNorm = simNormDict[sim]
            # Avoid division by zero
            if(simNorm != 0 and testNorm != 0):
                simData[user][sim] /= np.multiply(simNorm, testNorm)
            else:
                simData[user][sim] = 0.0


def COR_ACOS_Similarity(simData, trainingData, testData, values):
    # The same method can be used for these two similarities
    for user in simData:
        # Square root the x and y values we summed earlier in UseData function for each user
        values[user] = [math.sqrt(values[user][0]), math.sqrt(values[user][1])]

        for sim in simData[user]:
            # If both x and y values are not 0
            if(values[user].count(0) == 0):
                simData[user][sim] /= (values[user][0] * values[user][1])
            else: # If one of them is
                simData[user][sim] = 0.0


def Average(lst):
    return sum(lst) / len(lst)
