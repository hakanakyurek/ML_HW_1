import numpy as np
from numpy import dot
from numpy.linalg import norm
from pympler import asizeof


def ConstructTrainMatrix(tempUsers, tempBooks, ratings):

    array = np.zeros((len(tempUsers), len(tempBooks)), dtype=int)

    tempUsers = list(tempUsers)
    tempBooks = list(tempBooks)

    bookIndices = {}
    userIndices = {}

    for i in range(0, len(tempBooks)):
        bookIndices[tempBooks[i]] = i

    for i in range(0, len(tempUsers)):
        userIndices[tempUsers[i]] = i

    #    tempBooks = {ele:tempBooks.index(ele) for ele in tempBooks}
    #    tempUsers = {ele:tempUsers.index(ele) for ele in tempUsers}

    for rat in ratings:
        try:
            array[userIndices[rat[0]]][bookIndices[rat[1]]] = rat[2]

        except KeyError:
            continue

    print("Train matrix size: ", asizeof.asizeof(array))

    return array, bookIndices

def GetNeighbours(dataset, testData, k):

    count = 0

    neightbours = []

    for data in dataset:
        sim = CosineBasedSimilarity(data, testData)
        for i in range (0, k):

            if(len(neightbours) == 0):
                neightbours.append([data, sim])
                break

            elif len(neightbours) < k:
                neightbours.append([data, sim])
                break

            elif(sim > neightbours[i][1]):
                neightbours.pop(i)
                neightbours.append([data, sim])
                break

    return neightbours

def CosineBasedSimilarity(var1, var2):
    #return 1 - spatial.distance.cosine(var1, var2)
    return dot(var1, var2) / (norm(var1) * norm(var2))

