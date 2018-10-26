import numpy as np
from numpy import dot
from numpy.linalg import norm
from pympler import asizeof
import pandas as pd
import math

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


def CosineSimiarity():
    print("empty")

def AdjCosineSimilarity():
    print("empty")

def CorrolationSimilarity():
    print("empty")