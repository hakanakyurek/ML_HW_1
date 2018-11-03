import csv
import numpy as np
import pandas as pd
import time
import math

def PandaReader(ratings, users, books):

    ratingData = pd.read_csv(ratings, sep=';', encoding='latin-1', error_bad_lines=False, warn_bad_lines=False)
    userData = pd.read_csv(users, sep=';', encoding='latin-1', error_bad_lines=False, warn_bad_lines=False)
    bookData = pd.read_csv(books, sep=';', encoding='latin-1', error_bad_lines=False, warn_bad_lines=False)

    combinedData = userData.merge(ratingData, right_on= 'User-ID', left_on= 'User-ID', how='inner')
    combinedData = bookData.merge(combinedData, right_on= 'ISBN', left_on= 'ISBN', how='inner')

    filteredData = combinedData[combinedData['Location'].str.contains("usa|canada")]

    filteredData = filteredData[['ISBN', 'User-ID', 'Book-Rating']]

    return filteredData


def ReadTest(testcsv):

    testData = pd.read_csv(testcsv, sep=',', encoding='latin-1', error_bad_lines=False, warn_bad_lines=False)
    testData = testData[['ISBN', 'User-ID', 'Book-Rating']]

    return testData