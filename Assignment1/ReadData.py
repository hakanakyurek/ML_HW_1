import csv
import numpy as np
import pandas as pd
import time
import math

def PandaReader(ratings, users, books):

    ratingData = pd.read_csv(ratings, sep=';', encoding='latin-1', error_bad_lines=False)
    userData = pd.read_csv(users, sep=';', encoding='latin-1', error_bad_lines=False)
    bookData = pd.read_csv(books, sep=';', encoding='latin-1', error_bad_lines=False)

    combinedData = userData.merge(ratingData, right_on= 'User-ID', left_on= 'User-ID', how='inner')
    combinedData = bookData.merge(combinedData, right_on= 'ISBN', left_on= 'ISBN', how='inner')

    filteredData = combinedData[combinedData['Location'].str.contains("usa|canada")]

    filteredData = filteredData[['ISBN', 'User-ID', 'Book-Rating']]

    return filteredData



def ReadUsers(usersFile, users, dataCount = -1):

    with open(usersFile, newline='', encoding='latin-1') as csvfile:

        reader = csv.DictReader(csvfile, delimiter=';', quotechar='|')

        count = 0

        for row in reader:

            if count > dataCount and dataCount is not -1:
                break

            if "usa" in row['"Location"'] or "canada" in row['"Location"']:

                users[row['"User-ID"']] = row['"Location"'], row['"Age"']

            count += 1

def ReadBooks(booksFile, books, dataCount = -1):

    with open(booksFile, newline='', encoding='latin-1') as csvfile:

        reader = csv.DictReader(csvfile, delimiter=';', quotechar='|')

        count = 0

        for row in reader:

            if count > dataCount and dataCount is not -1:
                break

            books[row['"ISBN"']] = row['"Book-Title"'], row['"Book-Author"'], \
                                   row['"Year-Of-Publication"'], row['"Publisher"']

            count += 1

def ReadBookRatings(bookRatingsFile, dataCount = 0):

    with open(bookRatingsFile, newline='', encoding='latin-1') as csvfile:

        reader = csv.reader(csvfile, delimiter=';', quotechar='|')
        reader = list(reader)
        reader = reader[:dataCount or None]
        return reader

def FilterRatings(ratings, users, books):

    print(len(books), len(users), len(ratings))

    tempBooks = set()
    tempUsers = set()
    tempRatings = []
    count = 0

    for rat in ratings:

        if ('"' + rat[0] + '"' in users.keys()):

            if ('"' + rat[1] + '"' in books.keys()):
                if(rat[2] != '0'):
                    tempBooks.add(rat[1])
                    tempUsers.add(rat[0])
                    tempRatings.append(rat)

                    count += 1

    print("Rating number: ", len(tempRatings))

    print("Unique books, users in training data: ", len(tempBooks), len(tempUsers))

    ratings = tempRatings

    return tempUsers, tempBooks
