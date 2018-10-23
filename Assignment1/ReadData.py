import csv
import numpy as np
import pandas as p
import time

def PandaReader(ratings, users, books):
    start = time.time()
    ratingData = p.read_csv(ratings, sep=';', encoding='latin-1', error_bad_lines=False)
    userData = p.read_csv(users, sep=';', encoding='latin-1', error_bad_lines=False)
    bookData = p.read_csv(books, sep=';', encoding='latin-1', error_bad_lines=False,
                          usecols= ['ISBN', 'Book-Title', 'Book-Author', 'Publisher'])

    combinedData = userData.merge(ratingData, right_on= 'User-ID', left_on= 'User-ID', how='inner')
    combinedData = bookData.merge(combinedData, right_on= 'ISBN', left_on= 'ISBN', how='inner')

    filteredData = combinedData[combinedData['Location'].str.contains("usa|canada")]

    filteredData = filteredData.sort_values(by=['Book-Rating'])



    end = time.time()
    print(end - start)

    print(filteredData)




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
        #for row in reader:

            #bookRatings[row['User-ID']] = row['ISBN'], row['Book-Rating']


def FilterRatings(ratings, users, books):

    print(len(books), len(users), len(ratings))

    tempBooks = set()
    tempUsers = set()
    count =0
    #TODO: Extra filtering uygulanabilir.
    for rat in ratings:

        if ('"' + rat[0] + '"' in users.keys()):
            tempUsers.add(rat[0])

            if ('"' + rat[1] + '"' in books.keys()):
                tempBooks.add(rat[1])

                count += 1

    print(count)

    print(len(tempBooks), len(tempUsers))

    return tempUsers, tempBooks
