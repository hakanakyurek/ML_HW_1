import csv
import numpy as np

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

    print(len(books), len(users))

    tempBooks = set()
    tempUsers = set()

    #TODO: Extra filtering uygulanabilir.
    for rat in ratings:
        tempBooks.add(rat[1])

        if('"' + rat[0] + '"' in users.keys()):
            tempUsers.add(rat[0])

    print(len(tempBooks), len(tempUsers))

    array = np.zeros((len(tempUsers), len(tempBooks)))

    tempUsers = list(tempUsers)
    tempBooks = list(tempBooks)

    for rat in ratings:

        try:
            array[tempUsers.index(rat[0])][tempBooks.index(rat[1])] = rat[2]
        except:
            continue


    print(array)

