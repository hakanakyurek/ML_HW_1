import csv


def ReadUsers(usersFile, users):

    with open(usersFile, newline='', encoding='latin-1') as csvfile:

        reader = csv.DictReader(csvfile, delimiter=';', quotechar='|')


        for row in reader:

            users[row['"User-ID"']] = row['"Location"'], row['"Age"']

def ReadBooks(booksFile, books):

    with open(booksFile, newline='', encoding='latin-1') as csvfile:

        reader = csv.DictReader(csvfile, delimiter=';', quotechar='|')



        for row in reader:

            books[row['"ISBN"']] = row['"Book-Title"'], row['"Book-Author"'], \
                                   row['"Year-Of-Publication"'], row['"Publisher"']


def ReadBookRatings(bookRatingsFile, bookRatings):

    with open(bookRatingsFile, newline='', encoding='latin-1') as csvfile:

        reader = csv.DictReader(csvfile, delimiter=';', quotechar='|')

        for row in reader:

            bookRatings[row['User-ID']] = row['ISBN'], row['Book-Rating']