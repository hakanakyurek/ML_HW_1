import csv


def ReadUsers(usersFile, users):

    with open(usersFile, newline='', encoding='latin-1') as csvfile:

        reader = csv.DictReader(csvfile, delimiter=';', quotechar='|')


        for row in reader:

            if "usa" in row['"Location"'] or "canada" in row['"Location"']:

                users[row['"User-ID"']] = row['"Location"'], row['"Age"']

def ReadBooks(booksFile, books):

    with open(booksFile, newline='', encoding='latin-1') as csvfile:

        reader = csv.DictReader(csvfile, delimiter=';', quotechar='|')



        for row in reader:

            books[row['"ISBN"']] = row['"Book-Title"'], row['"Book-Author"'], \
                                   row['"Year-Of-Publication"'], row['"Publisher"']


def ReadBookRatings(bookRatingsFile):

    with open(bookRatingsFile, newline='', encoding='latin-1') as csvfile:

        reader = csv.reader(csvfile, delimiter=';', quotechar='|')

        return list(reader)
        #for row in reader:

            #bookRatings[row['User-ID']] = row['ISBN'], row['Book-Rating']