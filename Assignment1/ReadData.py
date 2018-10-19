import csv

users = {}

def ReadUsers(usersFile):

    with open(usersFile, newline='', encoding='latin-1') as csvfile:

        reader = csv.DictReader(csvfile, delimiter=';', quotechar='|')



        for row in reader:

            users[row['"User-ID"']] = row['"Location"'], row['"Age"']

