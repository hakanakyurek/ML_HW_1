import numpy as np
from sklearn import neighbors, datasets
import ReadData as r

print(np.__version__)

users = {}
books = {}
bookRatings = {}

r.ReadBooks("/home/hakanmint/Desktop/oKuL/409/ASSignment 1/Assignment1/data/BX-Books.csv", books)

print(books)