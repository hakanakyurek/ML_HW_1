import numpy as np
from sklearn import neighbors
import ReadData as r
import kNN as knn
import time
import threading


dataSplit = 4000/12779
timer = time.time()

ratings = r.PandaReader("/home/hakanmint/Desktop/oKuL/409/ASSignment 1/Assignment1/data/BX-Book-Ratings-Train.csv",
                        "/home/hakanmint/Desktop/oKuL/409/ASSignment 1/Assignment1/data/BX-Users.csv",
                        "/home/hakanmint/Desktop/oKuL/409/ASSignment 1/Assignment1/data/BX-Books.csv")

print("Read data time: ", time.time() - timer)

timer = time.time()

knn.ConstructTrainModel(ratings)

print("matrix creation time: ", time.time() - timer)

timer = time.time()

print("knn time: ", time.time() - timer)

##
#TODO:1. Matrix oluştur, her satır bir user her stun bir kitap
#TODO:2. Satırlar arasında similarity bul.
#TODO:3. Verilen dataya en yakın k elemanı bul(similarity)
#
##
