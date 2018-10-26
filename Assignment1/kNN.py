import numpy as np
from numpy import dot
from numpy.linalg import norm
from pympler import asizeof
import pandas as pd
import math

def ConstructTrainModel(filteredData):

    filteredData.set_index("User-ID", drop=False, inplace=True)
    userRatingMap = filteredData.to_dict(orient="index")

    filteredData.set_index("ISBN", drop=True, inplace=True)
    bookRatingMap = filteredData.to_dict(orient="index")

    for rat in userRatingMap.values():
        del rat['User-ID']
        print(rat)

