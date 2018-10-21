from scipy import spatial


def GetNeighbours():
    print("empty")
    #TODO: empty


def CosineBasedSimilarity(var1, var2):
    return 1 - spatial.distance.cosine(var1, var2)


