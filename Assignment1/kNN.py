from scipy import spatial


def GetNeighbours(dataset, testData, k):

    count = 0

    neightbours = []

    for data in dataset:
        sim = CosineBasedSimilarity(data, testData)
        for i in range (0, k):

            if(len(neightbours) == 0):
                neightbours.append([data, sim])
                break

            elif len(neightbours) < k:
                neightbours.append([data, sim])
                break

            elif(sim > neightbours[i][1]):
                neightbours.pop(i)
                neightbours.append([data, sim])
                break

    return neightbours

def CosineBasedSimilarity(var1, var2):
    return 1 - spatial.distance.cosine(var1, var2)

