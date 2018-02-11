#! /usr/bin/env python
#coding: utf-8

import numpy as np
import pandas as ps
from munkres import Munkres

def get_groundTruth(groundTruthPath):
    groundTruth_df = pd.read_csv(groundTruthPath)
    groundTruth_arr = np.array(groundTruth_df)

    return groundTruth_arr


def accuracy(estCentroid_arr, groundTruth_arr, distTreshold):
# distance of estimation and groundtruth is less than distThreshold --> True
    def get_norm(x, y):
        return (x**2 + y**2)**0.5

    # make distance matrix
    n = max(len(estCentroid_arr), len(groundTruth_arr))
    distMatrix = np.zeros((n,n))
    for i in range(len(estCentroid_arr)):
        for j in range(len(groundTruth_arr)):
            distX = estCentroid_arr[i][0] - groundTruth_arr[j][0]
            distY = estCentroid_arr[i][1] - groundTruth_arr[j][1]
            distMatrix[i][j] = abs(get_norm(distX, distY))

    # munkres algorithm
    indexes = Munkres().compute(distMatrix)
    validIndexLength = min(len(estCentroid_arr), len(groundTruth_arr))
    valid_lst = [i for i in range(validIndexLength)]
    if len(estCentroid_arr) >= len(groundTruth_arr):
        matchIndexes = list(filter((lambda index : index[1] in valid_lst), indexes))
    else:
        matchIndexes = list(filter((lambda index : index[0] in valid_lst), indexes))

    trueCount = 0
    for i in range(len(matchIndexes)):
        estIndex = matchIndexes[i][0]
        truthIndex = matchIndexes[i][1]
        if distMatrix[estIndex][truthIndex] < distTreshold:
            trueCount += 1

    accuracy = trueCount / n
    print("Accuracy: {}".format(accuracy))


if __name__ = "__main__":
    estCentroid_arr = np.load("")
    groundTruth_arr = get_groundTruth("")
    accuracy(estCentroid_arr, groundTruth_arr, 6)
