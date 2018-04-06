#! /usr/bin/env python
#coding: utf-8

import numpy as np
import pandas as ps
from munkres import Munkres

import estimate_pixel

def get_groundTruth(groundTruthPath):
    return np.array(pd.read_csv(groundTruthPath))


def accuracy(estCentroid_arr, groundTruth_arr, distTreshold):
# the distance between the estimation and groundtruth is less than distThreshold --> True
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
    bandWidth = 15
    estDensMap = np.load("./estimation/estimation.npy")
    centroid_arr = clustering(estDensMap, bandWidth, thresh=0.7)
    groundTruth_arr = get_groundTruth("../data/cord/16_100920.csv")
    accuracy(centroid_arr, groundTruth_arr, bandWidth)
