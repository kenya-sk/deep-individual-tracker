#! /usr/bin/env python
#coding: utf-8

import numpy as np
import pandas as pd
from scipy import optimize

from clustering import clustering
from cnn_pixel import get_masked_index

def get_groundTruth(groundTruthPath, maskPath=None):
    groundTruth_arr = np.array(pd.read_csv(groundTruthPath))
    if maskPath is None:
        return groundTruth_arr
    else:
        validH, validW = get_masked_index(maskPath)
        validGroundTruth_lst = []

        for i in range(groundTruth_arr.shape[0]):
            indexW = np.where(validW == groundTruth_arr[i][0])
            indexH = np.where(validH == groundTruth_arr[i][1])
            intersect = np.intersect1d(indexW, indexH)
            if len(intersect) == 1:
                validGroundTruth_lst.append([validW[intersect[0]], validH[intersect[0]]])
            else:
                pass
        return np.array(validGroundTruth_lst)


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

    # calculate by hangarian algorithm
    row, col = optimize.linear_sum_assignment(distMatrix)
    indexes = []
    for i in range(n):
        indexes.append([row[i], col[i]])
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
        if distMatrix[estIndex][truthIndex] <= distTreshold:
            trueCount += 1

    accuracy = trueCount / n
    print("******************************************")
    print("Accuracy: {}".format(accuracy))
    print("******************************************")


if __name__ == "__main__":
    bandWidth = 20
    estDensMap = np.load("./estimation/estimation.npy")
    centroid_arr = clustering(estDensMap, bandWidth, thresh=0.7)
    groundTruth_arr = get_groundTruth("/data/sakka/cord/12_3480.csv", maskPath="../image/mask.png")
    accuracy(centroid_arr, groundTruth_arr, bandWidth)
