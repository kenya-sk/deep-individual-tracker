#! /usr/bin/env python
#coding: utf-8

import numpy as np
import pandas as ps

def get_groundTruth(groundTruthPath):
    groundTruth_df = pd.read_csv(groundTruthPath)
    groundTruth_arr = np.array(df)

    return groundTruth_arr

def accuracy(estCentroid_arr, groundTruth_arr):
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

    





if __name__ = "__main__":
    estCentroid_arr = np.load("")
    groundTruth_arr = get_groundTruth("")
    accuracy(estCentroid_arr, groundTruth_arr)
