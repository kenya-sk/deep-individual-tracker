#! /usr/bin/env python
#coding: utf-8

import numpy as np
import pandas as pd
import csv
from scipy import optimize

from clustering import clustering
from cnn_pixel import get_masked_index

def get_groundTruth(groundTruthPath, maskPath=None):
    """
    plots the coordinates of answer label on the black image(all value 0) and
    creates a correct image for accuracy evaluation.

    input:
        groundTruthPath: coordinates of the estimated target (.csv)
        maskPath: enter path only when using mask image

    output:
        array showing the positon of target
    """
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
    """
    the distance between the estimation and groundtruth is less than distThreshold --> True

    input:
        estCentroid_arr: coordinates of the estimated target (.npy). array shape == original image shape
        groundTruth_arr: coordinates of the answer label (.csv)
        distThreshold: it is set maximum value of kernel width

    output:
        accuracy per frame
    """

    # make distance matrix
    n = max(len(estCentroid_arr), len(groundTruth_arr))
    distMatrix = np.zeros((n,n))
    for i in range(len(estCentroid_arr)):
        for j in range(len(groundTruth_arr)):
            diff_cord = estCentroid_arr[i] - groundTruth_arr[j]
            distMatrix[i][j] = np.linalg.norm(diff_cord[0])

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
    print("******************************************\n")

    return accuracy

if __name__ == "__main__":
    bandWidth = 25
    skip_lst = [15]
    for skip in skip_lst:
        accuracy_lst = []
        for file_num in range(1, 36):
            estDensMap = np.load("/data/sakka/estimation/test_image/model_201806142123/dens/{0}/{1}.npy".format(skip, file_num))
            centroid_arr = clustering(estDensMap, bandWidth, thresh=0.4)
            np.save("/data/sakka/estimation/test_image/model_201806142123/cord/{0}/{1}.npy".format(skip, file_num), centroid_arr)
            if centroid_arr.shape[0] == 0:
                print("Not found point of centroid\nAccuracy is 0.0")
                accuracy_lst.append(0.0)
            else:
                groundTruth_arr = get_groundTruth("/data/sakka/cord/test_image/model_201806142123/{0}.csv".format(file_num), maskPath="/data/sakka/image/mask.png")
                accuracy_lst.append(accuracy(centroid_arr, groundTruth_arr, bandWidth))

        print("\n******************************************")
        print("Toal Accuracy (data size {0}, sikp size {1}): {2}".format(len(accuracy_lst), skip, sum(accuracy_lst)/len(accuracy_lst)))
        print("******************************************")

        with open("/data/sakka/estimation/test_image/model_201806142123/accuracy/{}/accuracy.csv".format(skip), "w") as f:
            writer = csv.writer(f)
            writer.writerow(accuracy_lst)
        print("SAVE: accuracy data")
