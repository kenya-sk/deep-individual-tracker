#! /usr/bin/env python
# coding: utf-8

import math
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.cluster import MeanShift


def clustering(densMap, bandwidth, thresh=0):
    # search high value cordinates
    while True:
        # point[0]: y  point[1]: x
        point = np.where(densMap > thresh)
        # X[:, 0]: x  X[:,1]: y
        X = np.vstack((point[1], point[0])).T
        if X.shape[0] > 0:
            break
        else:
            if thresh > 0:
                thresh -= -0.05
            else:
                return np.zeros((0,2))

    # MeanShift clustering
    print("START: clustering")
    ms = MeanShift(bandwidth=bandwidth, seeds=X)
    ms.fit(X)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_
    labels_unique = np.unique(labels)
    n_clusters = len(labels_unique)
    centroid_arr = np.zeros((n_clusters, 2))
    for k in range(n_clusters):
        centroid_arr[k] = cluster_centers[k]
    print("DONE: clustering")

    return centroid_arr

def plot_estimation_box(img, centroid_arr,hour, minute, boxSize=12):
    # get cordinates of vertex(lert top and right bottom)
    def get_rect_vertex(x, y, boxSize):
        vertex = np.zeros((2, 2), dtype=np.uint16)
        shift = int(boxSize/2)
        # left top corner
        vertex[0][0] = x - shift
        vertex[0][1] = y - shift
        # right bottom corner
        vertex[1][0] = x + shift
        vertex[1][1] = y + shift

        return vertex

    print("Number of cluster: {}".format(centroid_arr.shape[0]))
    for i in range(centroid_arr.shape[0]):
        x = int(centroid_arr[i][0])
        y = int(centroid_arr[i][1])
        img = cv2.circle(img, (x, y), 2, (0, 0, 255), -1, cv2.LINE_AA)
        vertex = get_rect_vertex(x, y, boxSize)
        img = cv2.rectangle(img, (vertex[0][0], vertex[0][1]), (vertex[1][0], vertex[1][1]), (0, 0, 255), 3)

    cv2.imwrite("/data/sakka/image/estBox/{0}_{1}.png".format(hour, minute), img)
    print("Done({0}:{1}): plot estimation box\n".format(hour, minute))


if __name__ == "__main__":
    for hour in range(10, 17):
        for minute in range(1, 62):
            estDensMap = np.load("/data/sakka/estimation/{0}_{1}.npy".format(hour, minute))
            img = cv2.imread("/data/sakka/image/est/{0}_{1}.png".format(hour, minute))
            centroid_arr = clustering(estDensMap, 20, 0.7)
            plot_estimation_box(img, centroid_arr, hour, minute, 12)