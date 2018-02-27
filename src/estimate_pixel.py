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
    # plot sample point and centroid
    def plot_cluster(X, cluster_centers, labels, n_clusters):
        plt.figure()
        plt.scatter(X[:, 0],X[:,1], c=labels)
        for k in range(n_clusters):
            cluster_center = cluster_centers[k]
            plt.plot(cluster_center[0], cluster_center[1], "*", markersize=5, c="red")
        plt.title("Estimated number of clusters: {}".format(n_clusters))
        plt.savefig("./estimateMap.png")
        print("save estimateMap.png")

    point = np.where(densMap > thresh)
    X = np.vstack((point[1], point[0])).T
    # MeanShift
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

    plot_cluster(X, cluster_centers, labels, n_clusters)

    return centroid_arr

def plot_estimation_box(centroid_arr, boxSize=12):
    print("cluster: {}".format(len(centroid_arr)))
    plt.figure(figsize=(12.8, 7.2))
    ax = plt.axes()
    centroid_X = centroid_arr[:, 0]
    centroid_Y = centroid_arr[:, 1] + 250
    plt.scatter(centroid_X, centroid_Y, s=10)
    for x, y in zip(centroid_X, centroid_Y):
        r = plt.Rectangle(xy=(x-boxSize/2, y-boxSize/2), width=boxSize, height=boxSize, ec='#000000', fill=False, edgecolor='red')
        ax.add_patch(r)
    plt.xlim(0, 1280)
    plt.ylim(0, 720)
    plt.savefig("./estimateBox.png")



if __name__ == "__main__":
    estDensMap = np.load("./estimation/2018_2_9_17_31/estimation.npy")
    centroid_arr = clustering(estDensMap, 5, 0)
    plot_estimation_box(centroid_arr, 12)
