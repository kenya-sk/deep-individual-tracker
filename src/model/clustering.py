#! /usr/bin/env python
# coding: utf-8

import math
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import glob
from sklearn.cluster import MeanShift


def clustering(dens_map, band_width, thresh=0):
    # search high value cordinates
    while True:
        # point[0]: y  point[1]: x
        point = np.where(dens_map > thresh)
        # X[:, 0]: x  X[:,1]: y
        X = np.vstack((point[1], point[0])).T
        if X.shape[0] > 0:
            break
        else:
            if thresh > 0:
                thresh -= 0.05
            else:
                return np.zeros((0, 2))

    # MeanShift clustering
    print("START: clustering")
    ms = MeanShift(bandwidth=band_width, seeds=X)
    ms.fit(X)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_
    labels_unique = np.unique(labels)
    n_clusters = len(labels_unique)
    centroid_arr = np.zeros((n_clusters, 2))
    for k in range(n_clusters):
        centroid_arr[k] = cluster_centers[k]
    print("DONE: clustering")

    return centroid_arr.astype(np.int32)


def plot_prediction_box(img, centroid_arr,hour, minute, out_pred_box_dirc,box_size=12):
    # get cordinates of vertex(lert top and right bottom)
    def get_rect_vertex(x, y, box_size):
        vertex = np.zeros((2, 2), dtype=np.uint16)
        shift = int(box_size/2)
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
        vertex = get_rect_vertex(x, y, box_size)
        img = cv2.rectangle(img, (vertex[0][0], vertex[0][1]), (vertex[1][0], vertex[1][1]), (0, 0, 255), 3)

    cv2.imwrite(out_pred_box_dirc + "{0}_{1}.png".format(hour, minute), img)
    print("Done({0}:{1}): plot estimation box\n".format(hour, minute))


if __name__ == "__main__":
    dens_map_path = "/data/sakka/estimation/1h_10/model_201804151507/dens/15/*.npy"
    out_clustering_dirc = "/data/sakka/estimation/1h_10/model_201804151507/cord/15/"
    out_pred_box_dirc = "/data/sakka/image/estBox/"
    
    # for hour in range(10, 17):
    #     for minute in range(1, 62):
    #         est_dens_map = np.load("/data/sakka/estimation/{0}_{1}.npy".format(hour, minute))
    #         img = cv2.imread("/data/sakka/image/est/{0}_{1}.png".format(hour, minute))
    #         centroid_arr = clustering(est_dens_map, 20, 0.7)
    #         plot_prediction_box(img, centroid_arr, hour, minute, out_pred_box_dirc,box_size=12)

    # check 1h data
    file_lst = glob.glob(dens_map_path)
    for file_path in file_lst:
        est_dens_map = np.load(file_path)
        centroid_arr = clustering(est_dens_map, 25, 0.4)
        file_num = file_path.split("/")[-1][:-4]
        np.save(out_clustering_dirc + "{}.npy".format(file_num), centroid_arr)
