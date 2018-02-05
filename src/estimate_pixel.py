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

import cnn_pixel


def estimate():
    X = tf.placeholder(tf.float32, [None, 72, 72, 3])
    y_ = tf.placeholder(tf.float32, [None])

    #model
    #layer1
    W_conv1 = tf.get_variable("conv1/weight1/weight1", [7,7,3,32])
    b_conv1 = tf.get_variable("conv1/bias1/bias1", [32])
    h_conv1 = tf.nn.leaky_relu(cnn_pixel.conv2d(X, W_conv1) + b_conv1)
    h_pool1 = cnn_pixel.max_pool_2x2(h_conv1)
    #layer2
    W_conv2 = tf.get_variable("conv2/weight2/weight2", [7,7,32,32])
    b_conv2 = tf.get_variable("conv2/bias2/bias2", [32])
    h_conv2 = tf.nn.leaky_relu(cnn_pixel.conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = cnn_pixel.max_pool_2x2(h_conv2)
    #layer3
    W_conv3 = tf.get_variable("conv3/weight3/weight3", [5,5,32,64])
    b_conv3 = tf.get_variable("conv3/bias3/bias3", [64])
    h_conv3 = tf.nn.leaky_relu(cnn_pixel.conv2d(h_pool2, W_conv3) + b_conv3)
    #layer4
    W_fc4 = tf.get_variable("fc4/weight4/weight4", [18*18*64, 1000])
    b_fc4 = tf.get_variable("fc4/bias4/bias4", [1000])
    h_conv3_flat = tf.reshape(h_conv3, [-1, 18*18*64])
    h_fc4 = tf.nn.leaky_relu(tf.matmul(h_conv3_flat, W_fc4) + b_fc4)
    #layer5
    W_fc5 = tf.get_variable("fc5/weight5/weight5", [1000, 400])
    b_fc5 = tf.get_variable("fc5/bias5/bias5", [400])
    h_fc5 = tf.nn.leaky_relu(tf.matmul(h_fc4, W_fc5) + b_fc5)
    #layer6
    W_fc6 = tf.get_variable("fc6/weight6/weight6", [400, 324])
    b_fc6 = tf.get_variable("fc6/bias6/bias6", [324])
    h_fc6 = tf.nn.leaky_relu(tf.matmul(h_fc5, W_fc6) + b_fc6)
    #layer7
    W_fc7 = tf.get_variable("fc7/weight7/weight7", [324, 1])
    b_fc7 = tf.get_variable("fc7/bias7/bias7", [1])
    h_fc7 = tf.nn.leaky_relu(tf.matmul(h_fc6, W_fc7) + b_fc7)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, "./model_pixel/2018_2_2_20_43/model.ckpt")

        img = cv2.imread("../image/original/11_20880.png")
        img = img[:470, :]
        height = img.shape[0]
        width = img.shape[1]
        img_lst = cnn_pixel.get_local_data(img, None, 72)
        assert len(img_lst) == height*width
        estDensMap = np.zeros((height, width), dtype="float64")

        i = 0
        for h in range(height):
            for w in range(width):
                output = sess.run(h_fc7, feed_dict={X: img_lst[i].reshape(1, 72, 72, 3)})
                estDensMap[h][w] = output
                i += 1
                if i%300 == 0:
                    print(i)
        print("DONE: estimate density map")

    return estDensMap


def clustering(densMap, bandwidth, thresh=0):
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


if __name__ == "__main__":
    #estDensMap = estimate()
    estDensMap = np.load("./estimation.npy")
    centroid_arr = clustering(estDensMap, 5, 0)
