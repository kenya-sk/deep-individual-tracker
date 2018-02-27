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
    is_training = tf.placeholder(tf.bool)

    #model
    #layer1
    W_conv1 = tf.get_variable("conv1/weight1/weight1", [7,7,3,32])
    h_conv1 = tf.nn.leaky_relu(cnn_pixel.conv2d(X, W_conv1))
    h_pool1 = cnn_pixel.max_pool_2x2(h_conv1)
    #layer2
    W_conv2 = tf.get_variable("conv2/weight2/weight2", [7,7,32,32])
    h_conv2 = tf.nn.leaky_relu(cnn_pixel.conv2d(h_pool1, W_conv2))
    h_pool2 = cnn_pixel.max_pool_2x2(h_conv2)
    #layer3
    W_conv3 = tf.get_variable("conv3/weight3/weight3", [5,5,32,64])
    h_conv3 = tf.nn.leaky_relu(cnn_pixel.conv2d(h_pool2, W_conv3))
    #layer4
    W_fc4 = tf.get_variable("fc4/weight4/weight4", [18*18*64, 1000])
    h_conv3_flat = tf.reshape(h_conv3, [-1, 18*18*64])
    h_fc4 = tf.nn.leaky_relu(tf.matmul(h_conv3_flat, W_fc4))
    #layer5
    W_fc5 = tf.get_variable("fc5/weight5/weight5", [1000, 400])
    h_fc5 = tf.nn.leaky_relu(tf.matmul(h_fc4, W_fc5))
    #layer6
    W_fc6 = tf.get_variable("fc6/weight6/weight6", [400, 324])
    h_fc6 = tf.nn.leaky_relu(tf.matmul(h_fc5, W_fc6))
    #layer7
    W_fc7 = tf.get_variable("fc7/weight7/weight7", [324, 1])
    b_fc7 = tf.get_variable("fc7/bias7/bias7", [1])
    h_fc7 = tf.nn.leaky_relu(tf.matmul(h_fc6, W_fc7) + b_fc7)

    batchSize = 20000
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, "./model_pixel/2018_2_14_13_12/model.ckpt")

        img = cv2.imread("../image/original/11_20880.png")
        img = img[:470, :]
        height = img.shape[0]
        width = img.shape[1]
        img_lst = cnn_pixel.get_local_data(img, None, 72)
        assert len(img_lst) == height*width
        estDensMap = np.zeros((height*width), dtype="float32")
        train_n_batches = int(len(img_lst) / batchSize)

        for batch in range(train_n_batches):
            startIndex = batch*batchSize
            endIndex = startIndex + batchSize
            estDensMap[startIndex:endIndex] = sess.run(h_fc7, feed_dict={
                            X: np.array(img_lst[startIndex:endIndex]).reshape(-1, 72, 72, 3),
                            is_training: False}).reshape(batchSize)
            print("DONE: batch:{}".format(batch))

        estDensMap = estDensMap.reshape(height, width)
        print("DONE: estimate density map")

        # calculate estimation loss
        label = np.load("../data/dens/10/11_20880.npy")[:470, :]
        loss = np.mean(label - estDensMap, dtype="float64")
        print("estimation loss: {}".format(loss))

    return estDensMap


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
    estDensMap = estimate()
    np.save("./estimation/estimation.npy", estDensMap)
    #estDensMap = np.load("./estimation/2018_2_9_17_31/estimation.npy")
    #centroid_arr = clustering(estDensMap, 5, 0)
    #plot_estimation_box(centroid_arr, 12)
