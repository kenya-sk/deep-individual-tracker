#! /usr/bin/env python
#coding: utf-8

import os
import re
import time
import sys
import cv2
import numpy as np
import pandas as pd
import math
import tensorflow as tf

import cnn_pixel

def estimation(modelPath):
    # ------------------------ MODEL -------------------------
    # start session
    sess = tf.InteractiveSession()
    X = tf.placeholder(tf.float32, [None, 72, 72, 3], name="input")
    y_ = tf.placeholder(tf.float32, [None, 1], name="label")
    is_training = tf.placeholder(tf.bool, name="is_training")

    W_conv1 = cnn_pixel.weight_variable([7,7,3,32])
    conv1 = cnn_pixel.conv2d(X, W_conv1)
    conv1_bn = cnn_pixel.batch_norm(conv1, [0, 1, 2], 32, is_training)
    h_conv1 = tf.nn.leaky_relu(conv1_bn)
    h_pool1 = cnn_pixel.max_pool_2x2(h_conv1)

    W_conv2 = cnn_pixel.weight_variable([7,7,32,32])
    conv2 = cnn_pixel.conv2d(h_pool1, W_conv2)
    conv2_bn = cnn_pixel.batch_norm(conv2, [0, 1, 2], 32, is_training)
    h_conv2 = tf.nn.leaky_relu(conv2_bn)
    h_pool2 = cnn_pixel.max_pool_2x2(h_conv2)

    W_conv3 = cnn_pixel.weight_variable([5,5,32,64])
    conv3 = cnn_pixel.conv2d(h_pool2, W_conv3)
    conv3_bn = cnn_pixel.batch_norm(conv3, [0, 1, 2], 64, is_training)
    h_conv3 = tf.nn.leaky_relu(conv3_bn)

    W_fc4 = cnn_pixel.weight_variable([18*18*64, 1000])
    h_conv3_flat = tf.reshape(h_conv3, [-1, 18*18*64])
    fc4 = tf.matmul(h_conv3_flat, W_fc4)
    fc4_bn = cnn_pixel.batch_norm(fc4, [0], 1000, is_training)
    h_fc4 = tf.nn.leaky_relu(fc4_bn)

    W_fc5 = cnn_pixel.weight_variable([1000, 400])
    fc5 = tf.matmul(h_fc4, W_fc5)
    fc5_bn = cnn_pixel.batch_norm(fc5, [0], 400, is_training)
    h_fc5 = tf.nn.leaky_relu(fc5_bn)

    W_fc6 = cnn_pixel.weight_variable([400, 324])
    fc6 = tf.matmul(h_fc5, W_fc6)
    fc6_bn = cnn_pixel.batch_norm(fc6, [0], 324, is_training)
    h_fc6 = tf.nn.leaky_relu(fc6_bn)

    W_fc7 = cnn_pixel.weight_variable([324, 1])
    b_fc7 = cnn_pixel.bias_variable([1])
    y = tf.nn.leaky_relu(tf.matmul(h_fc6, W_fc7) + b_fc7)
    # -----------------------------------------------------------------


    # ------------------------ ESTIMATION -------------------------
    # mask index
    # if you analyze all areas, please set a white image
    indexH, indexW = cnn_pixel.get_masked_index("/data/sakka/image/mask.png")
    assert len(indexH) == len(indexW)

    saver = tf.train.Saver() # save weight
    ckpt = tf.train.get_checkpoint_state(modelPath) # model exist: True or False

    # check if the ckpt exist
    if ckpt:
        lastModel = ckpt.model_checkpoint_path
        print("LODE: {}".format(lastModel))
        saver.restore(sess, lastModel)
    else:
        sys.stderr("Error: not found checkpoint file")
        sys.exit(1)

    img = cv2.imread("../image/original/16_100920.png")
    label = np.load("../data/dens/25/16_100920.npy")
    maskedImg = cnn_pixel.get_masked_data(img)
    maskedLabel = cnn_pixel.get_masked_data(label)
    X_local, y_local = cnn_pixel.get_local_data(maskedImg, maskedLabel, 72, indexH, indexW)
    est_arr = np.zeros(len(indexH))
    estDensMap = np.zeros((720, 1280), dtype="float32")

    estBatchSize = 2500
    est_n_batches = int(len(X_local)/estBatchSize)

    print("STSRT: estimate density map")
    for batch in range(est_n_batches):
        startIndex = batch*estBatchSize
        endIndex = startIndex + estBatchSize

        est_arr[startIndex:endIndex] = sess.run(y, feed_dict={
            X: X_local[startIndex:endIndex].reshape(-1, 72, 72, 3),
            y_: y_local[startIndex:endIndex].reshape(-1, 1),
            is_training: False}).reshape(estBatchSize)
        print("DONE: batch {}".format(batch))

    for i in range(len(indexH)):
        estDensMap[indexH[i], indexW[i]] = est_arr[i]

    np.save("./estimation/estimation.npy", estDensMap)
    print("END: estimate density map")

    # calculate estimation loss
    estLoss = np.mean(np.square(label - estDensMap), dtype="float32")
    print("estimation loss: {}".format(estLoss))
    # -----------------------------------------------------------------

if __name__ == "__main__":
    modelPath = "/data/sakka/tensor_model/2018_4_15_15_7/"
    estimation(modelPath)
