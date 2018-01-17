#! /usr/bin/env python
#coding: utf-8

import os
import re
import time
import cv2
import numpy as np
from math import floor
from sklearn.model_selection import train_test_split
import tensorflow as tf


def load_data(inputDirPath):
    def get_file_path(inputDirPath):
        file_lst = os.listdir(inputDirPath)
        pattern = r"^(?!._).*(.png)$"
        repattern = re.compile(pattern)
        file_lst = [name for name in file_lst if repattern.match(name)]
        return file_lst

    X = []
    y = []
    file_lst = get_file_path(inputDirPath)
    for path in file_lst:
        X.append(cv2.imread("../image/original/tmp/" + path))
        densPath = path.replace(".png", ".npy")
        y.append(np.load("../data/dens/tmp/" + densPath))
    X = np.array(X)
    y = np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    return X_train, X_test, y_train, y_test


def get_local_image(image, localImgSize, resize):
    height = image.shape[0]
    width = image.shape[1]
    pad = floor(localImgSize/2)
    if len(image.shape) == 3:
        padImg = np.zeros((height + pad * 2, width + pad * 2, image.shape[2]))
        localImg = np.zeros((localImgSize, localImgSize, image.shape[2]))
    else:
        padImg = np.zeros((height + pad * 2, width + pad * 2))
        localImg = np.zeros((localImgSize, localImgSize))

    padImg[pad:height+pad, pad:width+pad] = image
    localImg_lst = []
    for h in range(pad,height+pad,localImgSize):
        for w in range(pad,width+pad, localImgSize):
            tmpLocalImg = np.array(localImg)
            tmpLocalImg = padImg[h-pad:h+pad+1, w-pad:w+pad+1]
            if resize == True:
                # resize answer data and flat
                tmpLocalImg = cv2.resize(tmpLocalImg, (18, 18))
                tmpLocalImg = np.reshape(tmpLocalImg, -1)
            localImg_lst.append(tmpLocalImg)
    return localImg_lst



# initialize weight by normal distribution (standard deviation: 0.1)
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


# initialize bias by normal distribution (standard deviation: 0.1)
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# convolutional layer
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")


# pooling layer
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


def main(X_train, X_test, y_train, y_test):
    # input
    with tf.name_scope("X"):
        X = tf.placeholder(tf.float32, [None, 71, 71, 3])
    # answer data
    with tf.name_scope("y_"):
        y_ = tf.placeholder(tf.float32, [None, 18*18])


    # first layer
    # convlution -> ReLU -> max pooling
    # input 72x72x3 -> output 36x36x32
    with tf.name_scope("conv1"):
        #7x7x3 filter
        W_conv1 = weight_variable([7,7,3,32])
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(X, W_conv1) + b_conv1)

    with tf.name_scope("pool1"):
        h_pool1 = max_pool_2x2(h_conv1)

    # second layer
    # convlution -> ReLU -> max pooling
    # input 36x36x32 -> output 18x18x32
    with tf.name_scope("conv2"):
        # 7x7x32 filter
        W_conv2 = weight_variable([7,7,32,32])
        b_conv2 = bias_variable([32])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    with tf.name_scope("pool2"):
        h_pool2 = max_pool_2x2(h_conv2)

    # third layer
    # convolution -> ReLU
    # input 18x18x32 -> output 18x18x64
    with tf.name_scope("conv3"):
        # 5x5x32 filter
        W_conv3 = weight_variable([5,5,32,64])
        b_conv3 = bias_variable([64])
        h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)

    # fourth layer
    # fully connected layer
    # input 18x18x64 -> output 1000
    with tf.name_scope("fc4"):
        W_fc4 = weight_variable([18*18*64, 1000])
        b_fc4 = bias_variable([1000])
        h_conv3_flat = tf.reshape(h_conv3, [-1, 18*18*64])
        h_fc4 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc4) + b_fc4)

    # fifth layer
    # fully connected layer
    # input 1000 -> output 400
    with tf.name_scope("fc5"):
        W_fc5 = weight_variable([1000, 400])
        b_fc5 = bias_variable([400])
        h_fc5 = tf.nn.relu(tf.matmul(h_fc4, W_fc5) + b_fc5)

    # sixth layer
    # fully connected layer
    # input 400 -> output 324
    with tf.name_scope("fc6"):
        W_fc6 = weight_variable([400, 324])
        b_fc6 = bias_variable([324])
        h_fc6 = tf.nn.relu(tf.matmul(h_fc5, W_fc6) + b_fc6)

    # loss function
    h_fc6_flat = tf.reshape(h_fc6, [-1, 18*18])
    with tf.name_scope("loss"):
        loss = tf.reduce_mean(tf.square(y_ - h_fc6_flat))

    # learning algorithm (learning rate: 0.01)
    with tf.name_scope("train"):
        train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

    # variable of TensorBoard
    with tf.name_scope("summary"):
        tf.summary.scalar("loss", loss)
        merged = tf.summary.merge_all()

    # save weight
    saver = tf.train.Saver()

    # learning
    startTime = time.time()
    loss_lst = []
    n_steps = 10
    batchSize = 5

    # start session and initialize
    with tf.Session() as sess:
        writer = tf.summary.FileWriter("./logs", sess.graph)
        sess.run(tf.global_variables_initializer())

        print("Original Traning data size: {}".format(len(X_train)))
        for step in range(n_steps):
            print("elapsed time: {0:.3f} [sec]".format(time.time() - startTime))
            for i in range(len(X_train)):
                X_train_local = get_local_image(X_train[i], 71, False)
                y_train_local = get_local_image(y_train[i], 71, True)
                n_batches = int(len(X_train_local) / batchSize)

                for i in range(n_batches):
                    startIndex = i * batchSize
                    endIndex = startIndex + batchSize
                    if i%batchSize == 0:
                        print("step: {0}, batch: {1} / {2}".format(step, i, n_batches))
                        train_loss = loss.eval(feed_dict={
                                X: X_train_local[startIndex:endIndex],
                                y_: y_train_local[startIndex:endIndex]})
                        #sess.run(loss, feed_dict={
                        #        X: X_train_local[startIndex:endIndex],
                        #        y_: y_train_local[startIndex:endIndex]})
                        print("loss: {}".format(train_loss))
                        loss_lst.append(train_loss)

                    train_step.run(feed_dict={
                        X: X_train_local[startIndex:endIndex],
                        y_:
                        y_train_local[startIndex:endIndex]})
                    #sess.run(train_step, feed_dict={
                    #    X: X_train_local[startIndex:endIndex],
                    #    y_: y_train_local[startIndex:endIndex]})

        # test (every step)
        for i in range(len(X_test)):
            X_test_local = get_local_image(X_test[0], 71, False)
            y_test_local = get_local_image(y_test[0], 71, True)
            test_loss += loss.eval(feed_dict={X: X_test_local, y_: y_test_local})
        print("test accuracy {}".format(test_loss/len(X_test)))

        np.save("../loss.npy", np.array(loss_lst))
        sess.close()

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data("../image/original/tmp")
    # tmp variable (NOT WORK)
    main(X_train, X_test, y_train, y_test)
