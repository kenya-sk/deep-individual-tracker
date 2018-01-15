#! /usr/bin/env python
#coding: utf-8

import os
import re
import time
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf


def load_data(inputDirPath):
    def get_file_path(inputDirPath):
        file_lst = os.listdir(inputDircPath)
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
        y.append(np.load("../data/dens/" + densPath))
    X = np.array(X)
    y = np.array(y)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=1)
    return X_train, X_val, y_train, y_val


def get_local_image(image, localImgSize, resize):
    height = image.shpae[0]
    width = iamge.shape[1]
    pad = localImgSize - 1
    if image.shpape == 3:
        padImg = np.zeros((height + pad * 2, width + pad * 2, image.shape[2]))
        localImg = np.zeros((localImgSize, localImgSize, image.shape[2]))
    else:
        padImg = np.zeros((height + pad * 2, width + pad * 2))
        localImg = np.zeros((localImgSize, localImgSize))

    padImg[pad:heigh+pad, pad:width+pad] = image
    localImg_lst = []
    for h in range(pad,height+pad):
        for w in range(pad,width+pad):
            localImg = padImg
            if resize == True:
                # resize answer data
                cv2.resize(localImg, (18, 18))
            localImg_lst.append(localImg)
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


def main(X_train, X_val, y_train, y_val):
    inputDim = [72, 72, 3]
    outputDim = 1
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    # input
    with tf.name_scope("X"):
        X = tf.placeholder(tf.float32, [None, inputDim])
    # answer data
    with tf.name_scope("y_"):
        y_ = tf.placeholder(tf.float32, [None, outputDim])

    # first layer
    # convlution -> ReLU -> max pooling
    # input 72x72x3 -> output 36x36x32
    with tf.name_scope("conv1"):
        #7x7x3 filter
        W_conv1 = weight_variable([7,7,3,32])
        b_conv1 = bias_variable([32])
        #x_image = tf.reshape(x, [-1, 72, 72, 3])
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
        writer = tf.summary.FileWriter("./logs", sess.graph)

    # learning
    startTime = time.time()
    n_steps = 20000
    batch_size = 50
    for step in range(n_steps):
        if step % 100 == 0:
            print("step: {0}".format(i))
            print("elapsed time: {0:.3f} [sec]".format(time.time() - startTime))
            print("loss: {0}".format(loss))
        for i in range(len(X_train)):
            X_train_local = get_local_image(X_train[i], 71, False)
            y_train_local = get_local_image(y_train[i], 71, True)
            n_batches = int(len(X_train_local) / batch_size)
            for i in range(n_batches):
                startIndex = i * batch_size
                endIndex = startIndex + batch_size
                train_step.run(feed_dict={X: X_train_local[startIndex:endIndex], y_: y_train[startIndex:endIndex]})

    sess.close()

if __name__ == "__main__":
    X_train, X_val, y_train, y_val = load_data("../image/original/tmp")
    # tmp variable (NOT WORK)
    main(X_train, X_val, y_train, y_val)
