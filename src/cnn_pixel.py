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
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


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
        img = cv2.imread("../image/original/test/" + path)
        if img is None:
            print("Error: can not read image")
            sys.exit(1)
        else:
            X.append(img)
        densPath = path.replace(".png", ".npy")
        y.append(np.load("../data/dens/6/test/" + densPath))
    X = np.array(X)
    y = np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return X_train, X_test, y_train, y_test


def get_local_data(image, densMap, localImgSize):
    # trimming original image(there are many unnecessary parts)
    image = image[:470, :]
    # local image size is even number
    height = image.shape[0]
    width = image.shape[1]
    pad = math.floor(localImgSize/2)
    if len(image.shape) == 3:
        padImg = np.zeros((height + pad * 2, width + pad * 2, image.shape[2]))
        localImg = np.zeros((localImgSize, localImgSize, image.shape[2]))
    else:
        padImg = np.zeros((height + pad * 2, width + pad * 2))
        localImg = np.zeros((localImgSize, localImgSize))

    padImg[pad:height+pad, pad:width+pad] = image
    img_lst = []
    for h in range(pad, height+pad):
        for w in range(pad, width+pad):
            tmpLocalImg = np.array(localImg)
            tmpLocalImg = padImg[h-pad:h+pad, w-pad:w+pad]
            img_lst.append(tmpLocalImg)
    if densMap is not None:
        densMap = densMap[:470, :]
        df = pd.DataFrame({"img_arr":img_lst, "label":np.ravel(densMap).astype(np.float32)})
        return df
    else:
        return img_lst


def under_sampling(data_df, thresh):
    # low frequently data: value of label is below thresh
    low_frequently_data = data_df[data_df["label"] > thresh]

    high_frequently_index = data_df[data_df["label"] <= thresh].index
    random_indices = np.random.choice(high_frequently_index,  len(low_frequently_data), replace=False)
    high_frequently_data = data_df.loc[random_indices]
    pd.DataFrame(high_frequently_data)

    merged_data = pd.concat([high_frequently_data, low_frequently_data], ignore_index=True)
    balanced_data = pd.DataFrame(merged_data)

    return balanced_data


# processing variables and it output tensorboard
def variable_summaries(var):
    # output scalar (mean, stddev, max, min, histogram)
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))


# initialize weight by He initialization
def weight_variable(shape, name=None):
    # initial = tf.truncated_normal(shape, stddev=0.1, dtype=tf.float32)

    # He initialization
    if len(shape) == 4:
        n = shape[1] * shape[2] * shape[3]
    elif len(shape) == 2:
        n = shape[0]
    else:
        print("Error: shape size is not correct !")
        sys.exit(1)
    stddev = math.sqrt(2/n)
    initial = tf.random_normal(shape, stddev=stddev, dtype=tf.float32)
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.Variable(initial, name=name)


# initialize bias by normal distribution (standard deviation: 0.1)
def bias_variable(shape, name=None):
    initial = tf.constant(0.1, shape=shape)
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.Variable(initial, name=name)


# convolutional layer
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")


# pooling layer
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

# batch normalization
def batch_norm(X, axes, shape, is_training):
    if is_training is False:
        return X
    epsilon  = 1e-5
    mean, variance = tf.nn.moments(X, axes)
    scale = tf.Variable(tf.ones([shape]))
    offset = tf.Variable(tf.zeros([shape]))
    return tf.nn.batch_normalization(X, mean, variance, offset, scale, epsilon)


def main(X_train, X_test, y_train, y_test):
    date = datetime.now()
    date_dir = "{0}_{1}_{2}_{3}_{4}".format(date.year, date.month, date.day, date.hour, date.minute)
    log_dir = "./logs_pixel/" + date_dir
    # delete the specified directory if it exists, recreate it
    if tf.gfile.Exists(log_dir):
        tf.gfile.DeleteRecursively(log_dir)
    tf.gfile.MakeDirs(log_dir)

    # start session
    sess = tf.InteractiveSession()

    with tf.name_scope("input"):
        # input image
        with tf.name_scope("X"):
            X = tf.placeholder(tf.float32, [None, 36, 36, 3], name="input")
            _ = tf.summary.image("X(input)", X[:, :, :, 0:1], 5)
        # answer image
        with tf.name_scope("y_"):
            y_ = tf.placeholder(tf.float32, [None], name="label")

    # status: True(lerning) or False(test)
    is_training = tf.placeholder(tf.bool)


    # first layer
    # convlution -> ReLU -> max pooling
    # input 36x36x3 -> output 18x18x32
    with tf.name_scope("conv1"):
        # 7x7x3 filter
        with tf.name_scope("weight1"):
            W_conv1 = weight_variable([7,7,3,32], name="weight1")
            variable_summaries(W_conv1)
            _ = tf.summary.image("image1", tf.transpose(W_conv1, perm=[3, 0, 1, 2])[:,:,:,0:1], max_outputs=32)
        with tf.name_scope("batchNorm1"):
            conv1 = conv2d(X, W_conv1)
            conv1_bn = batch_norm(conv1, [0, 1, 2], 32, is_training)
        with tf.name_scope("leakyRelu1"):
            h_conv1 = tf.nn.leaky_relu(conv1_bn)
            variable_summaries(h_conv1)

    with tf.name_scope("pool1"):
        h_pool1 = max_pool_2x2(h_conv1)
        variable_summaries(h_pool1)


    # second layer
    # convlution -> ReLU -> max pooling
    # input 18x18x32 -> output 9x9x32
    with tf.name_scope("conv2"):
        # 7x7x32 filter
        with tf.name_scope("weight2"):
            W_conv2 = weight_variable([7,7,32,32], name="weight2")
            variable_summaries(W_conv2)
            _ = tf.summary.image("image2", tf.transpose(W_conv2, perm=[3, 0, 1, 2])[:,:,:,0:1], max_outputs=32)
        with tf.name_scope("batchNorm2"):
            conv2 = conv2d(h_pool1, W_conv2)
            conv2_bn = batch_norm(conv2, [0, 1, 2], 32, is_training)
        with tf.name_scope("leakyRelu2"):
            h_conv2 = tf.nn.leaky_relu(conv2_bn)
            variable_summaries(h_conv2)

    with tf.name_scope("pool2"):
        h_pool2 = max_pool_2x2(h_conv2)
        variable_summaries(h_pool2)

    # third layer
    # convolution -> ReLU
    # input 9x9x32 -> output 9x9x64
    with tf.name_scope("conv3"):
        # 5x5x32 filter
        with tf.name_scope("weight3"):
            W_conv3 = weight_variable([5,5,32,64], name="weight3")
            variable_summaries(W_conv3)
            _ = tf.summary.image("image3", tf.transpose(W_conv3, perm=[3, 0, 1, 2])[:,:,:,0:1], max_outputs=64)
        with tf.name_scope("batchNorm3"):
            conv3 = conv2d(h_pool2, W_conv3)
            conv3_bn = batch_norm(conv3, [0, 1, 2], 64, is_training)
        with tf.name_scope("leakyRelu3"):
            h_conv3 = tf.nn.leaky_relu(conv3_bn)
            variable_summaries(h_conv3)

    # fourth layer
    # fully connected layer
    # input 9x9x64 -> output 1000
    with tf.name_scope("fc4"):
        with tf.name_scope("weight4"):
            W_fc4 = weight_variable([9*9*64, 1000], name="weight4")
            variable_summaries(W_fc4)
        with tf.name_scope("batchNorm4"):
            h_conv3_flat = tf.reshape(h_conv3, [-1, 18*18*64])
            fc4 = tf.matmul(h_conv3_flat, W_fc4)
            fc4_bn = batch_norm(fc4, [0], 1000, is_training)
        with tf.name_scope("flat4"):
            h_fc4 = tf.nn.leaky_relu(fc4_bn)
            variable_summaries(h_fc4)

    # fifth layer
    # fully connected layer
    # input 1000 -> output 400
    with tf.name_scope("fc5"):
        with tf.name_scope("weight5"):
            W_fc5 = weight_variable([1000, 400], name="weight5")
            variable_summaries(W_fc5)
        with tf.name_scope("batchNorm5"):
            fc5 = tf.matmul(h_fc4, W_fc5)
            fc5_bn = batch_norm(fc5, [0], 400, is_training)
        with tf.name_scope("flat5"):
            h_fc5 = tf.nn.leaky_relu(fc5_bn)
            variable_summaries(h_fc5)

    # sixth layer
    # fully connected layer
    # input 400 -> output 324
    with tf.name_scope("fc6"):
        with tf.name_scope("weight6"):
            W_fc6 = weight_variable([400, 324], name="weight6")
            variable_summaries(W_fc6)
        with tf.name_scope("batchNorm6"):
            fc6 = tf.matmul(h_fc5, W_fc6)
            fc6_bn = batch_norm(fc6, [0], 324, is_training)
        with tf.name_scope("flat6"):
            h_fc6 = tf.nn.leaky_relu(fc6_bn)
            variable_summaries(h_fc6)

    with tf.name_scope("fc7"):
        with tf.name_scope("weight7"):
            W_fc7 = weight_variable([324, 1], name="weight7")
            variable_summaries(W_fc7)
        with tf.name_scope("bias7"):
            b_fc7 = bias_variable([1], name="bias7")
            variable_summaries(b_fc7)
        with tf.name_scope("flat7"):
            h_fc7 = tf.nn.leaky_relu(tf.matmul(h_fc6, W_fc7) + b_fc7, name="output")
            variable_summaries(h_fc7)

    # output
    tf.summary.histogram("output", h_fc7)

    # loss function
    with tf.name_scope("loss"):
        loss = tf.reduce_mean(tf.square(y_ - h_fc7))
        tf.summary.scalar("loss", loss)

    # learning algorithm (learning rate: 0.01)
    with tf.name_scope("train"):
        train_step = tf.train.GradientDescentOptimizer(5e-4).minimize(loss)

    # variable of TensorBoard
    trainStep = 0
    testStep = 0
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(log_dir + "/train", sess.graph)
    test_writer = tf.summary.FileWriter(log_dir + "/test")

    # learning
    startTime = time.time()
    n_epochs = 20
    batchSize = 100
    tf.global_variables_initializer().run() # initialize all variable
    saver = tf.train.Saver() # save weight


    print("Original traning data size: {}".format(len(X_train)))
    for epoch in range(n_epochs):
        print("elapsed time: {0:.3f} [sec]".format(time.time() - startTime))
        for i in range(len(X_train)):
            train_df = get_local_data(X_train[i], y_train[i], 36)
            train_df = under_sampling(train_df, thresh=0.01)
            X_train_local = train_df["img_arr"]
            y_train_local = train_df["label"]
            X_train_local, y_train_local = shuffle(X_train_local, y_train_local)
            train_n_batches = int(len(X_train_local) / batchSize)

            for batch in range(train_n_batches):
                trainStep += 1
                startIndex = batch * batchSize
                endIndex = startIndex + batchSize
                #record loss data
                if batch%100 == 0:
                    print("traning data: {0} / {1}".format(i, len(X_train)))
                    print("epoch: {0}, batch: {1} / {2}".format(epoch, batch, train_n_batches))
                    summary, train_loss = sess.run([merged, loss], feed_dict={
                            X: np.vstack(X_train_local[startIndex:endIndex].values).reshape(-1, 36, 36, 3),
                            y_: y_train_local[startIndex:endIndex].values,
                            is_training:True})
                    train_writer.add_summary(summary, trainStep)
                    print("loss: {}\n".format(train_loss))

                summary, _ = sess.run([merged, train_step], feed_dict={
                                    X: np.vstack(X_train_local[startIndex:endIndex].values).reshape(-1, 36, 36, 3),
                                    y_: y_train_local[startIndex:endIndex].values,
                                    is_training:True})
                #train_writer.add_summary(summary, trainStep)

    # test data
    print("TEST")
    test_loss = 0.0
    for i in range(len(X_test)):
        test_df = get_local_data(X_test[i], y_test[i], 36)
        X_test_local = test_df["img_arr"]
        y_test_local = test_df["label"]
        test_n_batches = int(len(X_test_local) / batchSize)
        for batch in range(test_n_batches):
            startIndex = batch * batchSize
            endIndex = startIndex + batchSize

            summary, tmp_loss = sess.run([merged, loss], feed_dict={
                                    X: np.vstack(X_test_local[startIndex:endIndex]).reshape(-1, 36, 36, 3),
                                    y_: y_test_local[startIndex:endIndex].values,
                                    is_training:False})
            test_writer.add_summary(summary, testStep)
            test_loss += tmp_loss
            testStep += 1

    print("test loss: {}\n".format(test_loss/(len(X_test)*test_n_batches)))

    # end processing
    saver.save(sess, "./model_pixel/" + date_dir + "/model.ckpt")
    train_writer.close()
    test_writer.close()
    sess.close()

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data("../image/original/test")
    main(X_train, X_test, y_train, y_test)
