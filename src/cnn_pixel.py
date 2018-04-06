#! /usr/bin/env python
#coding: utf-8

import os
import re
import time
import sys
import signal
import cv2
import numpy as np
import pandas as pd
import math
import tensorflow as tf
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

ANALYSIS_HEIGHT = (0, 470)
ANALYSIS_WIDTH = (0, 1280)


def load_data(inputImageDirPath, inputDensDirPath):
    def get_file_path(inputDirPath):
        try:
            file_lst = os.listdir(inputDirPath)
        except FileNotFoundError:
            sys.stderr.write("Error: not found directory")
            sys.exit(1)
        pattern = r"^(?!._).*(.png)$"
        repattern = re.compile(pattern)
        file_lst = [name for name in file_lst if repattern.match(name)]
        return file_lst

    X = []
    y = []
    file_lst = get_file_path(inputImageDirPath)
    for path in file_lst:
        img = cv2.imread(inputImageDirPath + path)
        if img is None:
            sys.stderr.write("Error: can not read image")
            sys.exit(1)
        else:
            X.append(get_masked_data(img))
        densPath = path.replace(".png", ".npy")
        densMap = np.load(inputDensDirPath + densPath)
        y.append(get_masked_data(densMap))
    X = np.array(X)
    y = np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return X_train, X_test, y_train, y_test


def get_masked_data(data):
    """
    data: image or density map
    mask: 3channel mask image. the value is 0 or 1
    """
    mask = cv2.imread("../image/mask.png")
    if mask is None:
        sys.stderr.write("Error: can not read mask image")
        sys.exit(1)

    if len(data.shape) == 3:
        maskData = data*mask
    else:
        maskData = mask[:,:,0]*data
    return maskData

def get_masked_index(maskPath):
    mask = cv2.imread(maskPath)
    if mask.shape[2] == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    index = np.where(mask > 0)
    indexH = index[0]
    indexW = index[1]
    assert len(indexH) == len(indexW)
    return indexH, indexW


def get_local_data(image, densMap, localImgSize, indexH, indexW):
    """
    ret: localImg_mat([#locals, localImgSize, localImgSize, image.shape[2]]), density_arr([#locals])
    """
    assert len(image.shape) == 3
    # trim original image
    image = image[ANALYSIS_HEIGHT[0]:ANALYSIS_HEIGHT[1], ANALYSIS_WIDTH[0]:ANALYSIS_WIDTH[1]]
    height = image.shape[0]
    width = image.shape[1]

    pad = math.floor(localImgSize/2)
    padImg = np.zeros((height + pad * 2, width + pad * 2, image.shape[2]), dtype="uint8")
    padImg[pad:height+pad, pad:width+pad] = image

    localImg_mat = np.zeros((len(indexW), localImgSize, localImgSize, image.shape[2]), dtype="uint8")
    density_arr = np.zeros((len(indexW)), dtype="float32")
    for idx in range(len(indexW)):
        # fix index(padImage)
        h = indexH[idx]
        w = indexW[idx]
        localImg_mat[idx] = padImg[h:h+2*pad,w:w+2*pad]
        density_arr[idx] = densMap[h, w]
    return localImg_mat, density_arr


def under_sampling(localImg_mat, density_arr, thresh):
    """
    ret: undersampled (localImg_mat, density_arr)
    """

    def select(length, k):
        """
        ret: array of boolean which length = length and #True = k
        """
        seed = np.arange(length)
        np.random.shuffle(seed)
        return seed < k

    assert localImg_mat.shape[0] == len(density_arr)

    msk = density_arr >= thresh # select all positive samples first
    msk[~msk] = select((~msk).sum(), msk.sum()) # select same number of negative samples with positive samples
    return localImg_mat[msk], density_arr[msk]


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
        #convolution layer
        n = shape[1] * shape[2] * shape[3]
    elif len(shape) == 2:
        # fully conected layer
        n = shape[0]
    else:
        sys.stderr.write("Error: shape size is not correct !")
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


def main(X_train, X_test, y_train, y_test, modelPath):
    # start session
    sess = tf.InteractiveSession()

    # ------------------------------- MODEL -----------------------------------
    # input image
    with tf.name_scope("input"):
        with tf.name_scope("X"):
            X = tf.placeholder(tf.float32, [None, 72, 72, 3], name="input")
            _ = tf.summary.image("X", X[:, :, :, :], 5)
        # answer image
        with tf.name_scope("y_"):
            y_ = tf.placeholder(tf.float32, [None, 1], name="label")
        # status: True(lerning) or False(test)
        with tf.name_scope("is_training"):
            is_training = tf.placeholder(tf.bool, name="is_training")


    # first layer
    # convlution -> ReLU -> max pooling
    # input 72x72x3 -> output 36x36x32
    with tf.name_scope("conv1"):
        # 7x7x3 filter
        with tf.name_scope("weight1"):
            W_conv1 = weight_variable([7,7,3,32])
            variable_summaries(W_conv1)
            _ = tf.summary.image("image1", tf.transpose(W_conv1, perm=[3, 0, 1, 2])[:,:,:,0:1], max_outputs=3)
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
    # input 36x36x32 -> output 18x18x32
    with tf.name_scope("conv2"):
        # 7x7x32 filter
        with tf.name_scope("weight2"):
            W_conv2 = weight_variable([7,7,32,32])
            variable_summaries(W_conv2)
            _ = tf.summary.image("image2", tf.transpose(W_conv2, perm=[3, 0, 1, 2])[:,:,:,0:1], max_outputs=3)
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
    # input 18x18x32 -> output 18x18x64
    with tf.name_scope("conv3"):
        # 5x5x32 filter
        with tf.name_scope("weight3"):
            W_conv3 = weight_variable([5,5,32,64])
            variable_summaries(W_conv3)
            _ = tf.summary.image("image3", tf.transpose(W_conv3, perm=[3, 0, 1, 2])[:,:,:,0:1], max_outputs=3)
        with tf.name_scope("batchNorm3"):
            conv3 = conv2d(h_pool2, W_conv3)
            conv3_bn = batch_norm(conv3, [0, 1, 2], 64, is_training)
        with tf.name_scope("leakyRelu3"):
            h_conv3 = tf.nn.leaky_relu(conv3_bn)
            variable_summaries(h_conv3)

    # fourth layer
    # fully connected layer
    # input 18x18x64 -> output 1000
    with tf.name_scope("fc4"):
        with tf.name_scope("weight4"):
            W_fc4 = weight_variable([18*18*64, 1000])
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
            W_fc5 = weight_variable([1000, 400])
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
            W_fc6 = weight_variable([400, 324])
            variable_summaries(W_fc6)
        with tf.name_scope("batchNorm6"):
            fc6 = tf.matmul(h_fc5, W_fc6)
            fc6_bn = batch_norm(fc6, [0], 324, is_training)
        with tf.name_scope("flat6"):
            h_fc6 = tf.nn.leaky_relu(fc6_bn)
            variable_summaries(h_fc6)

    # seven layer
    # fully connected layer
    # input 324 -> output 1
    with tf.name_scope("fc7"):
        with tf.name_scope("weight7"):
            W_fc7 = weight_variable([324, 1])
            variable_summaries(W_fc7)
        with tf.name_scope("bias7"):
            b_fc7 = bias_variable([1])
            variable_summaries(b_fc7)
        with tf.name_scope("flat7"):
            y = tf.nn.leaky_relu(tf.matmul(h_fc6, W_fc7) + b_fc7)
            variable_summaries(y)

    # output
    tf.summary.histogram("output", y)

    # loss function
    with tf.name_scope("loss"):
        loss = tf.reduce_mean(tf.square(y_ - y))
        tf.summary.scalar("loss", loss)

    # learning algorithm (learning rate: 0.0001)
    with tf.name_scope("train"):
        #train_step = tf.train.GradientDescentOptimizer(1e-4).minimize(loss)
        train_step = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False).minimize(loss)
    # -------------------------------------------------------------------------



    # mask index
    # if you analyze all areas, please set a white image
    indexH, indexW = get_masked_index("../image/mask.png")
    assert len(indexH) == len(indexW)

    # learning
    startTime = time.time()
    tf.global_variables_initializer().run() # initialize all variable
    saver = tf.train.Saver() # save weight
    ckpt = tf.train.get_checkpoint_state(modelPath) # model exist: True or False

    if ckpt:
        # ------------------------ CHECK ESTIMATION MODEL -------------------------
        lastModel = ckpt.model_checkpoint_path
        print("LODE: {}".format(lastModel))
        saver.restore(sess, lastModel)

        img = cv2.imread("../image/original/16_100920.png")
        label = np.load("../data/dens/25/16_100920.npy")
        maskedImg = get_masked_data(img)
        maskedLabel = get_masked_data(label)
        X_local, y_local = get_local_data(maskedImg, maskedLabel, 72, indexH, indexW)
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
        diffSquare = np.square(label - estDensMap, dtype="float32")
        estLoss = np.mean(diffSquare)
        print("estimation loss: {}".format(estLoss))
        # --------------------------------------------------------------------------

    else:
        # -------------------------- PRE PROCESSING --------------------------------
        # logs of tensor board directory
        date = datetime.now()
        dateDir = "{0}_{1}_{2}_{3}_{4}".format(date.year, date.month, date.day, date.hour, date.minute)
        logDir = "./logs_pixel/" + dateDir
        # delete the specified directory if it exists, recreate it
        if tf.gfile.Exists(logDir):
            tf.gfile.DeleteRecursively(logDir)
        tf.gfile.MakeDirs(logDir)

        # variable of TensorBoard
        trainStep = 0
        testStep = 0
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(logDir + "/train", sess.graph)
        test_writer = tf.summary.FileWriter(logDir + "/test")
        # --------------------------------------------------------------------------

        # -------------------------- LEARNING STEP --------------------------------
        n_epochs = 3
        batchSize = 100
        print("START: learning")
        print("Original traning data size: {}".format(len(X_train)))
        try:
            for epoch in range(n_epochs):
                print("elapsed time: {0:.3f} [sec]".format(time.time() - startTime))
                for i in range(len(X_train)):
                    X_train_local, y_train_local = get_local_data(X_train[i], y_train[i], 72, indexH, indexW)
                    X_train_local, y_train_local = under_sampling(X_train_local, y_train_local, thresh = 0.3)
                    X_train_local, y_train_local = shuffle(X_train_local, y_train_local)

                    train_n_batches = int(len(X_train_local) / batchSize)

                    for batch in range(train_n_batches):
                        trainStep += 1
                        startIndex = batch * batchSize
                        endIndex = startIndex + batchSize
                        #record loss data
                        if batch%100 == 0:
                            print("************************************************")
                            print("traning data: {0} / {1}".format(i, len(X_train)))
                            print("epoch: {0}, batch: {1} / {2}".format(epoch, batch, train_n_batches))
                            summary, train_loss = sess.run([merged, loss], feed_dict={
                                    X: X_train_local[startIndex:endIndex].reshape(-1, 72, 72, 3),
                                    y_: y_train_local[startIndex:endIndex].reshape(-1, 1),
                                    is_training:True})
                            train_writer.add_summary(summary, trainStep)
                            print("label mean: {}".format(np.mean(y_train_local[startIndex:endIndex])))
                            print("loss: {}".format(train_loss))
                            print("************************************************\n")

                        summary, _ = sess.run([merged, train_step], feed_dict={
                                            X: X_train_local[startIndex:endIndex].reshape(-1, 72, 72, 3),
                                            y_: y_train_local[startIndex:endIndex].reshape(-1, 1),
                                            is_training:True})
                        train_writer.add_summary(summary, trainStep)
            saver.save(sess, "./model_pixel/" + dateDir + "/model.ckpt")
            print("END: learning")
            # --------------------------------------------------------------------------


            # -------------------------------- TEST ------------------------------------
            print("START: test")
            test_loss = 0.0
            for i in range(len(X_test)):
                X_test_local, y_test_local = get_local_data(X_test[i], y_test[i], 72, indexH, indexW)
                X_test_local, y_test_local = under_sampling(X_test_local, y_test_local, thresh = 0)
                X_test_local, y_test_local = shuffle(X_test_local, y_test_local)
                test_n_batches = int(len(X_test_local) / batchSize)
                for batch in range(test_n_batches):
                    startIndex = batch * batchSize
                    endIndex = startIndex + batchSize

                    summary, tmp_loss = sess.run([merged, loss], feed_dict={
                                        X: X_test_local[startIndex:endIndex].reshape(-1, 72, 72, 3),
                                        y_: y_test_local[startIndex:endIndex].reshape(-1, 1),
                                        is_training:False})
                    test_writer.add_summary(summary, testStep)
                    test_loss += tmp_loss
                    testStep += 1

            print("test loss: {}\n".format(test_loss/(len(X_test)*test_n_batches)))
            print("END: test")
        except KeyboardInterrupt:
            #captured Ctrl + C
            print("\nPressed \"Ctrl + C\"")
            print("exit problem, save learning model")
            saver.save(sess, "./model_pixel/" + dateDir + "/model.ckpt")
            
        train_writer.close()
        test_writer.close()
        # --------------------------------------------------------------------------

    # --------------------------- END PROCESSING -------------------------------
    sess.close()
    # --------------------------------------------------------------------------

if __name__ == "__main__":
    inputImageDirPath = "/data/sakka/image/original/"
    inputDensDirPath = "/data/sakka/data/dens/25/"
    modelPath = "/data/sakka/tensor_model/2018/"
    X_train, X_test, y_train, y_test = load_data(inputImageDirPath, inputDensDirPath)
    main(X_train, X_test, y_train, y_test, modelPath)
