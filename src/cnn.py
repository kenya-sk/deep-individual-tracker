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
    # local image size is even number
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
            tmpLocalImg = padImg[h-pad:h+pad, w-pad:w+pad]
            if resize == True:
                # resize answer data and flat
                tmpLocalImg = cv2.resize(tmpLocalImg, (18, 18))
                tmpLocalImg = np.reshape(tmpLocalImg, -1)
            localImg_lst.append(tmpLocalImg)
    return localImg_lst

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
        #tf.summary.histogram('histogram', var)

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
    # delete the specified directory if it exists, recreate it
    log_dir = "./logs"
    if tf.gfile.Exists(log_dir):
        tf.gfile.DeleteRecursively(log_dir)
    tf.gfile.MakeDirs(log_dir)

    # start session
    sess = tf.InteractiveSession()

    with tf.name_scope("input"):
        # input image
        with tf.name_scope("X"):
            X = tf.placeholder(tf.float32, [None, 72, 72, 3])
        # answer image
        with tf.name_scope("y_"):
            y_ = tf.placeholder(tf.float32, [None, 18*18])


    # first layer
    # convlution -> ReLU -> max pooling
    # input 72x72x3 -> output 36x36x32
    with tf.name_scope("layer1"):
        with tf.name_scope("conv1"):
            #7x7x3 filter
            with tf.name_scope("weight1"):
                W_conv1 = weight_variable([7,7,3,32])
                variable_summaries(W_conv1)
                _ = tf.summary.image("image1", tf.transpose(W_conv1, perm=[3, 0, 1, 2])[:,:,:,0:1], max_outputs=10)
            with tf.name_scope("biases1"):
                b_conv1 = bias_variable([32])
                variable_summaries(b_conv1)
            with tf.name_scope("relu1"):
                h_conv1 = tf.nn.relu(conv2d(X, W_conv1) + b_conv1)
                variable_summaries(h_conv1)

        with tf.name_scope("pool1"):
            h_pool1 = max_pool_2x2(h_conv1)
            variable_summaries(h_pool1)


    # second layer
    # convlution -> ReLU -> max pooling
    # input 36x36x32 -> output 18x18x32
    with tf.name_scope("layer2"):
        with tf.name_scope("conv2"):
            # 7x7x32 filter
            with tf.name_scope("weights2"):
                W_conv2 = weight_variable([7,7,32,32])
                variable_summaries(W_conv2)
                _ = tf.summary.image("image2", tf.transpose(W_conv2, perm=[3, 0, 1, 2])[:,:,:,0:1], max_outputs=10)
            with tf.name_scope("biass2"):
                b_conv2 = bias_variable([32])
                variable_summaries(b_conv2)
            with tf.name_scope("relu2"):
                h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
                variable_summaries(h_conv2)

        with tf.name_scope("pool2"):
            h_pool2 = max_pool_2x2(h_conv2)
            variable_summaries(h_pool2)

    # third layer
    # convolution -> ReLU
    # input 18x18x32 -> output 18x18x64
    with tf.name_scope("layer3"):
        with tf.name_scope("conv3"):
            # 5x5x32 filter
            with tf.name_scope("weight3"):
                W_conv3 = weight_variable([5,5,32,64])
                variable_summaries(W_conv3)
                _ = tf.summary.image("image3", tf.transpose(W_conv3, perm=[3, 0, 1, 2])[:,:,:,0:1], max_outputs=10)
            with tf.name_scope("biass3"):
                b_conv3 = bias_variable([64])
                variable_summaries(b_conv3)
            with tf.name_scope("relu3"):
                h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
                variable_summaries(h_conv3)

    # fourth layer
    # fully connected layer
    # input 18x18x64 -> output 1000
    with tf.name_scope("layer4"):
        with tf.name_scope("fc4"):
            with tf.name_scope("weight4"):
                W_fc4 = weight_variable([18*18*64, 1000])
                variable_summaries(W_fc4)
            with tf.name_scope("biass4"):
                b_fc4 = bias_variable([1000])
                variable_summaries(b_fc4)
            with tf.name_scope("flat4"):
                h_conv3_flat = tf.reshape(h_conv3, [-1, 18*18*64])
                h_fc4 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc4) + b_fc4)
                variable_summaries(h_fc4)

    # fifth layer
    # fully connected layer
    # input 1000 -> output 400
    with tf.name_scope("layer5"):
        with tf.name_scope("fc5"):
            with tf.name_scope("weight5"):
                W_fc5 = weight_variable([1000, 400])
                variable_summaries(W_fc5)
            with tf.name_scope("biass5"):
                b_fc5 = bias_variable([400])
                variable_summaries(b_fc5)
            with tf.name_scope("flat5"):
                h_fc5 = tf.nn.relu(tf.matmul(h_fc4, W_fc5) + b_fc5)
                variable_summaries(h_fc5)

    # sixth layer
    # fully connected layer
    # input 400 -> output 324
    with tf.name_scope("layer6"):
        with tf.name_scope("fc6"):
            with tf.name_scope("weight6"):
                W_fc6 = weight_variable([400, 324])
                variable_summaries(W_fc6)
            with tf.name_scope("biass6"):
                b_fc6 = bias_variable([324])
                variable_summaries(b_fc6)
            with tf.name_scope("flat6"):
                h_fc6 = tf.nn.relu(tf.matmul(h_fc5, W_fc6) + b_fc6)
                variable_summaries(h_fc6)

    with tf.name_scope("y"):
        h_fc6_flat = tf.reshape(h_fc6, [-1, 18*18])
        #tf.summary.histogram("y", h_fc6_flat)

    # loss function
    with tf.name_scope("loss"):
        loss = tf.reduce_mean(tf.square(y_ - h_fc6_flat))
        tf.summary.scalar("loss", loss)

    # learning algorithm (learning rate: 0.01)
    with tf.name_scope("train"):
        train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

    # variable of TensorBoard
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(log_dir + "/train", sess.graph)
    test_writer = tf.summary.FileWriter(log_dir + "/test")

    # learning
    startTime = time.time()
    n_steps = 10
    batchSize = 5
    tf.global_variables_initializer().run() # initialize all variable
    saver = tf.train.Saver() # save weight


    print("Original traning data size: {}".format(len(X_train)))
    for step in range(n_steps):
        print("elapsed time: {0:.3f} [sec]".format(time.time() - startTime))
        for i in range(len(X_train)):
            X_train_local = get_local_image(X_train[i], 72, False)
            y_train_local = get_local_image(y_train[i], 72, True)
            n_batches = int(len(X_train_local) / batchSize)

            for batch in range(n_batches):
                startIndex = batch * batchSize
                endIndex = startIndex + batchSize
                if batch%(n_batches) == 0:
                    print("traning data: {0} / {1}".format(i, len(X_train)))
                    print("step: {0}, batch: {1} / {2}".format(step, batch, n_batches))
                    summary, train_loss = sess.run([merged, loss], feed_dict={
                            X: X_train_local[startIndex:endIndex],
                            y_: y_train_local[startIndex:endIndex]})
                    train_writer.add_summary(summary, i)
                    print("loss: {}\n".format(train_loss))
                    """
                    # output detail data
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                    summary, _ = sess.run([merged, train_step], feed_dict={
                                    X: X_train_local[startIndex:endIndex],
                                    y_: y_train_local[startIndex:endIndex]},
                                    options=run_options,
                                    run_metadata=run_metadata)
                    train_writer.add_run_metadata(run_metadata, "batch{0:04d}".format(i))
                    train_writer.add_summary(summary, i)
                    """
                #else:
                summary, _ = sess.run([merged, train_step], feed_dict={
                                    X: X_train_local[startIndex:endIndex],
                                    y_: y_train_local[startIndex:endIndex]})
                train_writer.add_summary(summary, i)

    saver.save(sess, "./model/model.ckpt")

    # test data
    test_loss = 0.0
    for i in range(len(X_test)):
        X_test_local = get_local_image(X_test[0], 72, False)
        y_test_local = get_local_image(y_test[0], 72, True)
        summary, tmp_loss = sess.run([merged, loss], feed_dict={X: X_test_local, y_: y_test_local})
        test_writer.add_summary(summary, i)
        test_loss += tmp_loss
    print("test accuracy {}".format(test_loss/len(X_test)))

    # end processing
    train_writer.close()
    test_writer.close()
    sess.close()

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data("../image/original/tmp")
    main(X_train, X_test, y_train, y_test)
