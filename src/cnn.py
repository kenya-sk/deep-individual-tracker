#! /usr/bin/env python
#coding: utf-8

import time
import numpy as np
import tensorflow as tf


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


def main(x, y_):
    inputDim = 72*72*3
    outputDim = 18*18
    sess = tf.Interactive Session()
    sess.run(tf.global_variables_initializer())

    # input
    with tf.name_scope("X"):
        x = tf.placeholder(tf.float32, [None, inputDim])
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
        x_image = tf.reshape(x, [-1, 72, 72, 3])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

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
    n_epochs = 20000
    batch_size = 50
    n_batches = int(x.shape[0] / batch_size)
    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            print("epoch: {0}".format(i))
            print("elapsed time: {0:.3f} [sec]".format(time.time() - startTime))
            print("loss: {0}".format(loss))
        for i in range(n_batches):
            startIndex = i * batch_size
            endIndex = startIndex + batch_size
        train_step.run(feed_dict={x: x[startIndex:endIndex], y_: y_[startIndex:endIndex]})

    sess.close()

if __name__ == "__main__":
    # tmp variable (NOT WORK)
    main(x, y_)
