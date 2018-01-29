#! /usr/bin/env python
# coding: utf-8

import numpy as np
import tensorflow as tf
import cv2

import cnn


def main():
    X = tf.placeholder(tf.float32, [None, 72, 72, 3])
    y_ = tf.placeholder(tf.float32, [None])

    #model
    #layer1
    W_conv1 = tf.get_variable("conv1/weight1/weight1", [7,7,3,32])
    b_conv1 = tf.get_variable("conv1/bias1/bias1", [32])
    h_conv1 = tf.nn.relu(cnn.conv2d(X, W_conv1) + b_conv1)
    h_pool1 = cnn.max_pool_2x2(h_conv1)
    #layer2
    W_conv2 = tf.get_variable("conv2/weight2/weight2", [7,7,32,32])
    b_conv2 = tf.get_variable("conv2/bias2/bias2", [32])
    h_conv2 = tf.nn.relu(cnn.conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = cnn.max_pool_2x2(h_conv2)
    #layer3
    W_conv3 = tf.get_variable("conv3/weight3/weight3", [5,5,32,64])
    b_conv3 = tf.get_variable("conv3/bias3/bias3", [64])
    h_conv3 = tf.nn.relu(cnn.conv2d(h_pool2, W_conv3) + b_conv3)
    #layer4
    W_fc4 = tf.get_variable("fc4/weight4/weight4", [18*18*64, 1000])
    b_fc4 = tf.get_variable("fc4/bias4/bias4", [1000])
    h_conv3_flat = tf.reshape(h_conv3, [-1, 18*18*64])
    h_fc4 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc4) + b_fc4)
    #layer5
    W_fc5 = tf.get_variable("fc5/weight5/weight5", [1000, 400])
    b_fc5 = tf.get_variable("fc5/bias5/bias5", [400])
    h_fc5 = tf.nn.relu(tf.matmul(h_fc4, W_fc5) + b_fc5)
    #layer6
    W_fc6 = tf.get_variable("fc6/weight6/weight6", [400, 324])
    b_fc6 = tf.get_variable("fc6/bias6/bias6", [324])
    h_fc6 = tf.nn.relu(tf.matmul(h_fc5, W_fc6) + b_fc6)
    #layer7
    W_fc7 = tf.get_variable("fc7/weight7/weight7", [324, 1])
    b_fc7 = tf.get_variable("fc7/bias7/bias7", [1])
    h_fc7 = tf.nn.relu(tf.matmul(h_fc6, W_fc7) + b_fc7)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, "./model_pixel/model.ckpt")

        test = cv2.imread("../image/original/tmp/90.png")
        test_lst = cnn.get_local_image(test, 72, False)
        answer = np.load("../data/dens/tmp/90.npy")
        answer_lst = cnn.get_local_image(answer, 72, True)

        testLoss = 0.0
        for i in range(len(test_lst)):
            testLoss += sess.run(loss, feed_dict={
                                        X: test_lst[i].reshape(1, 72, 72, 3),
                                        y_: np.max(answer_lst[i]).reshape(1)})
        print(testLoss/ len(test_lst))


if __name__ == "__main__":
    main()
