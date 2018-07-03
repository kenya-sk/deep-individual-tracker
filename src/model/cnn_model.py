#! /usr/bin/env python
#coding: utf-8

import os
import re
import time
import sys
import cv2
import math
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from cnn_util import *

ANALYSIS_HEIGHT = (0, 470)
ANALYSIS_WIDTH = (0, 1280)

class CNN_model(object):
    def __init__(self):
        # input image
        with tf.name_scope("input"):
            with tf.name_scope("X"):
                self.X = tf.placeholder(tf.float32, [None, 72, 72, 3], name="input")
                _ = tf.summary.image("X", self.X[:, :, :, :], 5)
            # answer image
            with tf.name_scope("y_"):
                self.y_ = tf.placeholder(tf.float32, [None, 1], name="label")
            # status: True(lerning) or False(test)
            with tf.name_scope("is_training"):
                self.is_training = tf.placeholder(tf.bool, name="is_training")

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
                conv1 = conv2d(self.X, W_conv1)
                conv1_bn = batch_norm(conv1, [0, 1, 2], 32, self.is_training)
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
                conv2_bn = batch_norm(conv2, [0, 1, 2], 32, self.is_training)
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
                conv3_bn = batch_norm(conv3, [0, 1, 2], 64, self.is_training)
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
                fc4_bn = batch_norm(fc4, [0], 1000, self.is_training)
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
                fc5_bn = batch_norm(fc5, [0], 400, self.is_training)
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
                fc6_bn = batch_norm(fc6, [0], 324, self.is_training)
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
            self.diff = tf.square(self.y_ - y)
            self.loss = tf.reduce_mean(self.diff)
            tf.summary.scalar("loss", self.loss)

        # learning algorithm (learning rate: 0.0001)
        with tf.name_scope("train"):
            #train_step = tf.train.GradientDescentOptimizer(1e-4).minimize(loss)
            self.learning_step = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False).minimize(self.loss)
