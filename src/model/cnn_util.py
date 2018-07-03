#! /usr/bin/env python
# coding: utf-8

import math
import cv2
import numpy as np
import tensorflow as tf


ANALYSIS_HEIGHT = (0, 470)
ANALYSIS_WIDTH = (0, 1280)


def get_masked_data(data, mask_path):
    """
    input:
        data: image or density map
        mask_path: binary mask path

    output:
        masked image or density map
    """

    # mask: 3channel mask image. the value is 0 or 1
    mask = cv2.imread(mask_path)
    if mask is None:
        sys.stderr.write("Error: can not read mask image")
        sys.exit(1)

    if len(data.shape) == 3:
        masked_data = data*mask
    else:
        masked_data = mask[:,:,0]*data
    return masked_data


def get_masked_index(mask_path=None):
    """
    input:
        mask_path: binay mask path

    output:
        valid index list (heiht and width)
    """

    if mask_path is None:
         mask = np.ones((720, 1280))
    else:
        mask = cv2.imread(mask_path)

    if mask.shape[2] == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    index = np.where(mask > 0)
    index_h = index[0]
    index_w = index[1]
    assert len(index_h) == len(index_w)
    return index_h, index_w


def get_local_data(img, dens_map, index_h, index_w, local_img_size=72):
    """
    input:
        img: original image
        dens_map: answer label
        index_h: valid height index (returned get_masked_index)
        index_w: valid width index (returned get_masked_index)
        local_img_size: square local image size (default: 72 pixel)

    output:
        localImg_mat: ([#locals, local_img_size, local_img_size, img.shape[2]])
        density_arr: ([#locals])
    """

    assert len(img.shape) == 3
    # trim original image
    img = img[ANALYSIS_HEIGHT[0]:ANALYSIS_HEIGHT[1], ANALYSIS_WIDTH[0]:ANALYSIS_WIDTH[1]]
    height = img.shape[0]
    width = img.shape[1]

    pad = math.floor(local_img_size/2)
    pad_img = np.zeros((height + pad * 2, width + pad * 2, img.shape[2]), dtype="uint8")
    pad_img[pad:height+pad, pad:width+pad] = img

    local_img_mat = np.zeros((len(index_w), local_img_size, local_img_size, img.shape[2]), dtype="uint8")
    density_arr = np.zeros((len(index_w)), dtype="float32")
    for idx in range(len(index_w)):
        # fix index(pad_img)
        h = index_h[idx]
        w = index_w[idx]
        local_img_mat[idx] = pad_img[h:h+2*pad,w:w+2*pad]
        density_arr[idx] = dens_map[h, w]
    return local_img_mat, density_arr


# def variable_summaries(var):
#     """
#     processing variables and it output tensorboard
#
#     input:
#         var: value of several layer
#
#     output:
#         mean, stddev, max, min, histogram
#     """
#
#     with tf.name_scope('summaries'):
#         mean = tf.reduce_mean(var)
#         tf.summary.scalar('mean', mean)
#         with tf.name_scope('stddev'):
#             stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
#         tf.summary.scalar('stddev', stddev)
#         tf.summary.scalar('max', tf.reduce_max(var))
#         tf.summary.scalar('min', tf.reduce_min(var))


def weight_variable(shape, name=None):
    """
    initialize weight by He initialization

    input:
        shape: size of weight filter
        name: variable name

    output:
        weight filter
    """

    # He initialization
    if len(shape) == 4:
        #convolution layer
        n = shape[1] * shape[2] * shape[3]
    elif len(shape) == 2:
        # fully conected layer
        n = shape[0]
    else:
        sys.stderr.write("Error: shape is not correct !")
        sys.exit(1)
    stddev = math.sqrt(2/n)
    initial = tf.random_normal(shape, stddev=stddev, dtype=tf.float32)
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.Variable(initial, name=name)


def bias_variable(shape, name=None):
    """
    initialize bias by normal distribution (standard deviation: 0.1)

    input:
        shape: size of bias
        name: variable name

    output:
        bias
    """

    initial = tf.constant(0.1, shape=shape)
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.Variable(initial, name=name)


def conv2d(x, W):
    """
    2d convolutional layer

    input:
        x: input value
        W: weight

    output:
        convolved value
    """

    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")


def max_pool_2x2(x):
    """
    2x2 maximum pooling layer

    input:
        x: input value

    output:
        pooled value
    """

    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


def batch_norm(X, axes, shape, is_training):
    """
    batch normalization

    input:
        X: input value
        axes: order of dimension
        shape: chanel number
        is_training: True or False

    output:
        batch normalized value
    """

    if is_training is False:
        return X
    epsilon  = 1e-5
    mean, variance = tf.nn.moments(X, axes)
    scale = tf.Variable(tf.ones([shape]))
    offset = tf.Variable(tf.zeros([shape]))
    return tf.nn.batch_normalization(X, mean, variance, offset, scale, epsilon)
