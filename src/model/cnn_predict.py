#! /usr/bin/env python
# coding: utf-8

import math
import sys
import cv2
import glob
import time
import numpy as np
import tensorflow as tf

from cnn_util import *


def cnn_predict(model_path, input_img_path, output_dirc_path, params_dict):
    # start session
    config = tf.ConfigProto(gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9))
    sess = tf.InteractiveSession(config=config)

    # ------------------------------- MODEL ------------------------------------
    # input image
    X = tf.placeholder(tf.float32, [None, 72, 72, 3], name="input")
    # answer image
    y_ = tf.placeholder(tf.float32, [None, 1], name="label")
    # status: True(lerning) or False(test)
    is_training = tf.placeholder(tf.bool, name="is_training")


    # first layer
    # convlution -> ReLU -> max pooling
    # input 72x72x3 -> output 36x36x32
    with tf.name_scope("conv1"):
        # 7x7x3 filter
        with tf.name_scope("weight1"):
            W_conv1 = weight_variable([7,7,3,32])
        with tf.name_scope("batchNorm1"):
            conv1 = conv2d(X, W_conv1)
            conv1_bn = batch_norm(conv1, [0, 1, 2], 32, is_training)
        with tf.name_scope("leakyRelu1"):
            h_conv1 = tf.nn.leaky_relu(conv1_bn)

    with tf.name_scope("pool1"):
        h_pool1 = max_pool_2x2(h_conv1)

    # second layer
    # convlution -> ReLU -> max pooling
    # input 36x36x32 -> output 18x18x32
    with tf.name_scope("conv2"):
        # 7x7x32 filter
        with tf.name_scope("weight2"):
            W_conv2 = weight_variable([7,7,32,32])
        with tf.name_scope("batchNorm2"):
            conv2 = conv2d(h_pool1, W_conv2)
            conv2_bn = batch_norm(conv2, [0, 1, 2], 32, is_training)
        with tf.name_scope("leakyRelu2"):
            h_conv2 = tf.nn.leaky_relu(conv2_bn)

    with tf.name_scope("pool2"):
        h_pool2 = max_pool_2x2(h_conv2)

    # third layer
    # convolution -> ReLU
    # input 18x18x32 -> output 18x18x64
    with tf.name_scope("conv3"):
        # 5x5x32 filter
        with tf.name_scope("weight3"):
            W_conv3 = weight_variable([5,5,32,64])
            variable_summaries(W_conv3)
        with tf.name_scope("batchNorm3"):
            conv3 = conv2d(h_pool2, W_conv3)
            conv3_bn = batch_norm(conv3, [0, 1, 2], 64, is_training)
        with tf.name_scope("leakyRelu3"):
            h_conv3 = tf.nn.leaky_relu(conv3_bn)

    # fourth layer
    # fully connected layer
    # input 18x18x64 -> output 1000
    with tf.name_scope("fc4"):
        with tf.name_scope("weight4"):
            W_fc4 = weight_variable([18*18*64, 1000])
        with tf.name_scope("batchNorm4"):
            h_conv3_flat = tf.reshape(h_conv3, [-1, 18*18*64])
            fc4 = tf.matmul(h_conv3_flat, W_fc4)
            fc4_bn = batch_norm(fc4, [0], 1000, is_training)
        with tf.name_scope("flat4"):
            h_fc4 = tf.nn.leaky_relu(fc4_bn)

    # fifth layer
    # fully connected layer
    # input 1000 -> output 400
    with tf.name_scope("fc5"):
        with tf.name_scope("weight5"):
            W_fc5 = weight_variable([1000, 400])
        with tf.name_scope("batchNorm5"):
            fc5 = tf.matmul(h_fc4, W_fc5)
            fc5_bn = batch_norm(fc5, [0], 400, is_training)
        with tf.name_scope("flat5"):
            h_fc5 = tf.nn.leaky_relu(fc5_bn)

    # sixth layer
    # fully connected layer
    # input 400 -> output 324
    with tf.name_scope("fc6"):
        with tf.name_scope("weight6"):
            W_fc6 = weight_variable([400, 324])
        with tf.name_scope("batchNorm6"):
            fc6 = tf.matmul(h_fc5, W_fc6)
            fc6_bn = batch_norm(fc6, [0], 324, is_training)
        with tf.name_scope("flat6"):
            h_fc6 = tf.nn.leaky_relu(fc6_bn)

    # seven layer
    # fully connected layer
    # input 324 -> output 1
    with tf.name_scope("fc7"):
        with tf.name_scope("weight7"):
            W_fc7 = weight_variable([324, 1])
        with tf.name_scope("bias7"):
            b_fc7 = bias_variable([1])
        with tf.name_scope("flat7"):
            y = tf.nn.leaky_relu(tf.matmul(h_fc6, W_fc7) + b_fc7)

    #---------------------------------------------------------------------------

    # ------------------------------- PREDICT ----------------------------------
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(model_path)
    if ckpt:
        last_model = ckpt.model_checkpoint_path
        print("LODE: {}".format(last_model))
        saver.restore(sess, last_model)
    else:
        sys.stderr("Eroor: Not exist model!")
        sys.stderr("Please check model_path")
        sys.exit(1)

    skip_width = params_dict["skip_width"]
    img_file_lst = glob.glob(input_img_path)
    mask_path = "/data/sakka/image/mask.png"
    index_h, index_w = get_masked_index(mask_path)
    est_start_time = time.time()

    for img_path in img_file_lst:
        img = cv2.imread(img_path)
        label = np.zeros((720, 1280))
        masked_img = get_masked_data(img, mask_path)
        masked_label = get_masked_data(label, mask_path)
        X_local, y_local = get_local_data(masked_img, masked_label, index_h, index_w, local_img_size=72)

        # local image index
        index_lst = []
        for step in range(len(index_h)):
            if step%skip_width == 0:
                index_lst.append(step)

        est_batch_size = params_dict["est_batch_size"]
        est_n_batches = int(len(index_lst)/est_batch_size)
        est_arr = np.zeros(est_batch_size)
        est_dens_map = np.zeros((720,1280), dtype="float32")

        print("STSRT: estimate density map")
        for batch in range(est_n_batches):
            # array of skiped local image
            X_skip = np.zeros((est_batch_size,72,72,3))
            y_skip = np.zeros((est_batch_size,1))
            for index_cord,index_local in enumerate(range(est_batch_size)):
                current_index = index_lst[batch*est_batch_size+index_local]
                X_skip[index_cord] = X_local[current_index]
                y_skip[index_cord] = y_local[current_index]

            # esimate each local image
            est_arr = sess.run(y, feed_dict={
                                        X: X_skip,
                                        y_: y_skip,
                                        is_training: False}).reshape(est_batch_size)
            print("DONE: batch {}".format(batch))

            for i in range(est_batch_size):
                h_est = index_h[index_lst[batch*est_batch_size+i]]
                w_est = index_w[index_lst[batch*est_batch_size+i]]
                est_dens_map[h_est,w_est] = est_arr[i]

        np.save(output_dirc_path + "{}/{}.npy".format(skip_width, outfile_path), est_dens_map)
        print("END: estimate density map")

        # calculate estimation loss
        est_loss = np.mean(np.square(label - est_dens_map), dtype="float32")
        print("estimation loss: {}".format(est_loss))

    #---------------------------------------------------------------------------

    with open(output_dirc_path + "{}/time.txt".format(skip), "a") as f:
        f.write("skip: {0}, frame num: {1} total time: {2}\n".format(skip_width, 35,time.time() - est_start_time)) # modify: division num


if __name__ == "__main__":
    model_path = "/data/sakka/tensor_model/2018_4_15_15_7/"
    input_img_path = "/data/sakka/image/1h_10/*.png"
    output_dirc_path = "/data/sakka/estimation/1h_10/model_201806142123/dens/"
    params_dict = {"skip_width": 15, "est_batch_size": 2500}
    cnn_predict(model_path, input_img_path, output_dirc_path, params_dict)
