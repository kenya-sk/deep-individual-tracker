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


def load_data(input_image_dirc_path, input_dens_dirc_path, test_size=0.2):
    def get_file_path(input_dirc_path):
        try:
            file_lst = os.listdir(input_dirc_path)
        except FileNotFoundError:
            sys.stderr.write("Error: not found directory")
            sys.exit(1)
        pattern = r"^(?!._).*(.png)$"
        repattern = re.compile(pattern)
        file_lst = [name for name in file_lst if repattern.match(name)]
        return file_lst

    X = []
    y = []
    mask_path = "/data/sakka/image/mask.png" #引数で受け取るべき
    file_lst = get_file_path(input_image_dirc_path)
    if len(file_lst) == 0:
        sys.stderr.write("Error: not found input image")
        sys.exit(1)

    for path in file_lst:
        img = cv2.imread(input_image_dirc_path + path)
        if img is None:
            sys.stderr.write("Error: can not read image")
            sys.exit(1)
        else:
            X.append(get_masked_data(img, mask_path))
        dens_path = path.replace(".png", ".npy")
        dens_map = np.load(input_dens_dirc_path + dens_path)
        y.append(get_masked_data(dens_map, mask_path))
    X = np.array(X)
    y = np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    return X_train, X_test, y_train, y_test


def hard_negative_mining(X, y, loss):
    #get index that error is greater than the threshold
    def hard_negative_index(loss, thresh):
        index = np.where(loss > thresh)[0]
        return index

    # the threshold is five times the average
    thresh = np.mean(loss) * 3
    index = hard_negative_index(loss, thresh)
    hard_negative_image_arr = np.zeros((len(index), 72, 72, 3), dtype="uint8")
    hard_negative_label_arr = np.zeros((len(index)), dtype="float32")
    for i, hard_index in enumerate(index):
        hard_negative_image_arr[i] = X[hard_index]
        hard_negative_label_arr[i] = y[hard_index]
    return hard_negative_image_arr, hard_negative_label_arr


def under_sampling(local_img_mat, density_arr, thresh):
    """
    ret: undersampled (local_img_mat, density_arr)
    """

    def select(length, k):
        """
        ret: array of boolean which length = length and #True = k
        """
        seed = np.arange(length)
        np.random.shuffle(seed)
        return seed < k

    assert local_img_mat.shape[0] == len(density_arr)

    msk = density_arr >= thresh # select all positive samples first
    msk[~msk] = select((~msk).sum(), msk.sum()) # select same number of negative samples with positive samples
    return local_img_mat[msk], density_arr[msk]


def main(X_train, X_test, y_train, y_test, model_path):
    # start session
    config = tf.ConfigProto(gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5))
    sess = tf.InteractiveSession(config=config)

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
        diff = tf.square(y_ - y)
        loss = tf.reduce_mean(diff)
        tf.summary.scalar("loss", loss)

    # learning algorithm (learning rate: 0.0001)
    with tf.name_scope("train"):
        #train_step = tf.train.GradientDescentOptimizer(1e-4).minimize(loss)
        train_step = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False).minimize(loss)
    # -------------------------------------------------------------------------



    # mask index
    # if you analyze all areas, please set a white image
    index_h, index_w = get_masked_index("/data/sakka/image/mask.png")

    # learning
    start_time = time.time()
    saver = tf.train.Saver() # save weight
    ckpt = tf.train.get_checkpoint_state(model_path) # model exist: True or False

    # -------------------------- PRE PROCESSING --------------------------------
    # logs of tensor board directory
    date = datetime.now()
    date_dirc = "{0}_{1}_{2}_{3}_{4}".format(date.year, date.month, date.day, date.hour, date.minute)
    log_dirc = "/data/sakka/tensor_log/" + date_dirc

    # delete the specified directory if it exists, recreate it
    if tf.gfile.Exists(log_dirc):
        tf.gfile.DeleteRecursively(log_dirc)
    tf.gfile.MakeDirs(log_dirc)

    # variable of TensorBoard
    train_step = 0
    test_step = 0
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(log_dirc + "/train", sess.graph)
    test_writer = tf.summary.FileWriter(log_dirc + "/test")
    # --------------------------------------------------------------------------

    # -------------------------- LEARNING STEP --------------------------------
    n_epochs = 10
    batch_size = 100
    hard_negative_image_arr = np.zeros((1, 72, 72, 3), dtype="uint8")
    hard_negative_label_arr = np.zeros((1), dtype="float32")
    print("Original traning data size: {}".format(len(X_train)))
    # check if the ckpt exist
    # relearning or not
    if ckpt:
        last_model = ckpt.model_checkpoint_path
        print("START: Relearning")
        print("LODE: {}".format(last_model))
        saver.restore(sess, last_model)
    else:
        print("START: learning")
        # initialize all variable
        tf.global_variables_initializer().run()

    try:
        for epoch in range(n_epochs):
            print("elapsed time: {0:.3f} [sec]".format(time.time() - start_time))
            for i in range(len(X_train)):

                # load traing dataset
                X_train_local, y_train_local = get_local_data(X_train[i], y_train[i], index_h, index_w, local_img_size=72)
                X_train_local, y_train_local = under_sampling(X_train_local, y_train_local, thresh = 0)
                print("hard negative data: {}".format(hard_negative_label_arr.shape[0] - 1))
                if hard_negative_label_arr.shape[0] > 1:
                    X_train_local = np.append(X_train_local, hard_negative_image_arr[1:], axis=0)
                    y_train_local = np.append(y_train_local, hard_negative_label_arr[1:], axis=0)
                X_train_local, y_train_local = shuffle(X_train_local, y_train_local)

                # learning by batch
                hard_negative_image_arr = np.zeros((1, 72, 72, 3), dtype="uint8")
                hard_negative_label_arr = np.zeros((1), dtype="float32")
                train_n_batches = int(len(X_train_local) / batch_size)
                for batch in range(train_n_batches):
                    train_step += 1
                    start_index = batch * batch_size
                    end_index = start_index + batch_size

                    train_diff = sess.run(diff, feed_dict={
                            X: X_train_local[start_index:end_index].reshape(-1, 72, 72, 3),
                            y_: y_train_local[start_index:end_index].reshape(-1, 1),
                            is_training:True})
                    summary, _ = sess.run([merged, train_step], feed_dict={
                            X: X_train_local[start_index:end_index].reshape(-1, 72, 72, 3),
                            y_: y_train_local[start_index:end_index].reshape(-1, 1),
                            is_training:True})
                    train_writer.add_summary(summary, train_step)
                    # hard negative mining
                    batch_hard_negative_image_arr, batch_hard_negative_label_arr = \
                            hard_negative_mining(X_train_local[start_index:end_index], y_train_local[start_index:end_index], train_diff)
                    if batch_hard_negative_label_arr.shape[0] > 0: # there are hard negative data
                        hard_negative_image_arr = np.append(hard_negative_image_arr, batch_hard_negative_image_arr, axis=0)
                        hard_negative_label_arr = np.append(hard_negative_label_arr, batch_hard_negative_label_arr, axis=0)
                    else:
                        pass

                    #record loss data
                    if batch%100 == 0:
                        print("************************************************")
                        print("traning data: {0} / {1}".format(i, len(X_train)))
                        print("epoch: {0}, batch: {1} / {2}".format(epoch, batch, train_n_batches))
                        print("label mean: {}".format(np.mean(y_train_local[start_index:end_index])))
                        print("loss: {}".format(np.mean(train_diff)))
                        print("************************************************\n")

        saver.save(sess, "/data/sakka/tensor_model/" + date_dirc + "/model.ckpt")
        print("END: learning")
        # --------------------------------------------------------------------------


        # -------------------------------- TEST ------------------------------------
        print("START: test")
        test_loss = 0.0
        for i in range(len(X_test)):
            X_test_local, y_test_local = get_local_data(X_test[i], y_test[i], index_h, index_w, local_img_size=72)
            X_test_local, y_test_local = under_sampling(X_test_local, y_test_local, thresh = 0)
            X_test_local, y_test_local = shuffle(X_test_local, y_test_local)
            test_n_batches = int(len(X_test_local) / batch_size)
            for batch in range(test_n_batches):
                start_index = batch * batch_size
                end_index = start_index + batch_size

                summary, tmp_loss = sess.run([merged, loss], feed_dict={
                                    X: X_test_local[start_index:end_index].reshape(-1, 72, 72, 3),
                                    y_: y_test_local[start_index:end_index].reshape(-1, 1),
                                    is_training:False})
                test_writer.add_summary(summary, test_step)
                test_loss += tmp_loss
                test_step += 1

        print("test loss: {}\n".format(test_loss/(len(X_test)*test_n_batches)))
        print("END: test")

    # capture Ctrl + C
    except KeyboardInterrupt:
        print("\nPressed \"Ctrl + C\"")
        print("exit problem, save learning model")
        saver.save(sess, "/data/sakka/tensor_model/" + date_dirc + "/model.ckpt")

    train_writer.close()
    test_writer.close()
    # --------------------------------------------------------------------------

    # --------------------------- END PROCESSING -------------------------------
    sess.close()
    # --------------------------------------------------------------------------

if __name__ == "__main__":
    input_image_dirc_path = "/data/sakka/image/original/20170422/"
    input_dens_dirc_path = "/data/sakka/dens/20170422/"
    model_path = "/data/sakka/tensor_model/2018_4_15_15_7/"
    X_train, X_test, y_train, y_test = load_data(input_image_dirc_path, input_dens_dirc_path, test_size=0.2)
    main(X_train, X_test, y_train, y_test, model_path)
