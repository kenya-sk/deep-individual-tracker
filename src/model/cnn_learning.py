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

from cnn_util import get_masked_data, get_masked_index, get_local_data
from cnn_model import CNN_model

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
    cnn_model = CNN_model()

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
                X_train_local, y_train_local = under_sampling(X_train_local, y_train_local, thresh = 0.2)
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

                    train_diff = sess.run(cnn_model.diff, feed_dict={
                            cnn_model.X: X_train_local[start_index:end_index].reshape(-1, 72, 72, 3),
                            cnn_model.y_: y_train_local[start_index:end_index].reshape(-1, 1),
                            cnn_model.is_training: True})

                    train_summary, _ = sess.run([merged, cnn_model.learning_step],feed_dict={
                            cnn_model.X: X_train_local[start_index:end_index].reshape(-1, 72, 72, 3),
                            cnn_model.y_: y_train_local[start_index:end_index].reshape(-1, 1),
                            cnn_model.is_training: True})
                    train_writer.add_summary(train_summary, train_step)

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
                test_step += 1
                start_index = batch * batch_size
                end_index = start_index + batch_size

                test_summary, tmp_loss = sess.run([merged, cnn_model.loss], feed_dict={
                                    X: X_test_local[start_index:end_index].reshape(-1, 72, 72, 3),
                                    y_: y_test_local[start_index:end_index].reshape(-1, 1),
                                    is_training:False})
                test_writer.add_summary(test_summary, test_step)
                test_loss += tmp_loss

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
