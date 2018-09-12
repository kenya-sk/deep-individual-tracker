#! /usr/bin/env python
#coding: utf-8

import os
import time
import sys
import cv2
import math
import glob
from tqdm import trange
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


def load_data(input_image_dirc_path, input_dens_dirc_path, mask_path, test_size=0.2):
    X = []
    y = []
    file_lst = glob.glob(input_image_dirc_path + "*.png")
    if len(file_lst) == 0:
        sys.stderr.write("Error: not found input image")
        sys.exit(1)

    for path in file_lst:
        img = cv2.imread(path)
        if img is None:
            sys.stderr.write("Error: can not read image")
            sys.exit(1)
        else:
            X.append(get_masked_data(img, mask_path))
        dens_path = path.replace(".png", ".npy").split("/")[-1]
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


def cnn_learning(X_train, X_test, y_train, y_test, mask_path, reuse_model_path, out_model_dirc):
    # -------------------------- PRE PROCESSING --------------------------------
    # start session
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.9))
    sess = tf.InteractiveSession(config=config)
    start_time = time.time()
    cnn_model = CNN_model()

    # mask index
    # if you analyze all areas, please set a white image
    index_h, index_w = get_masked_index(mask_path, horizontal_flip=False)
    flip_index_h, flip_index_w = get_masked_index(mask_path, horizontal_flip=True)

    # logs of tensor board directory
    date = datetime.now()
    learning_date = "{0}_{1}_{2}_{3}_{4}".format(date.year, date.month, date.day, date.hour, date.minute)
    log_dirc = "/data/sakka/tensor_log/" + learning_date

    # delete the specified directory if it exists, recreate it
    if tf.gfile.Exists(log_dirc):
        tf.gfile.DeleteRecursively(log_dirc)
    tf.gfile.MakeDirs(log_dirc)

    # variable of TensorBoard
    train_step = 0
    test_step = 0
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(log_dirc + "/train", sess.graph)
    val_writer = tf.summary.FileWriter(log_dir + "/val")
    test_writer = tf.summary.FileWriter(log_dirc + "/test")

    # split data: validation or test
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5)
    # --------------------------------------------------------------------------

    # -------------------------- LEARNING STEP --------------------------------
    n_epochs = 30
    batch_size = 500
    hard_negative_image_arr = np.zeros((1, 72, 72, 3), dtype="uint8")
    hard_negative_label_arr = np.zeros((1), dtype="float32")
    val_loss_lst = []
    minimum_epoch = 5
    not_improved_count = 0
    early_stopping_epoch = 2
    print("Original traning data size: {}".format(len(X_train)))

    # check if the ckpt exist
    # relearning or not
    saver = tf.train.Saver() # save weight
    ckpt = tf.train.get_checkpoint_state(reuse_model_path) # model exist: True or False
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
            print("************************************************")
            print("elapsed time: {0:.3f} [sec]".format(time.time() - start_time))
            train_loss = 0.0
            for train_i in trange(len(X_train), desc="training data"):
                # load traing dataset
                # data augmentation (horizontal flip)
                flip_prob = 0.5 # probality of horizontal flip
                if np.random.rand() < flip_prob:
                    X_train_local, y_train_local = get_local_data(X_train[train_i][:,::-1,:], y_train[train_i][:,::-1], flip_index_h, flip_index_w, local_img_size=72)
                else:
                    X_train_local, y_train_local = get_local_data(X_train[train_i], y_train[train_i], index_h, index_w, local_img_size=72)
                X_train_local, y_train_local = under_sampling(X_train_local, y_train_local, thresh = 0.2)

                #print("hard negative data (training {}/{}): {}".format(train_i, len(X_train), hard_negative_label_arr.shape[0] - 1))
                if hard_negative_label_arr.shape[0] > 1:
                    X_train_local = np.append(X_train_local, hard_negative_image_arr[1:], axis=0)
                    y_train_local = np.append(y_train_local, hard_negative_label_arr[1:], axis=0)
                X_train_local, y_train_local = shuffle(X_train_local, y_train_local)

                # learning by batch
                hard_negative_image_arr = np.zeros((1, 72, 72, 3), dtype="uint8")
                hard_negative_label_arr = np.zeros((1), dtype="float32")
                train_n_batches = int(len(X_train_local) / batch_size)
                for train_batch in range(train_n_batches):
                    train_step += 1
                    train_start_index = train_batch * batch_size
                    train_end_index = train_start_index + batch_size

                    train_diff = sess.run(cnn_model.diff, feed_dict={
                            cnn_model.X: X_train_local[train_start_index:train_end_index].reshape(-1, 72, 72, 3),
                            cnn_model.y_: y_train_local[train_start_index:train_end_index].reshape(-1, 1),
                            cnn_model.is_training: True})

                    train_loss += np.mean(train_diff)

                    train_summary, _ = sess.run([merged, cnn_model.learning_step],feed_dict={
                            cnn_model.X: X_train_local[train_start_index:train_end_index].reshape(-1, 72, 72, 3),
                            cnn_model.y_: y_train_local[train_start_index:train_end_index].reshape(-1, 1),
                            cnn_model.is_training: True})

                    # hard negative mining
                    batch_hard_negative_image_arr, batch_hard_negative_label_arr = \
                            hard_negative_mining(X_train_local[train_start_index:train_end_index], y_train_local[train_start_index:train_end_index], train_diff)
                    if batch_hard_negative_label_arr.shape[0] > 0: # there are hard negative data
                        hard_negative_image_arr = np.append(hard_negative_image_arr, batch_hard_negative_image_arr, axis=0)
                        hard_negative_label_arr = np.append(hard_negative_label_arr, batch_hard_negative_label_arr, axis=0)
                    else:
                        pass

            train_writer.add_summary(train_summary, epoch)

            # validation
            val_loss = 0.0
            for val_i in range(len(X_val)):
                X_val_local, y_val_local = get_local_data(X_val[val_i], y_val[val_i], index_h, index_w, local_img_size=72)
                val_n_batches = int(len(X_val_local) / batch_size)
                for val_batch in range(val_n_batches):
                    val_start_index = val_batch * batch_size
                    val_end_index = val_start_index + batch_size

                    tmp_val_loss = sess.run(cnn_model.loss, feed_dict={
                                        cnn_model.X: X_val_local[val_start_index:val_end_index].reshape(-1, 72, 72, 3),
                                        cnn_model.y_: y_val_local[val_start_index:val_end_index].reshape(-1, 1),
                                        cnn_model.is_training:False})
                    val_loss += tmp_val_loss
            val_loss_lst.append(val_loss/(len(X_val)*val_n_batches))

            #record loss data
            val_writer.add_summary(val_loss_lst[-1], train_step)
            print("epoch: {0}".format(epoch+1))
            print("train loss: {}".format(train_loss/(len(X_train)*train_n_batches)))
            print("validation loss: {}".format(val_loss_lst[epoch]))
        

            # early stopping
            if epoch > minimum_epoch:
                best_epoch_loss = np.argmin(val_loss_lst[:-1])
                if val_loss_lst[epoch] <= val_loss_lst[best_epoch_loss]:
                    # learning is going well
                    not_improved_count = 0
                else:
                    not_improved_count += 1
                
            if not_improved_count >= early_stopping_epoch:
                print("Early stopping due to no improvement after {} epochs.".format(early_stopping_epoch))
                break

            print("not improved count/early stopping epoch: {}/{}".format(not_improved_count, early_stopping_epoch))
            print("************************************************\n")


        saver.save(sess, out_model_dirc + learning_date + "/model.ckpt")
        print("END: learning")
        # --------------------------------------------------------------------------


        # -------------------------------- TEST ------------------------------------
        print("START: test")
        test_loss = 0.0
        for test_i in range(len(X_test)):
            X_test_local, y_test_local = get_local_data(X_test[test_i], y_test[test_i], index_h, index_w, local_img_size=72)
            test_n_batches = int(len(X_test_local) / batch_size)
            for test_batch in range(test_n_batches):
                test_step += 1
                test_start_index = test_batch * batch_size
                test_end_index = test_start_index + batch_size

                test_summary, tmp_test_loss = sess.run([merged, cnn_model.loss], feed_dict={
                                    cnn_model.X: X_test_local[test_start_index:test_end_index].reshape(-1, 72, 72, 3),
                                    cnn_model.y_: y_test_local[test_start_index:test_end_index].reshape(-1, 1),
                                    cnn_model.is_training:False})
                test_writer.add_summary(test_summary, test_step)
                test_loss += tmp_test_loss

        print("test loss: {}\n".format(test_loss/(len(X_test)*test_n_batches)))
        print("END: test")

    # capture Ctrl + C
    except KeyboardInterrupt:
        print("\nPressed \"Ctrl + C\"")
        print("exit problem, save learning model")
        saver.save(sess, out_model_dirc + learning_date + "/model.ckpt")
    # --------------------------------------------------------------------------

    # --------------------------- END PROCESSING -------------------------------
    train_writer.close()
    test_writer.close()
    sess.close()
    # --------------------------------------------------------------------------

if __name__ == "__main__":
    input_image_dirc_path = "/data/sakka/image/original/total/"
    input_dens_dirc_path = "/data/sakka/dens/total/"
    mask_path = "/data/sakka/image/mask.png"
    reuse_model_path = "/data/sakka/tensor_model/2018_4_15_15_7/"
    out_model_dirc = "/data/sakka/tensor_model/"
    X_train, X_test, y_train, y_test = load_data(input_image_dirc_path, input_dens_dirc_path, mask_path, test_size=0.2)
    cnn_learning(X_train, X_test, y_train, y_test, mask_path, reuse_model_path, out_model_dirc)
