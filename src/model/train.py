#! /usr/bin/env python
#coding: utf-8

import os
import time
import sys
import logging
import cv2
import math
import glob
import argparse
from tqdm import trange
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from cnn_util import get_masked_data, get_masked_index, get_local_data
from model import CNN_model

logger = logging.getLogger(__name__)


def load_data(args):
    """
    """

    X = []
    y = []
    file_lst = glob.glob("{0}/*.png".format(args.root_img_dirc))
    if len(file_lst) == 0:
        sys.stderr.write("Error: not found input image")
        sys.exit(1)

    for path in file_lst:
        img = cv2.imread(path)
        if img is None:
            sys.stderr.write("Error: can not read image")
            sys.exit(1)
        else:
            X.append(get_masked_data(img, args.mask_path))
        dens_path = path.replace(".png", ".npy").split("/")[-1]
        dens_map = np.load("{0}/{1}".format(args.root_dens_dirc, dens_path))
        y.append(get_masked_data(dens_map, args.mask_path))

    X = np.array(X)
    y = np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size)

    return X_train, X_test, y_train, y_test


def hard_negative_mining(X, y, loss):
    """
    """

    # get index that error is greater than the threshold
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

    # select all positive samples first
    msk = density_arr >= thresh
    # select same number of negative samples with positive samples
    msk[~msk] = select((~msk).sum(), msk.sum())

    return local_img_mat[msk], density_arr[msk]


def horizontal_flip(X_train, y_train, train_idx, params_dict):
    """
    """

    if np.random.rand() < params_dict["flip_prob"]:
        X_train_local, y_train_local = \
                get_local_data(X_train[train_idx][:,::-1,:], 
                                y_train[train_idx][:,::-1],
                                params_dict["flip_index_h"],
                                params_dict["flip_index_w"], 
                                local_img_size=params_dict["local_size"])
    else:
        X_train_local, y_train_local = \
                get_local_data(X_train[train_idx], 
                                y_train[train_idx], 
                                params_dict["index_h"], 
                                params_dict["index_w"], 
                                local_img_size=params_dict["local_size"])

    return X_train_local, y_train_local


def train(sess, epoch, cnn_model, X_train, y_train, params_dict, merged):
    """
    """

    train_loss = 0.0
    hard_negative_image_arr = np.zeros((1, local_size, local_size, 3),dtype="uint8")
    hard_negative_label_arr = np.zeros((1), dtype="float32")

    for train_idx in trange(len(X_train), desc="training data"):
        # load traing dataset
        # data augmentation (horizontal flip)
        X_train_locol, y_train_local = horizontal_flip(X_train, y_train, train_idx, params_dict)

        # under sampling
        X_train_local, y_train_local = \
                    under_sampling(X_train_local,
                                    y_train_local, 
                                    thresh=params_dict["under_sampling_thresh"])

        # hard negative mining
        if hard_negative_label_arr.shape[0] > 1:
            X_train_local = np.append(X_train_local, hard_negative_image_arr[1:], axis=0)
            y_train_local = np.append(y_train_local, hard_negative_label_arr[1:], axis=0)
        X_train_local, y_train_local = shuffle(X_train_local, y_train_local)

        # learning by batch
        hard_negative_image_arr = np.zeros((1, params_dict["local_size"], params_dict["local_size"], 3), dtype="uint8")
        hard_negative_label_arr = np.zeros((1), dtype="float32")
        train_n_batches = int(len(X_train_local) / params_dict["batch_size"])
        for train_batch in range(train_n_batches):
            train_start_index = train_batch * params_dict["batch_size"]
            train_end_index = train_start_index + params_dict["batch_size"]

            train_diff = sess.run(cnn_model.diff, feed_dict={
                cnn_model.X: X_train_local[train_start_index:train_end_index].reshape(-1, params_dict["local_size"], params_dict["local_size"], 3),
                cnn_model.y_: y_train_local[train_start_index:train_end_index].reshape(-1, 1),
                cnn_model.is_training: True,
                cnn_model.keep_prob: 0.5
                })

            train_loss += np.mean(train_diff)

            train_summary, _ = sess.run([merged, cnn_model.learning_step],feed_dict={
                cnn_model.X: X_train_local[train_start_index:train_end_index].reshape(-1, params_dict["local_size"], params_dict["local_size"], 3),
                cnn_model.y_: y_train_local[train_start_index:train_end_index].reshape(-1, 1),
                cnn_model.is_training: True,
                cnn_model.keep_prob: 0.5
                })

            # hard negative mining
            batch_hard_negative_image_arr, batch_hard_negative_label_arr = \
                    hard_negative_mining(X_train_local[train_start_index:train_end_index], 
                                            y_train_local[train_start_index:train_end_index],
                                            train_diff)
            if batch_hard_negative_label_arr.shape[0] > 0: # there are hard negative data
                hard_negative_image_arr = np.append(hard_negative_image_arr, batch_hard_negative_image_arr, axis=0)
                hard_negative_label_arr = np.append(hard_negative_label_arr, batch_hard_negative_label_arr, axis=0)
            else:
                pass
    
    # record training summary to TensorBoard
    train_writer.add_summary(train_summary, epoch)

    # mean train loss per 1 image
    mean_train_loss = train_loss/(len(X_train)*train_n_batches)

    return mean_train_loss


def validation(sess, epoch, cnn_model, X_val, y_val, params_dict, merged):
    """
    """

    val_loss = 0.0
    for val_idx in trange(len(X_val), desc="validation data"):
        X_val_local, y_val_local = get_local_data(X_val[val_idx], y_val[val_idx],
                                                    params_dict["index_h"], params_dict["index_w"], 
                                                    local_img_size=params_dict["local_size"])
        val_n_batches = int(len(X_val_local) / params_dict["batch_size"])
        for val_batch in range(val_n_batches):
            val_start_index = val_batch * params_dict["batch_size"]
            val_end_index = val_start_index + params_dict["batch_size"]

            val_loss_summary, val_batch_loss = sess.run([merged, cnn_model.loss], feed_dict={
                cnn_model.X: X_val_local[val_start_index:val_end_index].reshape(-1, params_dict["local_size"], params_dict["local_size"], 3),
                cnn_model.y_: y_val_local[val_start_index:val_end_index].reshape(-1, 1),
                cnn_model.is_training: False,
                cnn_model.keep_prob: 1.0
                })

            val_loss += val_batch_loss

    # record validation summary to TensorBoard
    val_writer.add_summary(val_loss_summary, epoch)

    # mean validation loss per 1 epoch
    mean_val_loss = val_loss/(len(X_val)*val_n_batches)

    return mean_val_loss


def test(sess, cnn_model, X_test, y_test, params_dict, merged):
    """
    """

    test_step = 0
    test_loss = 0.0
    for test_idx in trange(len(X_test), desc="test data"):
        X_test_local, y_test_local = get_local_data(X_test[test_idx], y_test[test_idx], 
                                                    params_dict["index_h"], params_dict["index_w"], 
                                                    local_img_size=params_dict["local_size"])
        test_n_batches = int(len(X_test_local) / batch_size)
        for test_batch in range(test_n_batches):
            test_step += 1
            test_start_index = test_batch * batch_size
            test_end_index = test_start_index + batch_size

            test_summary, test_batch_loss = sess.run([merged, cnn_model.loss], feed_dict={
                cnn_model.X: X_test_local[test_start_index:test_end_index].reshape(-1, params_dict["local_size"], params_dict["local_size"], 3),
                cnn_model.y_: y_test_local[test_start_index:test_end_index].reshape(-1, 1),
                cnn_model.is_training:False
                })
            test_writer.add_summary(test_summary, test_step)
            test_loss += test_batch_loss

    # mean test loss per 1 epoch
    mean_test_loss = test_loss/(len(X_test)*test_n_batches)

    return mean_test_loss


def cnn_learning(X_train, X_test, y_train, y_test, args):
    # -------------------------- PRE PROCESSING --------------------------------
    # start session
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(
            visible_device_list=args.visible_device,
            per_process_gpu_memory_fraction=args.memory_rate))
    sess = tf.InteractiveSession(config=config)
    start_time = time.time()
    cnn_model = CNN_model()

    # mask index
    # if you analyze all areas, please set a white image
    index_h, index_w = get_masked_index(args.mask_path, horizontal_flip=False)
    flip_index_h, flip_index_w = get_masked_index(args.mask_path, horizontal_flip=True)

    # logs of tensor board directory
    date = datetime.now()
    learning_date = "{0}_{1}_{2}_{3}_{4}".format(date.year, date.month, date.day, date.hour, date.minute)
    log_dirc = "{0}/{1}".format(args.root_log_dirc, learning_date)
    logger.debug("Training date: {0}".format(learning_date))

    # delete the specified directory if it exists, recreate it
    if tf.gfile.Exists(log_dirc):
        tf.gfile.DeleteRecursively(log_dirc)
    tf.gfile.MakeDirs(log_dirc)

    # variable of TensorBoard
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter("{0}/train".format(log_dirc), sess.graph)
    val_writer = tf.summary.FileWriter("{0}/val".format(log_dirc))
    test_writer = tf.summary.FileWriter("{0}/test".format(log_dirc))

    # split data: validation or test
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5)
    # --------------------------------------------------------------------------

    # -------------------------- LEARNING STEP --------------------------------
    n_epochs = args.n_epochs
    val_loss_lst = []
    not_improved_count = 0
    logger.debug("original traning data size: {0}".format(len(X_train)))

    # parameter dictionary for train, val, test
    params_dict = {"index_h"               : index_h,
                   "index_w"               : index_w,
                   "flip_index_h"          : flip_index_h,
                   "flip_index_w"          : flip_index_w,
                   "local_size"            : args.local_img_size,
                   "batch_size"            : args.batch_size,
                   "flip_prob"             : args.flip_prob,
                   "under_sampling_thresh" : args.under_sampling_thresh}

    # check if the ckpt exist
    # relearning or not
    saver = tf.train.Saver() # save weight
    ckpt = tf.train.get_checkpoint_state(args.reuse_model_path) # model exist: True or False
    if ckpt:
        last_model = ckpt.model_checkpoint_path
        logger.debug("START: relearning cnn model")
        logger.debug("LODE: {0}".format(last_model))
        saver.restore(sess, last_model)
    else:
        logger.debug("START: learning")
        # initialize all variable
        tf.global_variables_initializer().run()

    try:
        for epoch in range(n_epochs):
            logger.debug("************************************************")
            logger.debug("elapsed time: {0:.3f} [sec]".format(time.time() - start_time))

            # train CNN model for 1 epoch
            mean_train_loss = train(sess, epoch, cnn_model, X_train, y_train, params_dict, merged)

            # validation trained CNN model
            mean_val_loss = validation(sess, epoch, cnn_model, X_val, y_val, params_dict, merged)

            # record loss data
            val_loss_lst.append(mean_val_loss)
            logger.debug("epoch                            : {0}".format(epoch+1))
            logger.debug("mean train loss [per image]      : {0}".format(mean_train_loss))
            logger.debug("mean validation loss [per image] : {0}".format(val_loss_lst[epoch]))
        
            # early stopping
            if (epoch > args.min_epoch) and (val_loss_lst[-1] > val_loss_lst[-2]):
                not_improved_count += 1
            else:
                # learning is going well
                not_improved_count = 0
                # save best model
                saver.save(sess, "{0}/{1}/model.ckpt".format(args.save_model_dirc, learning_date))

            if not_improved_count >= args.stop_count:
                logger.debug("early stopping due to not improvement after {0} epochs.".format(args.stop_epoch))
                break

            logger.debug("not improved count / early stopping epoch: {0}/{1}".format(not_improved_count, args.stop_count))
            logger.debug("************************************************")
        # save best model
        saver.save(sess, "{0}/{1}/model.ckpt".format(args.save_model_dirc, learning_date))
        logger.debug("END: learning")
        # --------------------------------------------------------------------------


        # -------------------------------- TEST ------------------------------------
        logger.debug("START: test")
        mean_test_loss = test(sess, cnn_model, X_test, y_test, params_dict, merged)

        logger.debug("mean test loss [per image]: {0}".format(mean_test_loss))
        logger.debug("END: test")
        # --------------------------------------------------------------------------

    # capture Ctrl + C
    except KeyboardInterrupt:
        logger.debug("\nPressed \"Ctrl + C\"")
        logger.debug("exit problem, save learning model")
        saver.save(sess, "{0}/{1}/model.ckpt".format(args.save_model_dirc, learning_date))
    # --------------------------------------------------------------------------

    # ----------------------------- TERMINATE ---------------------------------
    train_writer.close()
    test_writer.close()
    sess.close()
    # -------------------------------------------------------------------------


def make_learning_parse():
    parser = argparse.ArgumentParser(
        prog="cnn_learning.py",
        usage="training model",
        description="description",
        epilog="end",
        add_help=True
    )

    # Data Argument
    parser.add_argument("--root_img_dirc", type=str,
                        default="/data/sakka/image/original/total")
    parser.add_argument("--root_dens_dirc", type=str,
                        default="/data/sakka/dens/total")
    parser.add_argument("--mask_path", type=str,
                        default="/data/sakka/image/mask.png")
    parser.add_argument("--reuse_model_path", type=str,
                        default="/data/sakka/tensor_model/2018_4_15_15_7")
    parser.add_argument("--root_log_dirc", type=str,
                        default="/data/sakka/tensor_log")
    parser.add_argument("--save_model_dirc", type=str,
                        default="/data/sakka/tensor_model")

    # GPU Argumant
    parser.add_argument("--visible_device", type=str,
                        default="0,1", help="ID of using GPU: 0-max number of available GPUs")
    parser.add_argument("--memory_rate", type=float,
                        default=0.9, help="useing each GPU memory rate: 0.0-1.0")

    # Parameter Argument
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--local_img_size", type=int,
                        default=72, help="square local image size: > 0")
    parser.add_argument("--n_epochs", type=int,
                        default=30, help="number of epoch: > 0")
    parser.add_argument("--batch_size", type=int,
                        default=512, help="batch size of learning: > 0")
    parser.add_argument("--min_epoch", type=int,
                        default=5, help="minimum learning epoch (not apply early stopping): > 0")
    parser.add_argument("--stop_count", type=int,
                        default=2, help="over this number, learning is stop: > 0")
    parser.add_argument("--flip_prob", type=float,
                        default=0.5, help="probability of horaizontal flip. (apply only training data): 0.0-1.0")
    parser.add_argument("--under_sampling_thresh", type=float,
                        default=0.2, help="over this value, positive data: 0.0-1.0 (value of density map)")

    args = parser.parse_args()
    
    return args


if __name__ == "__main__":
    logs_path = "../../logs/train.log"
    logging.basicConfig(filename=logs_path,
                        level=logging.DEBUG,
                        format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s")
    args = make_learning_parse()
    logger.debug("Running with args: {0}".format(args))
    logger.debug("Log path: {0}".format(logs_path))

    X_train, X_test, y_train, y_test = load_data(args)
    cnn_learning(X_train, X_test, y_train, y_test, args)
