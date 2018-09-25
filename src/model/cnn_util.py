#! /usr/bin/env python
# coding: utf-8

import os
import math
import cv2
import numpy as np
import tensorflow as tf

from cnn_model import CNN_model

ANALYSIS_HEIGHT = (0, 470)
ANALYSIS_WIDTH = (0, 1280)


def display_data_info(input_img_path, output_dirc_path, skip_width, pred_batch_size, band_width, cluster_thresh, save_map):
    print("*************************************************")
    print("INPUT IMG DIRC: {}".format(input_img_path))
    print("OUTPUT DIRC: {}".format(output_dirc_path))
    print("SKIP WIDTH: {}".format(skip_width))
    print("PRED BATCH SIZE: {}".format(pred_batch_size))
    print("BAND WIDTH: {}".format(band_width))
    print("CLUSTER THRESH: {}".format(cluster_thresh))
    print("SAVE DENS MAP: {}".format(save_map))
    print("*************************************************\n")


def pretty_print(true_positive_lst, false_positive_lst, false_negative_lst):
    def eval_metrics(true_positive, false_positive, false_negative):
        n = true_positive + false_positive + false_negative
        accuracy = true_positive/n
        precision = true_positive/(true_positive+false_positive)
        recall = true_positive/(true_positive+false_negative)
        f_measure = (2*recall*precision)/(recall+precision)
        return accuracy, precision, recall, f_measure

    accuracy_lst = []
    precision_lst = []
    recall_lst = []
    f_measure_lst = []
    for i in range(len(true_positive_lst)):
        accuracy, precision, recall, f_measure = \
                eval_metrics(true_positive_lst[i], false_positive_lst[i], false_negative_lst[i])
        accuracy_lst.append(accuracy)
        precision_lst.append(precision)
        recall_lst.append(recall)
        f_measure_lst.append(f_value)

    print("\n**************************************************************")

    print("                        GROUND TRUTH          ")
    print("                    |     P   |     N    |           ")
    print("          -----------------------------------------")
    print("                P   |     {0}   |     {1}    |           ".format(sum(true_positive_lst), sum(false_positive_lst)))
    print("PRED      -----------------------------------------")
    print("                N   |     {0}   |     /    |           ".format(sum(false_negative_lst)))
    print("          -----------------------------------------")

    print("\nToal Accuracy (data size {0}, sikp size {1}): {2}".format(len(accuracy_lst), skip, sum(accuracy_lst)/len(accuracy_lst)))
    print("Toal Precision (data size {0}, sikp size {1}): {2}".format(len(precision_lst), skip, sum(precision_lst)/len(precision_lst)))
    print("Toal Recall (data size {0}, sikp size {1}): {2}".format(len(recall_lst), skip, sum(recall_lst)/len(recall_lst)))
    print("Toal F measure (data size {0}, sikp size {1}): {2}".format(len(f_measure_lst), skip, sum(f_measure_lst)/len(f_measure_lst)))
    print("****************************************************************")


def get_masked_data(data, mask_path=None):
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
        mask = np.ones((720, 1280, 3))
    else:
        mask = cv2.imread(mask_path)

    if len(data.shape) == 3:
        masked_data = data*mask
    else:
        masked_data = data*mask[:,:,0]

    return masked_data


def get_masked_index(mask_path=None, horizontal_flip=False):
    """
    input:
        mask_path: binay mask path
        horizontal_flip: default "NO" data augumentation
    output:
        valid index list (heiht and width)
    """

    if mask_path is None:
         mask = np.ones((720, 1280))
    else:
        mask = cv2.imread(mask_path)

    if mask.shape[2] == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # index of data augumentation 
    if horizontal_flip:
        mask = mask[:,::-1]

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


def load_model(model_path, device_id, memory_rate):
    """
    input:
        model_path: path of learned model
        memory_fraction_rate: rate of GPU memory (0.0-1.0)

    output:
        model: CNN model (defined TensorFlow graph)
        sess: TensorFlow session

    """

    config = tf.ConfigProto(gpu_options=tf.GPUOptions(
        visible_device_list = device_id,
        per_process_gpu_memory_fraction=memory_rate))
    sess = tf.InteractiveSession(config=config)

    model = CNN_model()
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(model_path)
    if ckpt:
        last_model = ckpt.model_checkpoint_path
        print("LODE MODEL: {}".format(last_model))
        saver.restore(sess, last_model)
    else:
        sys.stderr("Eroor: Not exist model!")
        sys.stderr("Please check model_path")
        sys.exit(1)

    return model, sess
