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


def load_model(model_path, memory_fraction_rate=0.9):
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(
        per_process_gpu_memory_fraction=memory_fraction_rate))
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
