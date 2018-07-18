#! /usr/bin/env python
# coding: utf-8

import math
import sys
import cv2
import glob
import time
import numpy as np
import tensorflow as tf

from cnn_util import get_masked_data, get_masked_index, get_local_data
from cnn_model import CNN_model


def cnn_predict(model_path, input_img_path, output_dirc_path, mask_path, params_dict):
    # start session
    config = tf.ConfigProto(gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9))
    sess = tf.InteractiveSession(config=config)

    cnn_model = CNN_model()

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
    index_h, index_w = get_masked_index(mask_path)
    pred_start_time = time.time()

    for i, img_path in enumerate(img_file_lst):
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

        pred_batch_size = params_dict["pred_batch_size"]
        pred_n_batches = int(len(index_lst)/pred_batch_size)
        pred_arr = np.zeros(pred_batch_size)
        pred_dens_map = np.zeros((720,1280), dtype="float32")

        print("*************************************************")
        print("STSRT: predict density map ({}/{})".format(i+1, len(img_file_lst)))
        for batch in range(pred_n_batches):
            # array of skiped local image
            X_skip = np.zeros((pred_batch_size,72,72,3))
            y_skip = np.zeros((pred_batch_size,1))
            for index_cord,index_local in enumerate(range(pred_batch_size)):
                current_index = index_lst[batch*pred_batch_size+index_local]
                X_skip[index_cord] = X_local[current_index]
                y_skip[index_cord] = y_local[current_index]

            # esimate each local image
            pred_arr = sess.run(cnn_model.y, feed_dict={
                                        cnn_model.X: X_skip,
                                        cnn_model.y_: y_skip,
                                        cnn_model.is_training: False}).reshape(pred_batch_size)
            print("DONE: batch {}/{}".format(batch, pred_n_batches))

            for i in range(pred_batch_size):
                h_est = index_h[index_lst[batch*pred_batch_size+i]]
                w_est = index_w[index_lst[batch*pred_batch_size+i]]
                pred_dens_map[h_est,w_est] = pred_arr[i]

        out_file_path = img_path.split("/")[-1][:-4]
        np.savetxt(output_dirc_path + "{}/{}.csv".format(skip_width, out_file_path), pred_dens_map, fmt="%i", delimiter=",")
        print("END: predict density map")

        # calculate prediction loss
        est_loss = np.mean(np.square(label - pred_dens_map), dtype="float32")
        print("prediction loss: {}".format(est_loss))
        print("END: predict density map")
        print("**************************************************\n")

    #---------------------------------------------------------------------------

    with open(output_dirc_path + "{}/time.txt".format(skip), "a") as f:
        f.write("skip: {0}, frame num: {1} total time: {2}\n".format(skip_width, 35,time.time() - pred_start_time)) # modify: division num


if __name__ == "__main__":
    model_path = "/data/sakka/tensor_model/2018_4_15_15_7/"
    input_img_path = "/data/sakka/image/original/20170421/9/*.png"
    output_dirc_path = "/data/sakka/estimation/20170421/9/dens/"
    mask_path = "/data/sakka/image/mask.png"
    params_dict = {"skip_width": 15, "pred_batch_size": 2500}
    cnn_predict(model_path, input_img_path, output_dirc_path, mask_path, params_dict)
