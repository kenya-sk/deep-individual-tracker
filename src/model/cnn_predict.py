#! /usr/bin/env python
# coding: utf-8

import os
import math
import sys
import cv2
import glob
import time
import numpy as np
import tensorflow as tf

from cnn_util import get_masked_data, get_masked_index, get_local_data
from cnn_model import CNN_model
from clustering import clustering


def cnn_predict(model_path, input_img_path, output_dirc_path, mask_path, params_dict, save_map=False):
    
    # ------------------------------- PRE PROCWSSING ----------------------------------
    config = tf.ConfigProto(gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9))
    sess = tf.InteractiveSession(config=config)

    print("*************************************************")
    cnn_model = CNN_model()
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

    skip_width = params_dict["skip_width"]
    img_file_lst = glob.glob(input_img_path)
    index_h, index_w = get_masked_index(mask_path)
    print("INPUT IMG DIRC: {}".format(input_img_path))
    print("OUTPUT DIRC: {}".format(output_dirc_path))
    print("SKIP WIDTH: {}".format(params_dict["skip_width"]))
    print("PRED BATCH SIZE: {}".foramt(params_dict["pred_batch_size"]))
    print("BAND WIDTH: {}".format(params_dict["band_width"]))
    print("CLUSTER THRESH: {}".format(params_dict["cluster_thresh"]))
    print("SAVE DENS MAP: {}".format(save_map))
    print("*************************************************\n")
    # --------------------------------------------------------------------------

    # ------------------------------- PREDICT ----------------------------------
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
            print("DONE: batch {}/{}".format(batch+1, pred_n_batches))

            for i in range(pred_batch_size):
                h_est = index_h[index_lst[batch*pred_batch_size+i]]
                w_est = index_w[index_lst[batch*pred_batch_size+i]]
                pred_dens_map[h_est,w_est] = pred_arr[i]

        out_file_path = img_path.split("/")[-1][:-4]
        if save_map:
            np.save(output_dirc_path + "dens/" + "{}.npy".format(out_file_path), pred_dens_map)
        print("END: predict density map\n")

        # calculate centroid by clustering 
        centroid_arr = clustering(pred_dens_map, params_dict["band_width"], params_dict["cluster_thresh"])
        np.savetxt(output_dirc_path + "cord/" + "{}.csv".format(out_file_path),centroid_arr, fmt="%i", delimiter=",")

        # calculate prediction loss
        est_loss = np.mean(np.square(label - pred_dens_map), dtype="float32")
        print("prediction loss: {}".format(est_loss))
        print("END: predict density map")
        print("**************************************************\n")

    #---------------------------------------------------------------------------

    with open(output_dirc_path + "time.txt", "a") as f:
        f.write("skip: {0}, frame num: {1} total time: {2}\n".format(skip_width, 35,time.time() - pred_start_time)) # modify: division num


if __name__ == "__main__":
    model_path = "/data/sakka/tensor_model/2018_7_24_20_41/"
    input_img_root_dirc = "/data/sakka/image/20170416/"
    output_root_dirc = "/data/sakka/estimation/model_20180724/20170416/"
    mask_path = "/data/sakka/image/mask.png"
    params_dict = {"skip_width": 15, "pred_batch_size": 2500, "band_width":25, "cluster_thresh":0.8}
    
    input_img_dirc_lst = [f for f in os.listdir(input_img_root_dirc) if not f.startswith(".")]
    for dirc in input_img_dirc_lst:
        input_ing_path = input_img_root_dirc + dirc + "/*.png"
        output_dirc_path = output_root_dirc + dirc  + "/"
        cnn_predict(model_path, input_img_path, output_dirc_path, mask_path, params_dict, save_map=True)
