#! /usr/bin/env python
# coding: utf-8

import sys
import cv2
import glob
import time
import numpy as np
import tensorflow as tf
from cnn_util import get_masked_data, get_masked_index

def cnn_predict(model_path, input_img_path, output_direc, params_dict):
    # start session
    config = tf.ConfigProto(gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9))
    sess = tf.InteractiveSession(config=config)

    saver = tf.Train.Saver()
    ckpt = tf.train.get_checkpoint_state(modelPath)
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
    index_h, index_w = cnn_util.get_masked_index(mask_path)
    est_start_time = time.time()

    for img_path in img_file_lst:
        img = cv2.imread(img_path)
        label = np.zeros((720, 1280))
        mask_img = cnn_util.get_masked_data(img, mask_path)
        mask_label = cnn_util.get_masked_data(label, mask_path)
        X_local, y_local = cnn_util.get_local_data(masked_img, masked_label, index_h, index_w, local_img_size=72)

        # local image index
        index_lst = []
        for step in range(len(index_h)):
            if step%skip == 0:
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

        np.save(output_direc + "{}/{}.npy".format(skip, outfile_path), est_dens_map)
        print("END: estimate density map")

        # calculate estimation loss
        est_loss = np.mean(np.square(label - est_dens_map), dtype="float32")
        print("estimation loss: {}".format(est_loss))

    with open(output_direc + "{}/time.txt".format(skip), "a") as f:
        f.write("skip: {0}, frame num: {1} total time: {2}\n".format(skip, 35,time.time() - est_start_time)) # modify: division num


if __name__ == "__main__":
    mpdel_path = "/data/sakka/tensor_model/2018_4_15_15_7/"
    input_img_path = "/data/sakka/image/1h_10/*.png"
    output_direc = "/data/sakka/estimation/1h_10/model_201806142123/dens/"
    params_dict = {"skip_width": 15, "est_batch_size": 2500}
    cnn_predit(model_path, input_img_path, output_direc, params_dict)
