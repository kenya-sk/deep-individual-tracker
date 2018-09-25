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

from cnn_util import display_data_info, get_masked_data, get_masked_index, get_local_data, load_model
from clustering import clustering


def cnn_predict(cnn_model, sess, input_img_path, output_dirc_path, args):
    
    # ------------------------------- PRE PROCWSSING ----------------------------------
    img_file_lst = glob.glob(input_img_path)
    index_h, index_w = get_masked_index(args.mask_path)
    display_data_info(input_img_path, output_dirc_path, args.skip_width, 
                        args.pred_batch_size, args.cluster_thresh, args.save_map)
    # --------------------------------------------------------------------------

    # ------------------------------- PREDICT ----------------------------------
    pred_start_time = time.time()
    for i, img_path in enumerate(img_file_lst):
        img = cv2.imread(img_path)
        label = np.zeros((720, 1280))
        masked_img = get_masked_data(img, args.mask_path)
        masked_label = get_masked_data(label, args.mask_path)
        X_local, y_local = get_local_data(masked_img, masked_label, index_h, index_w, local_img_size=args.local_img_size)

        # local image index
        index_lst = []
        for step in range(len(index_h)):
            if step%args.skip_width == 0:
                index_lst.append(step)

        pred_batch_size = args.pred_batch_size
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
        if "_" in out_file_path:
            out_file_path = out_file_path.split("_")[-1]

        if args.save_map:
            np.save(output_dirc_path + "dens/" + "{}.npy".format(out_file_path), pred_dens_map)
        print("END: predict density map\n")

        # calculate centroid by clustering 
        centroid_arr = clustering(pred_dens_map, args.band_width, args.cluster_thresh)
        np.savetxt(output_dirc_path + "cord/" + "{}.csv".format(out_file_path),centroid_arr, fmt="%i", delimiter=",")

        # calculate prediction loss
        est_loss = np.mean(np.square(label - pred_dens_map), dtype="float32")
        print("prediction loss: {}".format(est_loss))
        print("END: predict density map")
        print("***************************************************\n")

    #---------------------------------------------------------------------------

    with open(output_dirc_path + "time.txt", "a") as f:
        f.write("skip: {0}, frame num: {1} total tme: {2}\n".format(args.skip_width, len(img_file_lst), time.time() - pred_start_time))


def make_pred_parse():
    parser = argparse.ArgumentParser(
        prog="cnn_predict.py",
        usage="prediction by learned model",
        description="description",
        epilog="end",
        add_help=True
    )

    # Data Argument
    parser.add_argument("--model_path", type=str,
                        default="/data/sakka/tensor_model/2018_4_15_15_7/")
    parser.add_argument("--input_img_root_dirc", type=str,
                        default="/data/sakka/image/20170416/")
    parser.add_argument("--output_root_dirc", type=str,
                        default="/data/sakka/estimation/model_201804151507/20170416/")
    parser.add_argument("--mask_path", type=str,
                        default="/data/sakka/image/mask.png")

    # GPU Argument
    parser.add_argument("--visible_device", type=str,
                        default="0,1", help="ID of using GPU: 0-max number of available GPUs")
    parser.add_argument("--memory_rate", type=float,
                        default=0.9, help="useing each GPU memory rate: 0.0-1.0")

    # Paraameter Argument
    parser.add_argument("--local_img_size", type=int,
                        default=72, help="square local image size: > 0")
    parser.add_argument("--skip_width", type=int,
                        default=15, help="skip width in horizontal direction ")
    parser.add_argument("--pred_batch_size", type=int,
                        default=2500, help="batch size for each epoch")
    parser.add_argument("--save_map", type=bool,
                        default=False, help="save pred density map (True of False)")
    parser.add_argument("--band_width", type=int,
                        default=25, "band width of Mean-Shift Clustering")
    parser.add_argument("--cluster_thresh", type=float,
                        default=0.4, help="thresh to be subjected to clustering")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    # model_path = "/data/sakka/tensor_model/2018_4_15_15_7/"
    # gpu_config_dict = {"visible_device":"0,1", "memory_rate":0.9}
    # cnn_model, sess = load_model(model_path, gpu_config_dict)
    # input_img_root_dirc = "/data/sakka/image/20170416/"
    # output_root_dirc = "/data/sakka/estimation/model_201804151507/20170416/"
    # mask_path = "/data/sakka/image/mask.png"
    # params_dict = {"skip_width": 15, "pred_batch_size": 2500, "band_width":25, "cluster_thresh":0.4}
    
    # input_img_dirc_lst = [f for f in os.listdir(input_img_root_dirc) if not f.startswith(".")]
    # for dirc in input_img_dirc_lst:
    #     input_img_path = input_img_root_dirc + dirc + "/*.png"
    #     output_dirc_path = output_root_dirc + dirc  + "/"
    #     os.makedirs(output_dirc_path+"/dens", exist_ok=True)
    #     os.makedirs(output_dirc_path+"/cord", exist_ok=True)
    #     cnn_predict(cnn_model, sess, input_img_path, output_dirc_path, mask_path, params_dict, save_map=True)


    args = make_pred_parse()
    cnn_model, sess = load_model(args.model_path, args.visible_device, args.memory_rate)
    input_img_dirc_lst = [f for f in os.listdir(args.input_img_root_dirc) if not f.startswith(".")]
    for dirc in input_img_dirc_lst:
        input_img_path = args.input_img_root_dirc + dirc + "/*.png"
        output_dirc_path = args.output_root_dirc + dirc + "/"
        os.makedirs(output_dirc_path + "/dens", exist_ok=True)
        os.makedirs(output_dirc_path + "/cord", exist_ok=True)
        cnn_predict(cnn_model, sess, input_img_path, output_dirc_path, args)
