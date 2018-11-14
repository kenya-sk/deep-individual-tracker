#! /usr/bin/env python
# coding: utf-8

import os
import math
import logging
import sys
import cv2
import glob
import time
import argparse
import numpy as np
import tensorflow as tf

from cnn_util import display_data_info, get_masked_data, get_masked_index, get_local_data, load_model, set_capture
from clustering import clustering


logger = logging.getLogger(__name__)


def cnn_predict(cnn_model, sess, img, frame_num, output_dirc, args):
    
    # ------------------------------- PRE PROCWSSING ----------------------------------
    index_h, index_w = get_masked_index(args.mask_path)
    # --------------------------------------------------------------------------

    # ------------------------------- PREDICT ----------------------------------
    label = np.zeros(args.original_img_size)
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
    pred_dens_map = np.zeros(args.original_img_size, dtype="float32")

    logger.debug("*************************************************")
    logger.debug("STSRT: predict density map")
    for batch in range(pred_n_batches):
        # array of skiped local image
        X_skip = np.zeros((pred_batch_size, args.local_img_size, args.local_img_size, 3))
        y_skip = np.zeros((pred_batch_size,1))
        for index_cord, index_local in enumerate(range(pred_batch_size)):
            current_index = index_lst[batch*pred_batch_size+index_local]
            X_skip[index_cord] = X_local[current_index]
            y_skip[index_cord] = y_local[current_index]

        # esimate each local image
        pred_arr = sess.run(cnn_model.y, feed_dict={
                                    cnn_model.X: X_skip,
                                    cnn_model.y_: y_skip,
                                    cnn_model.is_training: False}).reshape(pred_batch_size)
        logger.debug("DONE: batch {0}/{1}".format(batch+1, pred_n_batches))

        for batch_idx in range(pred_batch_size):
            h_est = index_h[index_lst[batch*pred_batch_size+batch_idx]]
            w_est = index_w[index_lst[batch*pred_batch_size+batch_idx]]
            pred_dens_map[h_est,w_est] = pred_arr[batch_idx]


    # save data
    if args.save_map:
        np.save("{0}/dens/{1}.npy".format(output_dirc, frame_num), pred_dens_map)
    logger.debug("END: predict density map")

    # calculate centroid by clustering 
    centroid_arr = clustering(pred_dens_map, args.band_width, args.cluster_thresh)
    np.savetxt("{0}/cord/{1}.csv".format(output_dirc, frame_num),centroid_arr, fmt="%i", delimiter=",")

    # calculate prediction loss
    pred_loss = np.mean(np.square(label - pred_dens_map), dtype="float32")
    logger.debug("prediction loss: {0}".format(pred_loss))
    logger.debug("END: predict density map")
    logger.debug("***************************************************")
    #---------------------------------------------------------------------------



def batch_predict(model, sess, args):
    input_img_dirc_lst = [f for f in os.listdir(args.input_img_root_dirc) if not f.startswith(".")]
    for dirc in input_img_dirc_lst:
        input_img_path = "{0}/{1}/*.png".format(args.input_img_root_dirc, dirc)
        img_path_lst = glob.glob(input_img_path)
        output_dirc = "{0}/{1}".format(args.output_root_dirc, dirc)
        os.makedirs("{0}/dens".format(output_dirc), exist_ok=True)
        os.makedirs("{0}/cord".format(output_dirc), exist_ok=True)
        display_data_info(input_img_path, output_dirc, args.skip_width,
                            args.pred_batch_size, args.band_width, args.cluster_thresh, args.save_map)
        for path in img_path_lst:
            img = cv2.imread(path)
            frame_num = path.split("/")[-1][:-4]
            cnn_predict(cnn_model, sess, img, frame_num, output_dirc, args)


def movie_predict(model, sess, args):
    cap, _, _, _, _, _ = set_capture(args.input_movie_path)
    frame_num = 0


    # initialize
    # skip first frame (company logo)
    for _ in range(4*30):
        _, prev = cap.read()
        frame_num += 1

    #output_dric = args.output_root_dirc
    output_dirc = "/data/sakka/estimation/model_201804151507/20170421/test"
    cnn_predict(model, sess, prev, frame_num, output_dirc, args)

    while (cap.isOpened()):
        ret, frame = cap.read()
        frame_num += 1
        break
        if (frame_num%args.pred_interval == 0):
            print("PREDICT")
        else:
            pass


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
                        default="/data/sakka/tensor_model/2018_4_15_15_7")
    parser.add_argument("--input_img_root_dirc", type=str,
                        default="/data/sakka/image/20170416")
    parser.add_argument("--input_movie_path", type=str,
                        default="/data/sakka/movie/20170416/201704160900.mp4")
    parser.add_argument("--output_root_dirc", type=str,
                        default="/data/sakka/estimation/model_201804151507/20170416")
    parser.add_argument("--mask_path", type=str,
                        default="/data/sakka/image/mask.png")

    # GPU Argument
    parser.add_argument("--visible_device", type=str,
                        default="0,1", help="ID of using GPU: 0-max number of available GPUs")
    parser.add_argument("--memory_rate", type=float,
                        default=0.9, help="useing each GPU memory rate: 0.0-1.0")

    # Parameter Argument
    parser.add_argument("--original_img_size", type=tuple,
                        default=(720, 1280), help="(height, width)")
    parser.add_argument("--local_img_size", type=int,
                        default=72, help="square local image size: > 0")
    parser.add_argument("--pred_interval", type=int,
                        default=30, help="skip interval of frame at prediction")
    parser.add_argument("--skip_width", type=int,
                        default=15, help="skip width in horizontal direction ")
    parser.add_argument("--pred_batch_size", type=int,
                        default=2500, help="batch size for each epoch")
    parser.add_argument("--save_map", type=bool,
                        default=False, help="save pred density map (True of False)")
    parser.add_argument("--band_width", type=int,
                        default=25, help="band width of Mean-Shift Clustering")
    parser.add_argument("--cluster_thresh", type=float,
                        default=0.4, help="threshold to be subjected to clustering")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    logs_path = "/home/sakka/cnn_by_density_map/logs/cnn_predict.log"
    logging.basicConfig(filename=logs_path,
                        level=logging.DEBUG,
                        format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s")
    args = make_pred_parse()
    logger.debug("Running with args: {0}".format(args))
    cnn_model, sess = load_model(args.model_path, args.visible_device, args.memory_rate)
    #batch_predict(cnn_model, sess, args)
    movie_predict(cnn_model, sess, args)
