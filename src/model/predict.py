#! /usr/bin/env python
# coding: utf-8

import os
import time
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

# ------------------------------- PRE PROCWSSING ----------------------------------
index_h, index_w = get_masked_index("/data/sakka/image/mask.png")
# --------------------------------------------------------------------------

def cnn_predict(cnn_model, sess, img, frame_num, output_dirc, args):
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
    logger.debug("STSRT: predict density map (frame number= {0})".format(frame_num))
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
                                    cnn_model.is_training: False,
                                    cnn_model.keep_prob: 1.0}).reshape(pred_batch_size)
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
    input_img_dirc_lst = [f for f in os.listdir("{0}/{1}".format(args.input_img_root_dirc, args.date)) if not f.startswith(".")]
    for dirc in input_img_dirc_lst:
        input_img_path = "{0}/{1}/{2}/*.png".format(args.input_img_root_dirc, args.date, dirc)
        img_path_lst = glob.glob(input_img_path)
        output_dirc = "{0}/{1}/{2}".format(args.output_root_dirc, args.date, dirc)
        os.makedirs("{0}/dens".format(output_dirc), exist_ok=True)
        os.makedirs("{0}/cord".format(output_dirc), exist_ok=True)
        display_data_info(input_img_path, output_dirc, args.skip_width,
                            args.pred_batch_size, args.band_width, args.cluster_thresh, args.save_map)
        for path in img_path_lst:
            img = cv2.imread(path)
            frame_num = path.split("/")[-1][:-4]
            cnn_predict(cnn_model, sess, img, frame_num, output_dirc, args)

        logger.debug("DONE: {0}".format(input_img_path))



def video_predict(model, sess, args):
    for time_idx in range(9, 17):
        output_dirc = "{0}/{1}/{2}".format(args.output_root_dirc, args.date, time_idx)
        os.makedirs(output_dirc, exist_ok=True)
        os.makedirs("{0}/dens".format(output_dirc), exist_ok=True)
        os.makedirs("{0}/cord".format(output_dirc), exist_ok=True)
        video_path = "{0}/{1}/{2}{3:0>2d}00.mp4".format(args.input_video_dirc, args.date, args.date, time_idx)
        display_data_info(video_path, output_dirc, args.skip_width,
                            args.pred_batch_size, args.band_width, args.cluster_thresh, args.save_map)

        # initialize
        cap, _, _, _, _, _ = set_capture(video_path)
        frame_num = 0

        # skip first frame (company logo)
        for _ in range(4*30):
            _, frame = cap.read()
            frame_num += 1
        cnn_predict(model, sess, frame, frame_num, output_dirc, args)

        # predict at regular interval (args.pred_interval)
        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                frame_num += 1
                if (frame_num%args.pred_interval == 0):
                    cnn_predict(model, sess, frame, frame_num, output_dirc, args)
            else:
                break

        logger.debug("DONE: {0}".format(video_path))
    
    # finalize
    cap.release()
    cv2.destoryAllWindows()


def make_pred_parse():
    parser = argparse.ArgumentParser(
        prog="predict.py",
        usage="prediction by learned model",
        description="description",
        epilog="end",
        add_help=True
    )

    # Data Argument
    parser.add_argument("--date", type=str, default="20170416")
    parser.add_argument("--model_path", type=str,
                        default="/data/sakka/tensor_model/2018_4_15_15_7")
    parser.add_argument("--input_img_root_dirc", type=str,
                        default="/data/sakka/image")
    parser.add_argument("--input_video_dirc", type=str,
                        default="/data/sakka/video")
    parser.add_argument("--output_root_dirc", type=str,
                        default="/data/sakka/estimation/model_201804151507")
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
    # parser.add_argument("--band_width", type=int,
    #                     default=25, help="band width of Mean-Shift Clustering")
    # parser.add_argument("--cluster_thresh", type=float,
    #                     default=0.4, help="threshold to be subjected to clustering")
    parser.add_argument("--band_width", type=int,
                        default=10, help="band width of Mean-Shift Clustering")
    parser.add_argument("--cluster_thresh", type=float,
                        default=0.5, help="threshold to be subjected to clustering")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    # set loggger
    logs_path = "/home/sakka/cnn_by_density_map/logs/predict.log"
    logging.basicConfig(filename=logs_path,
                        level=logging.DEBUG,
                        format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s")

    # set argument
    args = make_pred_parse()
    logger.debug("Running with args: {0}".format(args))

    # load prediction model
    cnn_model, sess = load_model(args.model_path, args.visible_device, args.memory_rate)

    start = time.time()
    # predict from image data
    batch_predict(cnn_model, sess, args)
    # predict from video data
    #video_predict(cnn_model, sess, args)

    elapsed_time = time.time() - start
    elapsed_hour = int(elapsed_time/3600)
    elapsed_min = int((elapsed_time - 3600*elapsed_hour)/60)
    elapsed_sec = elapsed_time - elapsed_hour*3600 - elapsed_min*60

    logger.debug("Elapsed Time: {0} [hour] {1}[min] {2}[sec]".format(
        elapsed_hour, elapsed_min, elapsed_sec))
