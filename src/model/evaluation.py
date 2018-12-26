#! /usr/bin/env python
#coding: utf-8

import logging
import numpy as np
import pandas as pd
import csv
import glob
import argparse
from tqdm import tqdm
from scipy import optimize

from cnn_util import eval_metrics, pretty_print, get_masked_index


logger = logging.getLogger(__name__)


def get_ground_truth(ground_truth_path, mask_path=None):
    """
    plots the coordinates of answer label on the black image(all value 0) and
    creates a correct image for accuracy evaluation.

    input:
        ground_truth_path: coordinates of the estimated target (.csv)
        mask_path: enter path only when using mask image

    output:
        array showing the positon of target
    """

    ground_truth_arr = np.array(pd.read_csv(ground_truth_path))
    if mask_path is None:
        return ground_truth_arr
    else:
        valid_h, valid_w = get_masked_index(mask_path)
        valid_ground_truth_lst = []

        for i in range(ground_truth_arr.shape[0]):
            index_w = np.where(valid_w == ground_truth_arr[i][0])
            index_h = np.where(valid_h == ground_truth_arr[i][1])
            intersect = np.intersect1d(index_w, index_h)
            if len(intersect) == 1:
                valid_ground_truth_lst.append([valid_w[intersect[0]], valid_h[intersect[0]]])
            else:
                pass
        return np.array(valid_ground_truth_lst)


def eval_detection(est_centroid_arr, ground_truth_arr, dist_thresh):
    """
    the distance between the estimation and groundtruth is less than distThreshold --> True

    input:
        est_centroid_arr: coordinates of the estimated target (.npy). array shape == original image shape
        ground_truth_arr: coordinates of the answer label (.csv)
        dist_threshold: it is set maximum value of kernel width

    output:
        true_positive, false_positive, false_negative
    """

    # make distance matrix
    est_centroid_num = len(est_centroid_arr)
    ground_truth_num = len(ground_truth_arr)
    n = max(est_centroid_num, ground_truth_num)
    dist_matrix = np.zeros((n,n))
    for i in range(len(est_centroid_arr)):
        for j in range(len(ground_truth_arr)):
            dist_cord = est_centroid_arr[i] - ground_truth_arr[j]
            dist_matrix[i][j] = np.linalg.norm(dist_cord)

    # calculate by hangarian algorithm
    row, col = optimize.linear_sum_assignment(dist_matrix)
    indexes = []
    for i in range(n):
        indexes.append([row[i], col[i]])
    valid_index_length = min(len(est_centroid_arr), len(ground_truth_arr))
    valid_lst = [i for i in range(valid_index_length)]
    if len(est_centroid_arr) >= len(ground_truth_arr):
        match_indexes = list(filter((lambda index : index[1] in valid_lst), indexes))
    else:
        match_indexes = list(filter((lambda index : index[0] in valid_lst), indexes))

    true_positive = 0
    if est_centroid_num > ground_truth_num:
        false_positive = est_centroid_num - ground_truth_num
        false_negative = 0
    else:
        false_positive = 0
        false_negative = ground_truth_num - est_centroid_num
        
    for i in range(len(match_indexes)):
        pred_index = match_indexes[i][0]
        truth_index = match_indexes[i][1]
        if dist_matrix[pred_index][truth_index] <= dist_thresh:
            true_positive += 1
        else:
            false_positive += 1
            false_negative += 1
    
    return true_positive, false_positive, false_negative, n


def evaluate(args):
    """
    evaluate by accuracy, precision, recall, f measure
    """
    abs_path_lst = glob.glob("{0}/*.csv".format(args.pred_centroid_dirc))
    for skip in args.skip_width_lst:
        true_positive_lst = []
        false_positive_lst = []
        false_negative_lst = []
        sample_num_lst = []
        for path in tqdm(abs_path_lst):
            file_name = path.split("/")[-1]
            centroid_arr = np.loadtxt(
                "{0}/{1}".format(args.pred_centroid_dirc, file_name), delimiter=",")
            if centroid_arr.shape[0] == 0:
                logger.debug("file num: {0}, Not found point of centroid. Accuracy is 0.0".format(file_num))
                true_positive_lst.append(0)
                false_positive_lst.append(0)
                false_negative_lst.append(0)
                sample_num_lst.append(1)
            else:
                ground_truth_arr = get_ground_truth(
                    "{0}/{1}".format(args.ground_truth_dirc, file_name), args.mask_path)
                true_pos, false_pos, false_neg, n = eval_detection(
                    centroid_arr, ground_truth_arr, args.dist_thresh)
                true_positive_lst.append(true_pos)
                false_positive_lst.append(false_pos)
                false_negative_lst.append(false_neg)
                sample_num_lst.append(n)
                
                # calculate evaluation metrics
                accuracy, precision, recall, f_measure = eval_metrics(true_pos, false_pos, false_neg, n)
                logger.debug("file_name: {0}, accuracy: {1}, precision: {2}, recall: {3}, f-measure: {4}".format(
                    file_name, accuracy, precision, recall, f_measure))

        pretty_print(true_positive_lst, false_positive_lst, false_negative_lst, sample_num_lst, skip=skip)

        logger.debug("DONE: evaluation (skip={0})".format(skip))


def make_eval_parse():
    parser = argparse.ArgumentParser(
        prog="evaluation.py",
        usage="evaluate model",
        description="description",
        epilog="end",
        add_help=True
    )

    # Data Argment
    parser.add_argument("--pred_centroid_dirc", type=str,
                        default="/Users/sakka/cnn_by_density_map/test_data/pred/cord")
    parser.add_argument("--ground_truth_dirc", type=str,
                        default="/Users/sakka/cnn_by_density_map/test_data/answer/cord")
    parser.add_argument("--mask_path", type=str,
                        default="/Users/sakka/cnn_by_density_map/image/mask.png")

    # Parameter Argument
    parser.add_argument("--dist_thresh", type=int,
                        default=25, help="threshold of detect or not")
    parser.add_argument("--skip_width_lst", type=list,
                        default=[15], help="test by each skip width")

    args = parser.parse_args()

    return args



if __name__ == "__main__":
    logs_path = "/Users/sakka/cnn_by_density_map/logs/evaluation.log"
    logging.basicConfig(filename=logs_path,
                        level=logging.DEBUG,
                        format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s")
    args = make_eval_parse()
    logger.debug("Running with args: {0}".format(args))
    evaluate(args)
