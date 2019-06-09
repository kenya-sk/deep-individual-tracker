#! /usr/bin/env python
#coding: utf-8

import os
import sys
import logging
import cv2
import re
import numpy as np


logger = logging.getLogger(__name__)


def plot_densMap(file_name, cordinate_matrix, cord_arr, sigma_pow, args):
    """
    """
    width = args.width
    height = args.height
    kernel = np.zeros((width, height))

    for point in cord_arr:
        tmp_cord_matrix = np.array(cordinate_matrix)
        point_matrix = np.full((width, height, 2), point)
        diff_matrix = tmp_cord_matrix - point_matrix
        pow_matrix = diff_matrix * diff_matrix
        norm = pow_matrix[:, :, 0] + pow_matrix[:, :, 1]
        kernel += np.exp(-norm/ (2 * sigma_pow))

    file_name = file_name.replace(".csv", "")
    np.save("../data/dens/{0}/{1}".format(sigma_pow, file_name), kernel.T)


def batch_processing(args):
    """
    """

    def read_csv(file_path):
        data_lst = []
        with open(file_path, "r") as f:
            lines = f.readlines()
            for data in lines:
                data_lst.append(data.strip().split(","))

        return np.array(data_lst).astype(np.int64)


    if not(os.path.isdir(args.cord_dirc)):
        logger.error("Error: Do not exist directory !")
        sys.exit(1)

    file_lst = os.listdir(args.cord_dirc)
    pattern = r"^(?!._).*(.csv)$"
    repattern = re.compile(pattern)
    logger.debug("Number of total file: {0}".format(len(file_lst)))

    width = args.width
    height = args.height
    cordinate_matrix = np.zeros((width, height, 2), dtype="int64")
    for i in range(width):
        for j in range(height):
            cordinate_matrix[i][j] = [i, j]
    for sigma_pow in args.sigma_pow_lst:
        logger.debug("sigma pow: {0}".format(sigma_pow))
        fileNum = 0
        for file_name in file_lst:
            if re.search(repattern, file_name):
                file_path = args.cord_dirc + file_name
                logger.debug("Number: {0}, File name: {1}".format(fileNum, file_path))
                cord_arr = read_csv(file_path)
                plot_densMap(file_name, cordinate_matrix, cord_arr, sigma_pow)
                fileNum += 1
    logger.debug("End: batch processing")


def cord2dens_parse():
    parser = argparse.ArgumentParser(
        prog="cord2dens.py",
        usage="calculate density map by input coordinate and several sigma power",
        description="description",
        epilog="end",
        add_help=True
    )

    # Data Argment
    parser.add_argument("--cord_dirc", type=str,
                        default="/data/sakka/image")

    # Parameter Argument
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--sigma_pow_lst", type=list, default=[8, 10, 15, 20, 25])

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    # set logger
    logs_path = "../../logs/cord2dens.log"
    logging.basicConfig(filename=logs_path,
                        level=logging.DEBUG,
                        format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s")
    # set argument
    args = cord2dens_parse()
    logger.debug("Running with args: {0}".format(args))

    batch_processing(args)
