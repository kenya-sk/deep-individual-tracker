#! /usr/bin/env python
# coding: utf-8

import os.path
import sys
import logging
import glob
import argparse
import numpy as np
import cv2

import video2training_data

# q key (end)
Q_KEY = 0x71
# s key (save data and restart)
S_KEY = 0x73


logger = logging.getLogger(__name__)


class ImgMotion(video2training_data.Motion):
    # constructor
    def __init__(self, args, input_file_path):
        super().__init__(args)
        self.input_file_path = input_file_path

    def run(self):
        self.frame = cv2.imread(self.input_file_path)
        if self.frame is None:
            sys.stderr.write("Error: Can not open image file")
            sys.exit(1)

        self.frame_num = self.get_frame_num()
        self.width = self.frame.shape[1]
        self.height = self.frame.shape[0]
        self.cordinate_matrix = np.zeros((self.width, self.height, 2), dtype="int64")
        for i in range(self.width):
            for j in range(self.height):
                self.cordinate_matrix[i][j] = [i, j]


        cv2.imshow("select feature points", self.frame)
        # return bool value: stop making or not
        while(True):
            key = cv2.waitKey(0) & 0xFF
            if key == S_KEY:
                super().save_data()
                return False
            elif key==Q_KEY:
                return True

        cv2.destroyAllWindows()


    def get_frame_num(self):
        return self.input_file_path.split("/")[-1].split(".")[0]


def batch_processing(args):
    if not(os.path.isdir(args.input_img_dirc)):
        sys.stderr.write("Error: Do not exist directory")
        sys.exit(1)

    file_lst = glob.glob("{0}/*.png".format(args.input_img_dirc))
    logger.debug("Number of total file: {0}".format(len(file_lst)))

    for file_num, file_name in enumerate(file_lst):
        logger.debug("File name: {0}, Number: {1}/{2}".format(file_name, file_num+1, len(file_lst)))
        stop_making = ImgMotion(args, file_name).run()
        if stop_making:
            logger.debug("STOP: making datasets")
            break


def image2train_parse():
    parser = argparse.ArgumentParser(
        prog="image2training_data.py",
        usage="create training data from image",
        description="description",
        epilog="end",
        add_help=True
    )

    # Data Argment
    parser.add_argument("--input_img_dirc", type=str,
                        default="/Users/sakka/cnn_by_density_map/test_data/image/original")
    parser.add_argument("--input_file_path", type=str,
                        default="None")

    # Parameter Argument
    parser.add_argument("--interval", type=int,
                        default=None, help="training data interval")
    parser.add_argument("--original_img_dirc", type=str,
                        default=None,
                        help="directory of raw image")
    parser.add_argument("--save_truth_img_dirc", type=str,
                        default="/Users/sakka/cnn_by_density_map/test_data/image/truth",
                        help="directory of save annotation image")
    parser.add_argument("--save_truth_cord_dirc", type=str,
                        default="/Users/sakka/cnn_by_density_map/test_data/cord",
                        help="directory of save ground truth cordinate")
    parser.add_argument("--save_answer_label_dirc", type=str,
                        default="/Users/sakka/cnn_by_density_map/test_data/dens",
                        help="directory of save answer label (density map)")
    parser.add_argument("--save_file_prefix", type=str,
                        default="label",
                        help="put in front of the file name. ex) (--save_data_prefix)_1.png")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    logs_path = "/Users/sakka/cnn_by_density_map/logs/image2training_data.log"
    logging.basicConfig(filename=logs_path,
                        level=logging.DEBUG,
                        format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s")

    args = image2train_parse()
    logger.debug("Running with args: {}".format(args))
    batch_processing(args)
