#! /usr/bin/env python
# coding: utf-8

import os.path
import sys
import logging
import glob
import numpy as np
import cv2

import movie2training_data

# q key (end)
Q_KEY = 0x71
# s key (save data and restart)
S_KEY = 0x73


logger = logging.getLogger(__name__)
logs_path = "/Users/sakka/cnn_by_density_map/logs/image2training_data.log"
logging.basicConfig(filename=logs_path,
                    leval=logging.DEBUG,
                    format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s")

class ImgMotion(movie2training_data.Motion):
    # constructor
    def __init__(self, input_file_path):
        super().__init__(input_file_path)

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


def batch_processing(input_img_dirc):
    if not(os.path.isdir(input_img_dirc)):
        sys.stderr.write("Error: Do not exist directory")
        sys.exit(1)

    file_lst = glob.glob(input_img_dirc+"*.png")
    logger.debug("Number of total file: {}".format(len(file_lst)))

    for file_num, file_name in enumerate(file_lst):
        logger.debug("File name: {0}, Number: {1}/{2}".format(file_name, file_num+1, len(file_lst)))
        stop_making = ImgMotion(file_name).run()
        if stop_making:
            logger.debug("STOP: making datasets")
            break


if __name__ == "__main__":
    input_img_dirc = input("input image directory path: ")
    logger.debug("input image directory: {}".format(input_img_dirc))
    batch_processing(input_img_dirc)
