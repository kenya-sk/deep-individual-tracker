#! /usr/bin/env python
# coding: utf-8

import os.path
import sys
import glob
import numpy as np
import cv2

import movie2training_data

# s key (save data and restart)
S_KEY = 0x73

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
        while(True):
            key = cv2.waitKey(0) & 0xFF
            if key == S_KEY:
                super().save_data()
                break
        cv2.destroyAllWindows()


    def get_frame_num(self):
        return self.input_file_path.split("/")[-1].split(".")[0]


def batch_processing(input_dirc_path):
    if not(os.path.isdir(input_dirc_path)):
        sys.stderr.write("Error: Do not exist directory")
        sys.exit(1)

    file_lst = glob.glob(input_dirc_path+"*.png")
    print("Number of total file: {}".format(len(file_lst)))

    for file_num, file_name in enumerate(file_lst):
        print("File name: {0}, Number: {1}/{2}".format(file_name, file_num, len(file_lst)))
        ImgMotion(file_name).run()


if __name__ == "__main__":
    input_dirc_path = input("input image directory path: ")
    batch_processing(input_dirc_path)
