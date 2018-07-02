#! /usr/bin/env python
#coding: utf-8

import os
import sys
import cv2
import re
import numpy as np


def plot_densMap(file_name, cordinate_matrix, cord_arr, sigma_pow):
    width, height = 1280, 720
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


def batch_processing(input_dirc_path, sigma_pow_lst):
    def read_csv(file_path):
        data_lst = []
        with open(file_path, "r") as f:
            lines = f.readlines()
            for data in lines:
                data_lst.append(data.strip().split(","))

        return np.array(data_lst).astype(np.int64)


    if not(os.path.isdir(input_dirc_path)):
        sys.stderr.write("Error: Do not exist directory !")
        sys.exit(1)

    file_lst = os.listdir(input_dirc_path)
    pattern = r"^(?!._).*(.csv)$"
    repattern = re.compile(pattern)
    print("Number of total file: {}".format(len(file_lst)))

    width, height = 1280, 720
    cordinate_matrix = np.zeros((width, height, 2), dtype="int64")
    for i in range(width):
        for j in range(height):
            cordinate_matrix[i][j] = [i, j]
    for sigma_pow in sigma_pow_lst:
        print("sigma pow: {}".format(sigma_pow))
        fileNum = 0
        for file_name in file_lst:
            if re.search(repattern, file_name):
                file_path = input_dirc_path + "/" + file_name
                print("Number: {0}, File name: {1}".format(fileNum, file_path))
                cord_arr = read_csv(file_path)
                plot_densMap(file_name, cordinate_matrix, cord_arr, sigma_pow)
                fileNum += 1
    print("End: batch processing")


if __name__ == "__main__":
    input_dirc_path = input("Input directory path: ")
    sigma_pow_lst = [8, 10, 15, 20, 25]
    batch_processing(input_dirc_path, sigma_pow_lst)
