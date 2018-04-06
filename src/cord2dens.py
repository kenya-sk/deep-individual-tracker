#! /usr/bin/env python
#coding: utf-8

import os
import sys
import cv2
import re
import numpy as np


def plot_densMap(fileName, cordinateMatrix, cord_arr, sigmaPow):
    width, height = 1280, 720
    kernel = np.zeros((width, height))

    for point in cord_arr:
        tmpCordMatrix = np.array(cordinateMatrix)
        pointMatrix = np.full((width, height, 2), point)
        diffMatrix = tmpCordMatrix - pointMatrix
        powMatrix = diffMatrix * diffMatrix
        norm = powMatrix[:, :, 0] + powMatrix[:, :, 1]
        kernel += np.exp(-norm/ (2 * sigmaPow))

    fileName = fileName.replace(".csv", "")
    np.save("../data/dens/{0}/{1}".format(sigmaPow, fileName), kernel.T)


def batch_processing(inputDirPath, sigmaPow_lst):
    def read_csv(filePath):
        data_lst = []
        with open(filePath, "r") as f:
            lines = f.readlines()
            for data in lines:
                data_lst.append(data.strip().split(","))

        return np.array(data_lst).astype(np.int64)


    if not(os.path.isdir(inputDirPath)):
        sys.stderr.write("Error: Do not exist directory !")
        sys.exit(1)

    file_lst = os.listdir(inputDirPath)
    pattern = r"^(?!._).*(.csv)$"
    repattern = re.compile(pattern)
    print("Number of total file: {}".format(len(file_lst)))

    width, height = 1280, 720
    cordinateMatrix = np.zeros((width, height, 2), dtype="int64")
    for i in range(width):
        for j in range(height):
            cordinateMatrix[i][j] = [i, j]
    for sigmaPow in sigmaPow_lst:
        print("sigma pow: {}".format(sigmaPow))
        fileNum = 0
        for fileName in file_lst:
            if re.search(repattern, fileName):
                filePath = inputDirPath + "/" + fileName
                print("Number: {0}, File name: {1}".format(fileNum, filePath))
                cord_arr = read_csv(filePath)
                plot_densMap(fileName, cordinateMatrix, cord_arr, sigmaPow)
                fileNum += 1
    print("End: batch processing")


if __name__ == "__main__":
    inputDirPath = input("Input directory path: ")
    sigmaPow_lst = [8, 10, 15, 20]
    batch_processing(inputDirPath, sigmaPow_lst)
