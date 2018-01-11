#! /usr/bin/env python
# coding: utf-8

import os.path
import sys
import re
import numpy as np
import cv2

import movie2trainingData

# s key (save data and restart)
S_KEY = 0x73

class ImgMotion(movie2trainingData.Motion):
    # constructor
    def __init__(self, inputFilePath):
        super().__init__(inputFilePath)

    def run(self):
        self.frame = cv2.imread(self.inputFilePath)
        if self.frame is None:
            print("Error: Can not open image file")
            sys.exit(1)

        self.frameNum = self.get_frameNum()
        self.width = self.frame.shape[1]
        self.height = self.frame.shape[0]
        self.cordinateMatrix = np.zeros((self.width, self.height, 2), dtype="int64")
        for i in range(self.width):
            for j in range(self.height):
                self.cordinateMatrix[i][j] = [i, j]


        cv2.imshow("select feature points", self.frame)
        while(True):
            key = cv2.waitKey(0) & 0xFF
            if key == S_KEY:
                super().save_data()
                break
        cv2.destroyAllWindows()


    def get_frameNum(self):
        return self.inputFilePath.split("/")[-1].split(".")[0]


def batch_processing(inputDirPath):
    if not(os.path.isdir(inputDirPath)):
        print("Error: Do not exist directory")
        sys.exit(1)

    file_lst = os.listdir(inputDirPath)
    pattern = r"^(?!._).*(.png)$"
    repattern = re.compile(pattern)
    for fileName in file_lst:
        if re.search(repattern, fileName):
            filePath = inputDirPath + "/" + fileName
            print("File name: {}".format(filePath))
            ImgMotion(filePath).run()


if __name__ == "__main__":
    inputDirPath = input("input image directory path: ")
    batch_processing(inputDirPath)
