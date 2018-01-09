#! /usr/bin/env python
# coding: utf-8

import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import cv2

# q key (end)
Q_KEY = 0x71
# p key (pause)
P_KEY = 0x70
# s key (save data and restart)
S_KEY = 0x73
# intervel of receive key
INTERVAL = 1


class Motion:
    # constructor
    def __init__(self, inputFilePath):
        cv2.namedWindow("select feature point")
        cv2.setMouseCallback("select feature point", self.mouse_event)
        self.video = cv2.VideoCapture(inputFilePath)
        self.interval = INTERVAL
        self.frame = None
        self.width = None
        self.height = None
        self.features = None
        self.status = None
        self.cordinateMatrix = None
        self.frameNum = 0


    # main method
    def run(self):
        if not(self.video.isOpened()):
            print("Can not read movie file")
            sys.exit(1)

        # processing of initial frame
        ret, self.frame = self.video.read()
        self.width = self.frame.shape[1]
        self.height = self.frame.shape[0]
        self.cordinateMatrix = np.zeros((self.width, self.height, 2), dtype="int64")
        for i in range(self.width):
            for j in range(self.height):
                self.cordinateMatrix[i][j] = [i, j]

        while ret:
            self.features = None
            self.frameNum += 1
            # display image
            cv2.imshow("select feature point", self.frame)
            # processing of next frame
            ret, self.frame = self.video.read()

            # each key operation
            key = cv2.waitKey(self.interval) & 0xFF
            if key == Q_KEY:
                break
            elif key == P_KEY:
                self.interval = 0
            elif key == S_KEY:
                self.save_data()
                self.interval = INTERVAL

        # end processing
        cv2.destroyAllWindows()
        self.video.release()


    # select feature point by mouse left click
    def mouse_event(self, event, x, y, flags, param):
        # other than left click
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        # draw and add feature point
        cv2.circle(self.frame, (x, y), 4, (0, 0, 255), -1, 8, 0)
        self.add_feature(x, y)
        cv2.imshow("select feature point", self.frame)
        return


    # add new feature point
    def add_feature(self, x, y):
        if self.features is None:
            self.features = np.array([[x, y]], np.uint16)
            self.status = np.array([1])
        else:
            self.features = np.append(self.features, [[x, y]], axis=0).astype(np.uint16)
            self.status = np.append(self.status, 1)


    # save cordinate and figure. there are feature point information
    def save_data(self):
        if self.features is None:
            print("Not select feature point")
        else:
            cv2.imwrite("../image/{}.png".format(self.frameNum), self.frame)
            #convert: opencv axis -> matplotlib axis
            self.features[:, 1] = self.frame.shape[0] - self.features[:, 1]
            np.savetxt("../data/cord/{}.csv".format(self.frameNum), self.features, delimiter=",", fmt="%d")
            self.gauss_kernel(sigmaPow=4)
            print("save data frame number: {}".format(self.frameNum))
        return

    # calculate density map by gauss kernel
    def gauss_kernel(self, sigmaPow):
        kernel = np.zeros((self.width, self.height))

        for point in self.features:
            tmpCordMatrix = np.array(self.cordinateMatrix)
            pointMatrix = np.full((self.width, self.height, 2), point)
            diffMatrix = tmpCordMatrix - pointMatrix
            powMatrix = diffMatrix * diffMatrix
            norm = powMatrix[:, :, 0] + powMatrix[:, :, 1]
            kernel += np.exp(-norm/ (2 * sigmaPow))

        np.save("../data/dens/0422_10_{}".format(self.frameNum), kernel)


if __name__ == "__main__":
    inputFilePath = input("input movie file path: ")
    Motion(inputFilePath).run()
