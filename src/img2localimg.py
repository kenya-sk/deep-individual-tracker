#! /usr/bin/env python
#coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import cv2

def read_csv(inputFilePath):
    cord_lst = []
    with open(inputFilePath, "r") as f:
        lines = f.readlines()
        for line in lines:
            cord_lst.append(line.strip().split(","))
    cord_arr = np.array(cord_lst, dtype=np.uint64)
    return cord_arr


def local_image(originalImg, cord ,size=15):
    widthMin = 0
    widthMax = int(originalImg.shape[1])
    heightMin = 0
    heightMax = int(originalImg.shape[0])
    band = int((size - 1)/2)
    # switch according to the number of channels
    if len(originalImg.shape) == 3:
        # color image
        localImg = np.zeros((originalImg.shape[2], size, size))
    else:
        # gray scale
        localImg = np.zeros((size, size))

    # edges of local image
    leftEdge = int(cord[0] - band)
    rightEdge = int(cord[0] + band)
    topEdge = int(cord[1] - band)
    bottomEdge = int(cord[1] + band)

    # check that the local image is in the original image
    left = abs(leftEdge - widthMin)
    right = abs(rightEdge - widthMax)
    top = abs(topEdge - heightMin)
    bottom = abs(bottomEdge - heightMax)

    if leftEdge < widthMin:
        if topEdge < heightMin:
            # over frame case1: left and top
            localImg[left:size, top:size] = originalImg.T[widthMin:rightEdge+1, heightMin:bottomEdge+1]
        else:
            # over frame case 2: left
            localImg[left:size, 0:size] = originalImg.T[widthMin:rightEdge+1, topEdge:bottomEdge+1]
        return localImg.T


    if topEdge < heightMin:
        if rightEdge > widthMax:
            # over frame case 3: top and right
            localImg[0:size-right, top:size] = originalImg.T[leftEdge:widthMax+1, heightMin:bottomEdge+1]
        else:
            # over frame case 4: top
            localImg[0:size, top:size] = originalImg.T[leftEdge:rightEdge+1, heightMin:bottomEdge+1]
        return localImg.T

    if rightEdge > widthMax:
        if bottomEdge > heightMax:
            # over frame case 5: right and bottom
            localImg[0:size-right, 0:size-bottom] = originalImg.T[leftEdge:widthMax+1, topEdge:heightMax+1]
        else:
            # over frame case 6: right
            localImg[0:size-right, 0:size] = originalImg.T[leftEdge:widthMax+1, topEdge:bottomEdge+1]
        return localImg.T

    if bottomEdge > heightMax:
        if leftEdge < widthMin:
            # over frame case 7: bottom and left
            localImg[left:size, 0:size-bottom] = originalImg.T[widthMin:rightEdge+1, topEdge:heightMax+1]
        else:
            # over frame case 8: bottom
            localImg[0:size, 0:size-bottom] = originalImg.T[leftEdge:rightEdge+1, topEdge:heightMax+1]
        return localImg.T

    # Not over frame
    localImg[0:size, 0:size] = originalImg.T[leftEdge:rightEdge+1, topEdge:bottomEdge+1]
    return localImg.T


if __name__ == "__main__":
    inputDensPath = input("Input file path (density map (.npy)): ")
    inputRawPath = input("Input file path (raw image (.png)): ")
    inputCordPath = input("Input file path (cordinate (.csv)): ")
    dens = np.load(inputDensPath)
    rawImg = np.array(cv2.imread(inputRawPath))
    cord_arr = read_csv(inputCordPath)
    fileName = inputDensPath.split("/")[-1].split(".")[0]

    for i, cord in enumerate(cord_arr):
        print("cordinate: {}".format(cord))
        rawLocalImg = local_image(rawImg, cord, 71)
        plt.imsave("../image/local/raw/{0}_{1}.png".format(fileName, i), rawLocalImg)
        np.save("../data/local/raw/{0}_{1}".format(fileName, i), rawLocalImg)
        densLocalImg = local_image(dens, cord, 15)
        plt.imsave("../image/local/density/{0}_{1}.png".format(fileName, i), densLocalImg)
        np.save("../data/local/density/{0}_{1}".format(fileName, i), densLocalImg)
