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
    cord_arr = np.array(cord_lst, dtype=np.uint32)
    return cord_arr


def local_image(originalImg, cord ,size=15):
    widthMin = 0
    widthMax = originalImg.shape[1]
    heightMin = 0
    heightMax = originalImg.shape[0]
    band = int((size - 1)/2)
    localImg = np.zeros((size, size))

    # edges of local image
    leftEdge = cord[0] - band
    rightEdge = cord[0] + band
    topEdge = cord[1] - band
    bottomEdge = cord[1] + band

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
        return localImg


    if topEdge < heightMin:
        if rightEdge > widthMax:
            # over frame case 3: top and right
            localImg[0:size-right, top:size] = originalImg.T[leftEdge:widthMax, heightMin:bottomEdge+1]
        else:
            # over frame case 4: top
            localImg[0:size, top:size] = originalImg.T[leftEdge:rightEdge+1, heightMin:bottomEdge+1]
        return localImg

    if rightEdge > widthMax:
        if bottomEdge > heightMax:
            # over frame case 5: right and bottom
            localImg[0:size-right, 0:size-bottom] = originalImg.T[leftEdge:widthMax, topEdge:heightMax]
        else:
            # over frame case 6: bottom
            localImg[0:size, 0:size-bottom] = originalImg.T[leftEdge:rightEdge+1, topEdge:heightMax]
        return localImg

    if bottomEdge > heightMax:
        if leftEdge < widthMin:
            # over frame case 7: bottom and left
            localImg[left:size, 0:size-bottom] = originalImg.T[widthMin:rightEdge+1, topEdge:heightMax]
        else:
            # over frame case 8: left
            localImg[left:size, 0:size] = originalImg.T[widthMin:rightEdge+1, topEdge:bottomEdge+1]
        return localImg

    # Not over frame
    localImg[0:size, 0:size] = originalImg.T[leftEdge:rightEdge+1, topEdge:bottomEdge+1]
    return localImg

if __name__ == "__main__":
    inputDensPath = input("Input file path (density map (.npy)): ")
    inputCordPath = input("Input file path (cordinate (.csv)): ")
    dens = np.load(inputDensPath)
    cord_arr = read_csv(inputCordPath)
    fileName = inputDensPath.split("/")[-1].split(".")[0]
    for i, cord in enumerate(cord_arr):
        print("cordinate: {}".format(cord))
        localImg = local_image(dens, cord, 15)
        plt.imsave("../image/local/{0}_{1}.png".format(fileName, i), localImg)
