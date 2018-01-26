#! /usr/bin/env python
#coding: utf-8

import sys
import cv2
import re
import numpy as np


def read_cord(inputDirPath):
    if not(os.path.isdir(inputDirPath)):
        print("Error: Do not exist directory !")
        sys.exit(1)

    file_lst = os.listdir(inputDirPath)
    pattern = r"^(?!._).*"

def plot_densMap(cord, sigmaPow):



if __name__ == "__main__":
    inputDirPath = input()
    cord = read_cord(inputDirPath)
    dens = plot_densMap(cord, sigmaPow)
