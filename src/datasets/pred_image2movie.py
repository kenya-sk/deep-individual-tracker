#! /usr/bin/env python
# coding: utf-8

import sys
import glob
import cv2
import numpy as np
from tqdm import tqdm

def pred_image2movie(image_dirc_path, cord_dirc_path, params_dict):
    file_num = len(glob.glob(image_dirc_path + "/*.png"))
    fourcc = cv2.VideoWriter_fourcc("a", "v", "c", "1")
    fps = params_dict["fps"]
    width = params_dict["width"]
    height = params_dict["height]
    out = cv2.VideoWriter("./out_movie.mp4", int(fourcc), fps, (int(width), int(height)))

    for i in tqdm(range(1,file_num+1)):
        img = cv2.imread(image_dirc_path + "/{}.png".format(i))
        cord_arr = np.load(cord_dirc_path + "/{}.npy".format(i))

        for cord in cord_arr:
            cv2.circle(img, (int(cord[0]), int(cord[1])), 3, (0, 0, 255), -1, cv2.LINE_AA)
        out.write(img)

    cv2.destroyAllWindows()
    print("SAVE: prediction movie (./out_movie.mp4)")

if __name__ == "__main__":
    image_dirc_path = input("Input path of estimate image direcory: ")
    cord_dirc_path = input("Input path of estimate cordinate direcory: ")
    params_dict = {"fps": 30, "width": 1280, "height": 720}
    pred_image2movie(image_dirc_path, cord_dirc_path)
