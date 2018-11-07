#! /usr/bin/env python
# coding: utf-8

import sys
import glob
import logging
import argparse
import cv2
import numpy as np
from tqdm import tqdm


logger = logging.getLogger(__name__)
logs_path = "/Users/sakka/cnn_by_density_map/logs/pred_image2movie.log"
logging.basicConfig(filename=logs_path,
                    level=logging.DEBUG,
                    format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s")


def pred_image2movie(args):
    file_num = len(glob.glob("{0}/*.png".format(args.img_dirc)))
    fourcc = cv2.VideoWriter_fourcc("a", "v", "c", "1")
    fps = args.fps
    width = args.width
    height = args.height
    out = cv2.VideoWriter(args.save_movie_path, int(fourcc), fps, (int(width), int(height)))

    for i in tqdm(range(1,file_num+1)):
        img = cv2.imread("{0}/{1}.png".format(args.img_dirc, i))
        cord_arr = np.load("{0}/{1}.npy".format(args.cord_dirc, i))

        for cord in cord_arr:
            cv2.circle(img, (int(cord[0]), int(cord[1])), 3, (0, 0, 255), -1, cv2.LINE_AA)
        out.write(img)

    cv2.destroyAllWindows()
    logger.debug("SAVE: prediction movie (./{0})".format(args.save_movie_path))


def img2movie_parse():
    parser = argparse.ArgumentParser(
        prog="pred_image2movie.py",
        usage="create movie to detect target object",
        description="description",
        epilog="end",
        add_help=True
    )

    # Data Argment
    parser.add_argument("--img_dirc", type=str,
                        default="/data/sakka/image",
                        help="input path of estimate image direcory")
    parser.add_argument("--cord_dirc", type=str,
                        default="/data/sakka/estimation/cord",
                        help="input path of estimate cordinate direcory")
    parser.add_argument("--save_movie_path", type=str,
                        default="/data/sakka/movie/out.mp4")

    # Parameter Argument
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int,
                        default=1280, help="width of input image")
    parser.add_argument("--height", type=int,
                        default=720, help="height of input image")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = img2movie_parse()
    logger.debug("Running with args: {0}".format(args))
    pred_image2movie(args)