#! /usr/bin/env python
# coding: utf-8

import sys
import logging
import argparse
import numpy as np
import cv2

# q key (end)
Q_KEY = 0x71
# p key (pause)
P_KEY = 0x70
# s key (save data and restart)
S_KEY = 0x73


logger = logging.getLogger(__name__)
logs_path = "/Users/sakka/cnn_by_density_map/logs/movie2training_data.log"
logging.basicConfig(filename=logs_path,
                    level=logging.DEBUG,
                    format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s")


class Motion:
    # constructor
    def __init__(self, args):
        cv2.namedWindow("select feature points")
        cv2.setMouseCallback("select feature points", self.mouse_event)
        self.input_file_path = args.input_file_path
        self.interval = args.interval
        self.video = None
        self.frame = None
        self.width = None
        self.height = None
        self.features = None
        self.status = None
        self.cordinate_matrix = None
        self.frame_num = 0
        self.original_img_dirc = args.original_img_dirc
        self.save_truth_img_dirc = args.save_truth_img_dirc
        self.save_truth_cord_dirc = args.save_truth_cord_dirc
        self.save_answer_label_dirc = args.save_answer_label_dirc

    # main method
    def run(self):
        self.video = cv2.VideoCapture(self.input_file_path)
        if not(self.video.isOpened()):
            logger.error("Error: Can not read movie file")
            sys.exit(1)

        # processing of initial frame
        ret, self.frame = self.video.read()
        self.width = self.frame.shape[1]
        self.height = self.frame.shape[0]
        self.cordinate_matrix = np.zeros((self.width, self.height, 2), dtype="int64")
        for i in range(self.width):
            for j in range(self.height):
                self.cordinate_matrix[i][j] = [i, j]

        while ret:
            self.features = None
            self.frame_num += 1
            # display image
            cv2.imshow("select feature points", self.frame)
            # processing of next frame
            ret, self.frame = self.video.read()

            # each key operation
            key = cv2.waitKey(self.interval) & 0xFF
            if key == Q_KEY:
                break
            elif key == P_KEY:
                self.interval = 0
                # save original image
                cv2.imwrite(self.original_img_dirc + "{}.png".format(self.frame_num), self.frame)
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
        cv2.imshow("select feature points", self.frame)
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
            logger.error("Error: Not select feature point")
        else:
            cv2.imwrite(self.save_truth_img_dirc + "{}.png".format(self.frame_num), self.frame)
            np.savetxt(self.save_truth_cord_dirc + "{}.csv".format(self.frame_num), self.features, delimiter=",", fmt="%d")
            self.gauss_kernel(sigma_pow=25)
            logger.debug("save data frame number: {}".format(self.frame_num))
        return

    # calculate density map by gauss kernel
    def gauss_kernel(self, sigma_pow):
        kernel = np.zeros((self.width, self.height))

        for point in self.features:
            tmp_cord_matrix = np.array(self.cordinate_matrix)
            point_matrix = np.full((self.width, self.height, 2), point)
            diff_matrix = tmp_cord_matrix - point_matrix
            pow_matrix = diff_matrix * diff_matrix
            norm = pow_matrix[:, :, 0] + pow_matrix[:, :, 1]
            kernel += np.exp(-norm/ (2 * sigma_pow))

        np.save(self.save_answer_label_dirc + "{}.npy".format(self.frame_num), kernel.T)


def movie2train_parse():
    parser = argparse.ArgumentParser(
        prog="movie2training_data.py",
        usage="create training data at regular intarval",
        description="description",
        epilog="end",
        add_help=True
    )

    # Data Argment
    parser.add_argument("--input_file_path", type=str,
                        default="/data/sakka/movie/201704210900.mp4")

    # Parameter Argument
    parser.add_argument("--interval", type=int,
                        default=1, help="training data interval")
    parser.add_argument("--original_img_dirc", type=str,
                        default="/data/sakka/image/original/",
                        help="directory of raw image")
    parser.add_argument("--save_truth_img_dirc", type=str,
                        default="/data/sakka/image/grandTruth/",
                        help="directory of save annotation image")
    parser.add_argument("--save_truth_cord_dirc", type=str,
                        default="/data/sakka/data/cord/",
                        help="directory of save ground truth cordinate")
    parser.add_argument("--save_answer_label_dirc", type=str,
                        default="/data/sakka/data/dens/",
                        help="directory of save answer label (density map)")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = movie2train_parse()
    logger.debug("Running with args: {}".format(args))
    Motion(args).run()
