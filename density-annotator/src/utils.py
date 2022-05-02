import logging
import os
import sys
from glob import glob
from typing import List

import cv2
import numpy as np

# logging setting
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s"
)
logger = logging.getLogger(__name__)


def get_path_list(path: str, working_directory: str = "") -> List:
    """
    Takes a file or directory path and creates a list of full paths.

    :param path: input path that file or directory
    :param working_directory: current working directory.
        if input path provided full path, set an empty string.
    :return: full path list
    """
    full_path = os.path.join(working_directory, path)
    if os.path.isfile(full_path):
        path_list = [full_path]
    elif os.path.isdir(full_path):
        path_list = [current_path for current_path in glob(f"{full_path}/*")]
    else:
        logger.error(f"Error: Invalid Path Name: {full_path}")
        sys.exit(1)

    return path_list


def get_full_path_list(current_working_dirc: str, relative_path_list: List):
    """
    Join the current working directory name and relative path to get a list of full paths.

    :param current_working_dirc: current working directory name
    :param relative_path_list: list of relative paths to be converted
    :return: List of converted full path
    """
    full_path_list = [
        os.path.join(current_working_dirc, path) for path in relative_path_list
    ]
    return full_path_list


def get_input_data_type(path: str) -> str:
    """
    Get the extension from the input data path and get the data processing format.
    The target format is images or videos.

    :param path: file path to be annotated
    :return: processing format of annotator
    """
    data_type = "invalid"
    if os.path.isfile(path):
        _, ext = os.path.splitext(path)
        if ext in [".png", ".jpg", ".jpeg"]:
            data_type = "image"
        elif ext in [".mp4", ".mov"]:
            data_type = "video"

    logger.info(f"Input Data Type: {data_type}")

    return data_type


def load_image(path: str) -> np.array:
    """
    Loads image data from the input path and returns image in numpy array format.

    :param path: input image file path
    :return: loaded image
    """
    image = cv2.imread(path)
    if image is None:
        logger.error(
            f"Error: Can not read image file. Please check input file path. {path}"
        )
        sys.exit(1)
    logger.info(f"Loaded Image: {path}")

    return image


def load_video(path: str) -> cv2.VideoCapture:
    """
    Loads video data from the input path and returns video in cv2.VideoCapture format.

    :param path: input video file path
    :return: loaded video
    """
    video = cv2.VideoCapture(path)
    if not (video.isOpened()):
        logger.error(
            f"Error: Can not read video file. Please check input file path. {path}"
        )
        sys.exit(1)
    logger.info(f"Loaded Video: {path}")

    return video


def save_image(path: str, image: np.array) -> None:
    """
    Save the image data in numpy format in the target path.

    :param path: save path of image
    :param image: target image
    :return: None
    """
    cv2.imwrite(path, image)
    logger.info(f"Saved Image: {path}")


def save_coordinate(path: str, coordinate: np.array) -> None:
    """
    Save the coordinate data (x, y) in numpy format in the target path.

    :param path: save path of coordinate
    :param coordinate: coordinate of annotated points
    :return: None
    """
    np.savetxt(path, coordinate, delimiter=",", fmt="%d")
    logger.info(f"Saved Coordinate: {path}")


def save_density_map(path: str, density_map: np.array) -> None:
    """
    Save the density map data in numpy format in the target path.

    :param path: save path of density map
    :param density_map: annotated density map
    :return: None
    """
    np.save(path, density_map)
    logger.info(f"Save Density Map: {path}")
