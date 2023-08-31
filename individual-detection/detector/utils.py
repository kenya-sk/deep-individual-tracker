import os
import time
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.compat.v1.summary import FileWriter, merge_all

from detector.config import PredictConfig
from detector.exceptions import LoadVideoError
from detector.logger import logger

tf.compat.v1.disable_eager_execution()


def get_image_shape(image: np.ndarray) -> Tuple:
    """Get the height, width, and channel of the input image.

    Args:
        image (np.ndarray): input image

    Returns:
        Tuple: image shape=(height, width, channel)
    """
    height = image.shape[0]
    width = image.shape[1]
    if len(image.shape) == 3:
        channel = image.shape[2]
    else:
        channel = 1

    return (height, width, channel)


def display_data_info(
    input_path: Path, output_dirctory: Path, cfg: PredictConfig
) -> None:
    """Display setting information

    Args:
        input_path (Path): input data path
        output_dirctory (Path): output directory path
        cfg (PredictConfig): config for prediction

    Returns:
        None:
    """
    logger.info("*************************************************")
    logger.info(f"Input path          : {input_path}")
    logger.info(f"Output dirctory     : {output_dirctory}")
    logger.info(f"Skip pixel interval : {cfg.skip_pixel_interval}")
    logger.info(f"Pred batch size     : {cfg.predict_batch_size}")
    logger.info(f"Band width          : {cfg.band_width}")
    logger.info(f"Cluster threshold   : {cfg.cluster_threshold}")
    logger.info(f"Save density map    : {cfg.is_saved_map}")
    logger.info("*************************************************\n")


def set_capture(video_path: Path) -> Tuple:
    """Get input video information

    Args:
        video_path (Path): input video path

    Returns:
        Tuple: video information
    """
    # opencv cannot read Pathlib.Path format
    cap = cv2.VideoCapture(str(video_path))
    if cap is None:
        message = f'video_path="{video_path}" cannot load.'
        logger.error(message)
        raise LoadVideoError(message)

    fourcc = int(cv2.VideoWriter_fourcc(*"avc1"))
    fps = cap.get(cv2.CAP_PROP_FPS)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    logger.info("****************************************")
    logger.info(f"Video path             : {video_path}")
    logger.info(f"Fourcc                 : {fourcc}")
    logger.info(f"FPS                    : {fps}")
    logger.info(f"Size = (height, width) : ({height}, {width})")
    logger.info(f"Total frame            : {total_frame}")
    logger.info("****************************************")

    return cap, fourcc, fps, height, width, total_frame


def get_elapsed_time_str(start_time: float) -> str:
    """Format to display elapsed time from a specific time

    Args:
        start_time (float): elapsed time from unix epoch (get time.time())

    Returns:
        str: string representing elapsed time
    """
    total_elapsed_second = time.time() - start_time
    elapsed_hour = int(total_elapsed_second / 3600)
    elapsed_minute = int(total_elapsed_second % 3600 / 60)
    elapsed_second = int(total_elapsed_second % 60)

    return f"{elapsed_hour}[hour] {elapsed_minute}[min] {elapsed_second}[sec]"


def set_tensorboard(
    tensorboard_directory: Path,
    current_time_str: str,
    tf_session: InteractiveSession,
) -> Tuple:
    """Set tensorboard directory and writer

    Args:
        tensorboard_directory (Path): directory that save tensorflow log
        current_time_str (str): current time string
        tf_session (InteractiveSession): tensorflow session

    Returns:
        Tuple: tensorboard writers
    """
    # directory of TensorBoard
    log_directory = f"{tensorboard_directory}/{current_time_str}"
    logger.info(f"TensorBoard Directory: {log_directory}")

    # if the target directory exists, delete and recreate
    if tf.io.gfile.exists(log_directory):
        tf.compat.v1.gfile.DeleteRecursively(log_directory)
    tf.io.gfile.makedirs(log_directory)

    # set variable on TensorBoard
    summuray_merged = merge_all()
    train_writer = FileWriter(f"{log_directory}/train", tf_session.graph)
    valid_writer = FileWriter(f"{log_directory}/validation")
    test_writer = FileWriter(f"{log_directory}/test")

    return summuray_merged, train_writer, valid_writer, test_writer


def get_directory_list(root_path: Path) -> List:
    """Get a list of directories under the root path

    Args:
        root_path (Path): target root path

    Returns:
        List: list of directory
    """
    file_list = os.listdir(root_path)
    directory_list = [f for f in file_list if os.path.isdir(os.path.join(root_path, f))]

    return directory_list


def get_file_name_from_path(path: Path) -> str:
    """Get file name from file path.
    ex) path="./tmp/20170416_20111.png" -> file_name=20170416_20111

    Args:
        path (Path): file path

    Returns:
        int: extracted file name
    """
    file_name = str(path).split("/")[-1]
    return file_name.split(".")[0]
