import logging
import math
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import List, Tuple

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.compat.v1.summary import FileWriter, merge_all

from model import DensityModel

logger = logging.getLogger(__name__)


def display_data_info(input_path: str, output_dirctory: str, cfg: dict) -> None:
    """Display setting information

    Args:
        input_path (str): input data path
        output_dirctory (str): output directory
        cfg: config dictionary of parameter

    Returns:
        None:
    """
    logger.info("*************************************************")
    logger.info(f"Input path         : {input_path}")
    logger.info(f"Output dirctory    : {output_dirctory}")
    logger.info(f"Skip width         : {cfg['skip_width']}")
    logger.info(f"Pred batch size    : {cfg['predict_batch_size']}")
    logger.info(f"Band width         : {cfg['band_width']}")
    logger.info(f"Cluster threshold  : {cfg['cluster_thresh']}")
    logger.info(f"Save density map   : {cfg['is_saved_map']}")
    logger.info("*************************************************\n")


def eval_metrics(
    true_positive: int, false_positive: int, false_negative: int, sample_num: int
) -> Tuple:
    """Calculate accuracy, precision, recall, and f_measure.

    Args:
        true_positive (int): the number of true positive
        false_positive (int): the number of false positive
        false_negative (int): the number of false negative
        sample_num (int): the number of sample

    Returns:
        Tuple: calculated each metrics
    """
    accuracy = true_positive / sample_num
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    f_measure = (2 * recall * precision) / (recall + precision)
    return accuracy, precision, recall, f_measure


def pretty_print(
    true_positive_list: List,
    false_positive_list: List,
    false_negative_list: List,
    sample_num_list: List,
) -> None:
    """Outputs a formatted summary of evaluation results to the log.

    Args:
        true_positive_list (List): list containing the number of true-positive in each frame
        false_positive_list (List): list containing the number of false-positive in each frame
        false_negative_list (List): list containing the number of false-negative in each frame
        sample_num_list (List): list containing the number of sample in each frame
    """
    accuracy_list = []
    precision_list = []
    recall_list = []
    f_measure_list = []
    assert (
        len(true_positive_list)
        == len(false_positive_list)
        == len(false_negative_list)
        == len(sample_num_list)
    ), "List of each evaluation result are not same length."
    data_size = len(true_positive_list)
    for i in range(data_size):
        accuracy, precision, recall, f_measure = eval_metrics(
            true_positive_list[i],
            false_positive_list[i],
            false_negative_list[i],
            sample_num_list[i],
        )
        accuracy_list.append(accuracy)
        precision_list.append(precision)
        recall_list.append(recall)
        f_measure_list.append(f_measure)

    logger.info("\n**************************************************************")
    logger.info(f"Total data size: {data_size}")
    logger.info(f"Total true-positive number: {sum(true_positive_list)}")
    logger.info(f"Total false-positive number: {sum(false_positive_list)}")
    logger.info(f"Total false-negative number: {sum(false_negative_list)}")
    logger.info("Total false-positive number: /")
    logger.info(f"Total Accuracy: {np.mean(accuracy_list):.2f}")
    logger.info(f"Total Precision: {np.mean(precision_list):.2f}")
    logger.info(f"Total Recall: {np.mean(recall_list):.2f}")
    logger.info(f"Total F-measure: {np.mean(f_measure_list):.2f}")
    logger.info("****************************************************************")


def load_image(path: str, is_rgb: bool = True) -> np.array:
    """Loads image data frosm the input path and returns image in numpy array format.

    Args:
        path (str): input image file path
        is_rgb (bool, optional): whether convert RGB format. Defaults to True.

    Returns:
        np.array: loaded image
    """
    image = cv2.imread(path)
    if image is None:
        logger.error(
            f"Error: Can not read image file. Please check input file path. {path}"
        )
        sys.exit(1)
    logger.info(f"Loaded Image: {path}")

    # convert image BGR to RGB
    if is_rgb:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


def load_sample(
    X_path: str,
    y_path: str,
    input_image_shape: Tuple,
    mask_image: np.array,
    is_rgb: bool = True,
) -> Tuple:
    """_summary_

    Args:
        X_path (str): _description_
        y_path (str): _description_
        input_image_shape (Tuple): _description_
        mask_image (np.array): _description_
        is_rgb (bool, optional): _description_. Defaults to True.

    Returns:
        Tuple: _description_
    """
    X_image = load_image(X_path, is_rgb=is_rgb)
    assert (
        X_image.shape == input_image_shape
    ), f"Invalid image shape. Expected is {input_image_shape} but {X_image.shape}"

    y_dens = np.load(y_path)
    assert (
        X_image.shape[:2] == y_dens.shape
    ), f"The input image and the density map must have the same shape.\
    image={X_image.shape[:2]}, density map={y_dens.shape}"

    if mask_image is not None:
        X_image = apply_masking_on_image(X_image, mask_image, channel=3)
        y_dens = apply_masking_on_image(y_dens, mask_image, channel=1)

    return X_image, y_dens


def load_mask_image(mask_path: str = None, normalized: bool = True) -> np.array:
    """Load a binary mask image and normalizes the values as necessary.

    Args:
        mask_path (str, optional): binary mask image path. Defaults to None.
        normalized (bool, optional): whether execute normalization. Defaults to True.

    Returns:
        np.array: loaded binary masked image
    """
    if mask_path is not None:
        # load binary mask image
        mask = cv2.imread(mask_path)
        assert (
            1 <= len(np.unique(mask)) <= 2
        ), f"Error: mask image is not binary. (current unique value={len(np.unique(mask))})"

        # normalize mask image to (min, max)=(0, 1)
        if normalized:
            mask = np.array(mask / np.max(mask), dtype="uint8")
    else:
        mask = None

    return mask


def apply_masking_on_image(
    image: np.array, mask: np.array, channel: int = 3
) -> np.array:
    """Apply mask processing to image data.

    Args:
        image (np.array): image to be applied
        mask (np.array): mask image
        channel (int, optional): channel number of applied image. Defaults to 3.

    Returns:
        np.array: masked image
    """
    # apply mask to image
    if channel == 3:
        masked_image = image * mask
    else:
        masked_image = image * mask[:, :, 0]

    return masked_image


def get_masked_index(
    mask: np.array, params_dict: dict, horizontal_flip: bool = False
) -> Tuple:
    """Masking an image to get valid index

    Args:
        mask (np.array): binay mask image
        params_dict (dict): dictionary of parameters
        horizontal_flip (bool, optional): Whether to perform data augumentation. Defaults to False.

    Returns:
        Tuple: valid index list (heiht and width)
    """
    if mask is None:
        mask = np.ones((params_dict["image_height"], params_dict["image_width"]))

    # convert gray scale image
    if (len(mask.shape) == 3) and (mask.shape[2] == 3):
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # crop the image to the analysis area
    mask = mask[
        params_dict["analysis_image_height_min"] : params_dict[
            "analysis_image_height_max"
        ],
        params_dict["analysis_image_width_min"] : params_dict[
            "analysis_image_width_max"
        ],
    ]

    # index of data augumentation
    if horizontal_flip:
        mask = mask[:, ::-1]

    index = np.where(mask > 0)
    index_h = index[0]
    index_w = index[1]
    assert len(index_h) == len(index_w)

    return index_h, index_w


def get_local_data(
    image: np.array, density_map: np.array, params_dict: dict, is_flip: bool
) -> Tuple:
    """Get local image and density map from raw data

    Args:
        image (np.array): raw image
        density_map (np.array): raw density map
        params_dict (dict): dictionary of parameters
        is_flip (book): whether image is flip or not

    Returns:
        Tuple: numpy array of local image and density map
    """

    assert len(image.shape) == 3
    # triming original image
    image = image[
        params_dict["analysis_image_height_min"] : params_dict[
            "analysis_image_height_max"
        ],
        params_dict["analysis_image_width_min"] : params_dict[
            "analysis_image_width_max"
        ],
    ]
    height = image.shape[0]
    width = image.shape[1]
    channel = image.shape[2]

    pad = math.floor(params_dict["local_image_size"] / 2)
    pad_image = np.zeros((height + pad * 2, width + pad * 2, channel), dtype="uint8")
    pad_image[pad : pad + height, pad : pad + width] = image

    # get each axis index
    if is_flip:
        index_h = params_dict["flip_index_h"]
        index_w = params_dict["flip_index_w"]
    else:
        index_h = params_dict["index_h"]
        index_w = params_dict["index_w"]

    # extract local image
    assert len(index_w) == len(
        index_h
    ), "The number of indexes differs for height and width. It is expected that they will be the same number."
    local_data_number = len(index_w)
    local_image_array = np.zeros(
        (
            local_data_number,
            params_dict["local_image_size"],
            params_dict["local_image_size"],
            channel,
        ),
        dtype="uint8",
    )

    density_array = np.zeros((local_data_number), dtype="float32")
    for idx in range(local_data_number):
        # raw image index convert to padding image index
        h = index_h[idx]
        w = index_w[idx]
        local_image_array[idx] = pad_image[h : h + 2 * pad, w : w + 2 * pad]
        if density_map is not None:
            density_array[idx] = density_map[h, w]

    return local_image_array, density_array


def load_model(model_path: str, device_id: str, memory_rate: float) -> Tuple:
    """Load trained Convolutional Neural Network model that defined by TensorFlow

    Args:
        model_path (str): path of trained model
        device_id (str): GPU divice ID
        memory_rate (float): use rate of GPU memory (0.0-1.0)

    Returns:
        Tuple: loaded model and tensorflow session
    """

    config = tf.ConfigProto(
        gpu_options=tf.GPUOptions(
            visible_device_list=device_id, per_process_gpu_memory_fraction=memory_rate
        )
    )
    sess = InteractiveSession(config=config)

    model = DensityModel()
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(model_path)
    if ckpt:
        last_model = ckpt.model_checkpoint_path
        logger.info("LODE MODEL: {}".format(last_model))
        saver.restore(sess, last_model)
    else:
        logger.error("Eroor: Not exist model!")
        logger.error("Please check model_path")
        sys.exit(1)

    return model, sess


def set_capture(video_path: str) -> Tuple:
    """Get input video information

    Args:
        video_path (str): input video path

    Returns:
        Tuple: video information
    """
    cap = cv2.VideoCapture(video_path)
    if cap is None:
        logger.error("ERROR: Not exsit video")
        logger.error("Please check video path: {0}".format(video_path))
        sys.exit(1)
    fourcc = int(cv2.VideoWriter_fourcc(*"avc1"))
    fps = cap.get(cv2.CAP_PROP_FPS)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    logger.info(f"****************************************")
    logger.info(f"Video path             : {video_path}")
    logger.info(f"Fourcc                 : {fourcc}")
    logger.info(f"FPS                    : {fps}")
    logger.info(f"Size = (height, width) : ({height}, {width})")
    logger.info(f"Total frame            : {total_frame}")
    logger.info("****************************************")

    return cap, fourcc, fps, height, width, total_frame


def get_current_time_str(
    time_difference: int = 9, time_format: str = "%Y%m%d_%H%M%S"
) -> str:
    """Get the current time and convert formatted string.

    Args:
        time_difference (int, optional): time difference from UTC time. Defaults to 9.
        time_format (str, optional): string format. Defaults to "%Y%m%d%H%M%S".

    Returns:
        str: formatted current time string
    """
    current_time = datetime.now(timezone(timedelta(hours=time_difference)))
    str_current_time = current_time.strftime(time_format)

    return str_current_time


def get_elapsed_time_str(start_time: float) -> str:
    """Format to display elapsed time from a specific time

    Args:
        start_time (float): elapsed time from unix epoch (get time.time())

    Returns:
        str: string representing elapsed time
    """
    total_elpased_second = time.time() - start_time
    elapsed_hour = int(total_elpased_second / 3600)
    elapsed_minute = int(total_elpased_second % 3600 / 60)
    elapsed_second = int(total_elpased_second % 60)

    return f"{elapsed_hour}[hour] {elapsed_minute}[min] {elapsed_second}[sec]"


def set_tensorboard(
    tensorboard_directory: str,
    current_time_str: str,
    tf_session: InteractiveSession,
) -> Tuple:
    """Set tensorboard directory and writer

    Args:
        tensorboard_directory (str): directory that save tensorflow log
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


def get_directory_list(root_path: str) -> List:
    """Get a list of directories under the root path

    Args:
        root_path (str): target root path

    Returns:
        List: list of directory
    """
    file_list = os.listdir(root_path)
    directory_list = [f for f in file_list if os.path.isdir(os.path.join(root_path, f))]

    return directory_list


def get_frame_number_from_path(path: str) -> int:
    """Get frame number from file path.
    ex) path="./tmp/20111.png" -> frame_number=20111

    Args:
        path (str): file path

    Returns:
        int: extracted frame number
    """
    file_name = path.split("/")[-1]
    frame_num = int(file_name.split(".")[0])
    return frame_num


def save_dataset_path(X_path_list: List, y_path_list: List, save_path: str) -> None:
    """_summary_

    Args:
        X_path_list (List): _description_
        y_path_list (List): _description_
        save_path (str): _description_
    """
    pd.DataFrame({"X_path": X_path_list, "y_path": y_path_list}).to_csv(
        save_path, index=False
    )
