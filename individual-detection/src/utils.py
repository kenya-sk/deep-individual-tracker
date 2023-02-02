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
    logger.info(f"Input path          : {input_path}")
    logger.info(f"Output dirctory     : {output_dirctory}")
    logger.info(f"Skip pixel interval : {cfg['skip_pixel_interval']}")
    logger.info(f"Pred batch size     : {cfg['predict_batch_size']}")
    logger.info(f"Band width          : {cfg['band_width']}")
    logger.info(f"Cluster threshold   : {cfg['cluster_thresh']}")
    logger.info(f"Save density map    : {cfg['is_saved_map']}")
    logger.info("*************************************************\n")


def load_image(path: str, is_rgb: bool = True, normalized: bool = False) -> np.array:
    """Loads image data frosm the input path and returns image in numpy array format.

    Args:
        path (str): input image file path
        is_rgb (bool, optional): whether convert RGB format. Defaults to True.
        normalized (bool, optional): whether normalize loaded image. Defaults to False.

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

    # normalization
    if normalized:
        image = image / 255.0

    return image


def load_sample(
    X_path: str,
    y_path: str,
    input_image_shape: Tuple,
    mask_image: np.array,
    is_rgb: bool,
    normalized: bool,
) -> Tuple:
    """Load samples for input into the model.
    Each sample is a set of image and label, with mask image applied as needed.

    Args:
        X_path (str): x (input) path
        y_path (str): y (label) path
        input_image_shape (Tuple): raw input image shape
        mask_image (np.array): mask image array
        is_rgb (bool): whether to convert the image to RGB format
        normalized (bool): whether to normalize the image

    Returns:
        Tuple: raw image and label set
    """
    X_image = load_image(X_path, is_rgb=is_rgb, normalized=normalized)
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


def get_image_shape(image: np.array) -> Tuple:
    """Get the height, width, and channel of the input image.

    Args:
        image (np.array): input image

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


def extract_local_data(
    image: np.array,
    density_map: np.array,
    params_dict: dict,
    is_flip: bool,
    index_list: List = None,
) -> Tuple:
    """Extract local image and density map from raw data

    Args:
        image (np.array): raw image
        density_map (np.array): raw density map
        params_dict (dict): dictionary of parameters
        is_flip (bool): whether image is flip or not
        index_list (List): target index list of index_h and index_w

    Returns:
        Tuple: numpy array of local image and density map
    """
    # triming original image
    image = image[
        params_dict["analysis_image_height_min"] : params_dict[
            "analysis_image_height_max"
        ],
        params_dict["analysis_image_width_min"] : params_dict[
            "analysis_image_width_max"
        ],
    ]
    height, width, channel = get_image_shape(image)

    pad = math.floor(params_dict["local_image_size"] / 2)
    pad_image = np.zeros((height + pad * 2, width + pad * 2, channel), dtype="float32")
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
    if index_list is None:
        index_list = [i for i in range(local_data_number)]

    local_image_list = []
    local_density_list = []
    for idx in index_list:
        # raw image index convert to padding image index
        h = index_h[idx]
        w = index_w[idx]
        local_image_list.append(pad_image[h : h + 2 * pad, w : w + 2 * pad])
        if density_map is not None:
            local_density_list.append(density_map[h, w])

    return local_image_list, local_density_list


def load_model(model_path: str, device_id: str, memory_rate: float) -> Tuple:
    """Load trained Convolutional Neural Network model that defined by TensorFlow

    Args:
        model_path (str): path of trained model
        device_id (str): GPU divice ID
        memory_rate (float): use rate of GPU memory (0.0-1.0)

    Returns:
        Tuple: loaded model and tensorflow session
    """

    config = tf.compat.v1.ConfigProto(
        gpu_options=tf.compat.v1.GPUOptions(
            visible_device_list=device_id, per_process_gpu_memory_fraction=memory_rate
        )
    )
    sess = InteractiveSession(config=config)

    model = DensityModel()
    saver = tf.compat.v1.train.Saver()
    ckpt = tf.train.get_checkpoint_state(model_path)
    if ckpt:
        last_model = ckpt.model_checkpoint_path
        logger.info("Load model: {}".format(last_model))
        saver.restore(sess, last_model)
    else:
        logger.error("Eroor: Not exist model!")
        logger.error(f"Please check model_path (model_path={model_path})")
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

    logger.info("****************************************")
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


def get_file_name_from_path(path: str) -> int:
    """Get file name from file path.
    ex) path="./tmp/20170416_20111.png" -> file_name=20170416_20111

    Args:
        path (str): file path

    Returns:
        int: extracted file name
    """
    file_name = path.split("/")[-1]
    return file_name.split(".")[0]


def save_dataset_path(X_path_list: List, y_path_list: List, save_path: str) -> None:
    """Save the file names contained in the data set in CSV format.

    Args:
        X_path_list (List): x (input) path list
        y_path_list (List): y (label) path list
        save_path (str): path of save destination
    """
    pd.DataFrame({"X_path": X_path_list, "y_path": y_path_list}).to_csv(
        save_path, index=False
    )
