import logging
import math
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import List, NoReturn, Tuple

import cv2
import numpy as np
import tensorflow as tf

from model import DensityModel

logger = logging.getLogger(__name__)


def display_data_info(
    input_path: str,
    output_dirctory: str,
    skip_width: int,
    pred_batch_size: int,
    band_width: int,
    cluster_thresh: float,
    is_saved_map: bool,
) -> NoReturn:
    """Display setting information

    Args:
        input_path (str): input data path
        output_dirctory (str): output directory
        skip_width (int): skip width in horizontal direction
        pred_batch_size (int): batch size
        band_width (int): band width of Mean-Shift Clustering
        cluster_thresh (float): threshold to be subjected to clustering
        is_saved_map (bool):  whether to save the density map

    Returns:
        NoReturn:
    """
    logger.info("*************************************************")
    logger.info("Input path      : {0}".format(input_path))
    logger.info("Output dirctory : {0}".format(output_dirctory))
    logger.info("Skip width      : {0}".format(skip_width))
    logger.info("Pred batch size : {0}".format(pred_batch_size))
    logger.info("Band width      : {0}".format(band_width))
    logger.info("Cluster thresh  : {0}".format(cluster_thresh))
    logger.info("Save density map: {0}".format(is_saved_map))
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
    true_positive_lst, false_positive_lst, false_negative_lst, sample_num_lst, skip=0
):
    accuracy_lst = []
    precision_lst = []
    recall_lst = []
    f_measure_lst = []
    for i in range(len(true_positive_lst)):
        accuracy, precision, recall, f_measure = eval_metrics(
            true_positive_lst[i],
            false_positive_lst[i],
            false_negative_lst[i],
            sample_num_lst[i],
        )
        accuracy_lst.append(accuracy)
        precision_lst.append(precision)
        recall_lst.append(recall)
        f_measure_lst.append(f_measure)

    logger.info("\n**************************************************************")

    logger.info("                          GROUND TRUTH          ")
    logger.info("                    |     P     |     N     |           ")
    logger.info("          -----------------------------------------")
    logger.info(
        "                P   |    {0}    |     {1}     |           ".format(
            sum(true_positive_lst), sum(false_positive_lst)
        )
    )
    logger.info("PRED      -----------------------------------------")
    logger.info(
        "                N   |    {0}    |     /     |           ".format(
            sum(false_negative_lst)
        )
    )
    logger.info("          -----------------------------------------")

    logger.info(
        "\nToal Accuracy (data size {0}, sikp size {1}) : {2}".format(
            len(accuracy_lst), skip, sum(accuracy_lst) / len(accuracy_lst)
        )
    )
    logger.info(
        "Toal Precision (data size {0}, sikp size {1})  : {2}".format(
            len(precision_lst), skip, sum(precision_lst) / len(precision_lst)
        )
    )
    logger.info(
        "Toal Recall (data size {0}, sikp size {1})     : {2}".format(
            len(recall_lst), skip, sum(recall_lst) / len(recall_lst)
        )
    )
    logger.info(
        "Toal F measure (data size {0}, sikp size {1})  : {2}".format(
            len(f_measure_lst), skip, sum(f_measure_lst) / len(f_measure_lst)
        )
    )
    logger.info("****************************************************************")


def apply_masking_on_image(image: np.array, mask_path: str = None) -> np.array:
    """Apply mask processing to image data.

    Args:
        image (np.array): image to be applied
        mask_path (str, optional): binay mask path. Defaults to None.

    Returns:
        np.array: masked image
    """
    height = image.shape[0]
    width = image.shape[1]
    channel = image.shape[2]

    # mask: 3channel mask image. the value is 0 or 1
    mask = cv2.imread(mask_path)
    if mask is None:
        mask = np.ones((height, width, channel))
    else:
        mask = cv2.imread(mask_path)

    if channel == 3:
        masked_image = image * mask
    else:
        masked_image = image * mask[:, :, 0]

    return masked_image


def get_masked_index(mask_path: str = None, horizontal_flip: bool = False) -> Tuple:
    """Masking an image to get valid index

    Args:
        mask_path (str, optional): binay mask path. Defaults to None.
        horizontal_flip (bool, optional): Whether to perform data augumentation. Defaults to False.

    Returns:
        Tuple: valid index list (heiht and width)
    """

    if mask_path is None:
        mask = np.ones((720, 1280))
    else:
        mask = cv2.imread(mask_path)

    if mask.shape[2] == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # index of data augumentation
    if horizontal_flip:
        mask = mask[:, ::-1]

    index = np.where(mask > 0)
    index_h = index[0]
    index_w = index[1]
    assert len(index_h) == len(index_w)

    return index_h, index_w


def get_local_data(
    image: np.array, dens_map: np.array, params_dict: dict, is_flip: bool
) -> Tuple:
    """Get local image and density map from raw data

    Args:
        image (np.array): raw image
        dens_map (np.array): raw density map
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

    pad = math.floor(params_dict["local_image_size"] / 2)
    pad_image = np.zeros(
        (height + pad * 2, width + pad * 2, image.shape[2]), dtype="uint8"
    )
    pad_image[pad : height + pad, pad : width + pad] = image

    # get each axis index
    if is_flip:
        index_h = params_dict["flip_index_h"]
        index_w = params_dict["flip_index_w"]
    else:
        index_h = params_dict["index_h"]
        index_w = params_dict["index_w"]

    # extract local image
    local_img_array = np.zeros(
        (
            len(index_w),
            params_dict["local_image_size"],
            params_dict["local_image_size"],
            image.shape[2],
        ),
        dtype="uint8",
    )
    density_array = np.zeros((len(index_w)), dtype="float32")
    for idx in range(len(index_w)):
        # fix index(pad_image)
        h = index_h[idx]
        w = index_w[idx]
        local_img_array[idx] = pad_image[h : h + 2 * pad, w : w + 2 * pad]
        density_array[idx] = dens_map[h, w]

    return local_img_array, density_array


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
    sess = tf.InteractiveSession(config=config)

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

    logger.info("****************************************")
    logger.info("Video path             : {0}".format(video_path))
    logger.info("Fourcc                 : {0}".format(fourcc))
    logger.info("FPS                    : {0}".format(fps))
    logger.info("Size = (height, width) : ({0}, {1})".format(height, width))
    logger.info("Total frame            : {0}".format(total_frame))
    logger.info("****************************************")

    return cap, fourcc, fps, height, width, total_frame


def get_current_time_str(
    time_difference: int = 9, time_format: str = "%Y%m%d%H%M%S"
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
    """_summary_

    Args:
        start_time (float): _description_

    Returns:
        str: _description_
    """
    total_elpased_second = time.time() - start_time
    elapsed_hour = int(total_elpased_second / 3600)
    elapsed_minute = int(total_elpased_second % 3600 / 60)
    elapsed_second = int(total_elpased_second % 60)

    return f"{elapsed_hour}[hour] {elapsed_minute}[min] {elapsed_second}[sec]"


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


def set_tensorboard(
    tensorboard_directory: str,
    current_time_str: str,
    tf_session: tf.InteractiveSession,
) -> Tuple:
    """_summary_

    Args:
        tensorboard_directory (str): _description_
        tf_session (tf.InteractiveSession): _description_

    Returns:
        Tuple: _description_
    """
    # directory of TensorBoard
    log_directory = "{tensorboard_directory}/{current_time_str}"
    logger.info(f"TensorBoard Directory: {log_directory}")

    # if the target directory exists, delete and recreate
    if tf.io.gfile.exists(log_directory):
        tf.compat.v1.gfile.DeleteRecursively(log_directory)
    tf.io.gfile.makedirs(log_directory)

    # set variable on TensorBoard
    summuray_merged = tf.compat.v1.summary.merge_all()
    train_writer = tf.summary.FileWriter(f"{0}/train", tf_session.graph)
    valid_writer = tf.summary.FileWriter(f"{0}/validation")
    test_writer = tf.summary.FileWriter(f"{0}/test")

    return summuray_merged, train_writer, valid_writer, test_writer
