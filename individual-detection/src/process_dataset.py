import logging
import math
import sys
from glob import glob
from typing import List, Tuple

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from constatns import (ANALYSIS_HEIGHT_MAX, ANALYSIS_HEIGHT_MIN,
                       ANALYSIS_WIDTH_MAX, ANALYSIS_WIDTH_MIN, FRAME_CHANNEL,
                       FRAME_HEIGHT, FRAME_WIDTH, IMAGE_EXTENTION,
                       LOCAL_IMAGE_SIZE, RANDOM_SEED)

logger = logging.getLogger(__name__)


def load_dataset(
    image_directory: str, density_directory: str, file_pattern: str = "*.png"
) -> Tuple[List[str], List[str]]:
    """Load the file name of the data set.

    Args:
        image_directory (str): directory name of image (input)
        density_directory (str): directory name of density map (label)

    Returns:
        Tuple[List[str], List[str]]: tuple with image and label filename pairs
    """
    X_list, y_list = [], []
    file_list = glob(f"{image_directory}/{file_pattern}")
    if len(file_list) == 0:
        sys.stderr.write("Error: Not found input image file")
        sys.exit(1)

    for path in file_list:
        # get label path from input image path
        density_file_name = path.replace(IMAGE_EXTENTION, ".npy").split("/")[-1]
        density_path = f"{density_directory}/{density_file_name}"

        # store input and label path
        X_list.append(path)
        y_list.append(density_path)

    return X_list, y_list


def load_multi_date_datasets(
    image_directory: str, density_directory: str, date_list: List[str]
) -> Tuple[List[str], List[str]]:
    """Load input and output pairs based on a list of dates

    Args:
        image_directory (str): directory name of image (input)
        density_directory (str): directory name of density map (label)
        date_list (List[str]): list of dates to be used for splitting

    Returns:
        Tuple[List[str], List[str]]: tuple with image and label filename pairs
    """
    X_multi_list, y_multi_list = [], []
    for date in date_list:
        file_pattern = f"{date}_*{IMAGE_EXTENTION}"
        X_list, y_list = load_dataset(image_directory, density_directory, file_pattern)
        X_multi_list.extend(X_list)
        y_multi_list.extend(y_list)

    return X_multi_list, y_multi_list


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


def split_dataset(
    X_list: List,
    y_list: List,
    test_size: float,
    save_path_directory: str = None,
) -> Tuple[List[str], List[str], List[str], List[str], List[str], List[str]]:
    """Randomly split the dataset into train, validation, and test.

    Args:
        X_list (List): input image path list
        y_list (List): label path list
        test_size (float): test data size (0.0 - 1.0)
        save_path_directory (str, optional): directory name to save the file name of each dataset. Defaults to None.

    Returns:
        Tuple[List[str], List[str], List[str], List[str], List[str], List[str]]:
            tuple containing the filenames of train, validation, and test
    """
    # splite dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X_list, y_list, test_size=test_size, random_state=RANDOM_SEED
    )
    # split dataset into validation and test
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_test, y_test, test_size=0.5, random_state=RANDOM_SEED
    )

    if save_path_directory is not None:
        save_dataset_path(X_train, y_train, f"{save_path_directory}/train_dataset.csv")
        save_dataset_path(X_valid, y_valid, f"{save_path_directory}/valid_dataset.csv")
        save_dataset_path(X_test, y_test, f"{save_path_directory}/test_dataset.csv")

    return X_train, X_valid, X_test, y_train, y_valid, y_test


def split_dataset_by_date(
    image_directory: str,
    density_directory: str,
    train_date_list: List[str],
    valid_date_list: List[str],
    test_date_list: List[str],
    save_path_directory: str = None,
) -> Tuple[List[str], List[str], List[str], List[str], List[str], List[str]]:
    """split the dataset by date into train, validation, and test.

    Args:
        image_directory (str): directory name of image (input)
        density_directory (str): directory name of density map (label)
        train_date_list (List[str]): date list of training data
        valid_date_list (List[str]): date list of validation data
        test_date_list (List[str]): date list of test data
        save_path_directory (str, optional): directory name to save the file name of each dataset. Defaults to None.

    Returns:
        Tuple[List[str], List[str], List[str], List[str], List[str], List[str]]:
            tuple containing the filenames of train, validation, and test
    """
    X_train, y_train = load_multi_date_datasets(
        image_directory, density_directory, train_date_list
    )
    X_valid, y_valid = load_multi_date_datasets(
        image_directory, density_directory, valid_date_list
    )
    X_test, y_test = load_multi_date_datasets(
        image_directory, density_directory, test_date_list
    )

    if save_path_directory is not None:
        save_dataset_path(X_train, y_train, f"{save_path_directory}/train_dataset.csv")
        save_dataset_path(X_valid, y_valid, f"{save_path_directory}/valid_dataset.csv")
        save_dataset_path(X_test, y_test, f"{save_path_directory}/test_dataset.csv")

    return X_train, X_valid, X_test, y_train, y_valid, y_test


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
        X_image = apply_masking_on_image(X_image, mask_image, channel=FRAME_CHANNEL)
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


def get_masked_index(mask: np.array, horizontal_flip: bool = False) -> Tuple:
    """Masking an image to get valid index

    Args:
        mask (np.array): binay mask image
        horizontal_flip (bool, optional): Whether to perform data augumentation. Defaults to False.

    Returns:
        Tuple: valid index list (heiht and width)
    """
    if mask is None:
        mask = np.ones((FRAME_HEIGHT, FRAME_WIDTH))

    # convert gray scale image
    if (len(mask.shape) == 3) and (mask.shape[2] > 1):
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # crop the image to the analysis area
    mask = mask[
        ANALYSIS_HEIGHT_MIN:ANALYSIS_HEIGHT_MAX,
        ANALYSIS_WIDTH_MIN:ANALYSIS_WIDTH_MAX,
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
        ANALYSIS_HEIGHT_MIN:ANALYSIS_HEIGHT_MAX,
        ANALYSIS_WIDTH_MIN:ANALYSIS_WIDTH_MAX,
    ]
    height, width, channel = get_image_shape(image)

    pad = math.floor(LOCAL_IMAGE_SIZE / 2)
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
