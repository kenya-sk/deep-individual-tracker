import math
from glob import glob
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from detector.constants import (
    ANALYSIS_HEIGHT_MAX,
    ANALYSIS_HEIGHT_MIN,
    ANALYSIS_WIDTH_MAX,
    ANALYSIS_WIDTH_MIN,
    FRAME_CHANNEL,
    IMAGE_EXTENTION,
    LOCAL_IMAGE_SIZE,
    RANDOM_SEED,
)
from detector.exceptions import DatasetEmptyError, LoadImageError
from detector.index_manager import IndexManager
from detector.logger import logger
from detector.utils import get_image_shape

DatasetType = Tuple[
    List[Path], List[Path], List[Path], List[Path], List[Path], List[Path]
]


class Dataset:
    """Create dataset for model training.
    This class can choose between two types of data sets via methods.
    First, random split dataset that randomly splits the dataset into train, validation, and test.
    Second, date base split dataset that splits the dataset into train, validation, and test
    based on given list of dates.
    """

    def __init__(
        self,
        image_directory: Path,
        density_directory: Path,
        test_size: float = 0.2,
        train_date_list: Optional[List[str]] = None,
        valid_date_list: Optional[List[str]] = None,
        test_date_list: Optional[List[str]] = None,
        save_path_directory: Optional[Path] = None,
    ) -> None:
        self.image_directory = image_directory
        self.density_directory = density_directory
        self.test_size = test_size
        self.train_date_list = train_date_list
        self.valid_date_list = valid_date_list
        self.test_date_list = test_date_list
        self.save_path_directory = save_path_directory
        self.X_list: List[Path]
        self.y_list: List[Path]
        self.X_train: List[Path]
        self.X_valid: List[Path]
        self.X_test: List[Path]
        self.y_train: List[Path]
        self.y_valid: List[Path]
        self.y_test: List[Path]

    def create_random_dataset(self) -> None:
        """Create random splitted dataset."""
        self.X_list, self.y_list = load_dataset(
            self.image_directory, self.density_directory, f"*{IMAGE_EXTENTION}"
        )
        (
            self.X_train,
            self.X_valid,
            self.X_test,
            self.y_train,
            self.y_valid,
            self.y_test,
        ) = split_dataset(
            self.X_list, self.y_list, self.test_size, self.save_path_directory
        )

    def create_date_dataset(self) -> None:
        """Create date base splitted dataset. This dataset avoided data leakage."""
        if (
            self.train_date_list is None
            or self.valid_date_list is None
            or self.test_date_list is None
        ):
            message = "train_date_list, valid_date_list, test_date_list is None."
            logger.error(message)
            raise ValueError(message)

        (
            self.X_train,
            self.X_valid,
            self.X_test,
            self.y_train,
            self.y_valid,
            self.y_test,
        ) = split_dataset_by_date(
            self.image_directory,
            self.density_directory,
            self.train_date_list,
            self.valid_date_list,
            self.test_date_list,
            self.save_path_directory,
        )


def load_dataset(
    image_directory: Path, density_directory: Path, file_pattern: str
) -> Tuple[List[Path], List[Path]]:
    """Load the file name of the data set.

    Args:
        image_directory (Path): directory name of image (input)
        density_directory (Path): directory name of density map (label)
        file_pattern (str): file name pattern of image (input)

    Returns:
        Tuple[List[Path], List[Path]]: tuple with image and label filename pairs
    """
    X_list, y_list = [], []
    file_list = glob(f"{image_directory}/{file_pattern}")
    if len(file_list) == 0:
        message = f'dateset="{image_directory}/{file_pattern}"" is empty.'
        logger.error(message)
        raise DatasetEmptyError(message)

    for path in file_list:
        # get label path from input image path
        density_file_name = path.replace(IMAGE_EXTENTION, ".npy").split("/")[-1]
        density_path = density_directory / density_file_name

        # store input and label path
        X_list.append(Path(path))
        y_list.append(Path(density_path))

    return X_list, y_list


def load_multi_date_datasets(
    image_directory: Path, density_directory: Path, date_list: List[str]
) -> Tuple[List[Path], List[Path]]:
    """Load input and output pairs based on a list of dates

    Args:
        image_directory (Path): directory name of image (input)
        density_directory (Path): directory name of density map (label)
        date_list (List[str]): list of dates to be used for splitting

    Returns:
        Tuple[List[Path], List[Path]]: tuple with image and label filename pairs
    """
    X_multi_list, y_multi_list = [], []
    for date in date_list:
        file_pattern = f"{date}_*{IMAGE_EXTENTION}"
        X_list, y_list = load_dataset(image_directory, density_directory, file_pattern)
        X_multi_list.extend(X_list)
        y_multi_list.extend(y_list)

    return X_multi_list, y_multi_list


def save_dataset_path(
    X_path_list: List[Path], y_path_list: List[Path], save_path: Path
) -> None:
    """Save the file names contained in the data set in CSV format.

    Args:
        X_path_list (List[Path]): x (input) path list
        y_path_list (List[Path]): y (label) path list
        save_path (Path): path of save destination
    """
    pd.DataFrame({"X_path": X_path_list, "y_path": y_path_list}).to_csv(
        save_path, index=False
    )


def split_dataset(
    X_list: List[Path],
    y_list: List[Path],
    test_size: float,
    save_path_directory: Optional[Path] = None,
) -> DatasetType:
    """Randomly split the dataset into train, validation, and test.

    Args:
        X_list (List): input image path list
        y_list (List): label path list
        test_size (float): test data size (0.0 - 1.0)
        save_path_directory (Path, optional): directory name to save the file name of each dataset. Defaults to None.

    Returns:
        DatasetType:
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
        Path(save_path_directory).mkdir(parents=True, exist_ok=True)
        save_dataset_path(X_train, y_train, save_path_directory / "train_dataset.csv")
        save_dataset_path(X_valid, y_valid, save_path_directory / "valid_dataset.csv")
        save_dataset_path(X_test, y_test, save_path_directory / "test_dataset.csv")

    return X_train, X_valid, X_test, y_train, y_valid, y_test


def split_dataset_by_date(
    image_directory: Path,
    density_directory: Path,
    train_date_list: List[str],
    valid_date_list: List[str],
    test_date_list: List[str],
    save_path_directory: Optional[Path] = None,
) -> DatasetType:
    """split the dataset by date into train, validation, and test.

    Args:
        image_directory (Path): directory name of image (input)
        density_directory (Path): directory name of density map (label)
        train_date_list (List[str]): date list of training data
        valid_date_list (List[str]): date list of validation data
        test_date_list (List[str]): date list of test data
        save_path_directory (Path, optional): directory name to save the file name of each dataset. Defaults to None.

    Returns:
        DatasetType:
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
        Path(save_path_directory).mkdir(parents=True, exist_ok=True)
        save_dataset_path(X_train, y_train, save_path_directory / "train_dataset.csv")
        save_dataset_path(X_valid, y_valid, save_path_directory / "valid_dataset.csv")
        save_dataset_path(X_test, y_test, save_path_directory / "test_dataset.csv")

    return X_train, X_valid, X_test, y_train, y_valid, y_test


def load_image(path: Path, is_rgb: bool = True, normalized: bool = False) -> np.ndarray:
    """Loads image data frosm the input path and returns image in numpy array format.

    Args:
        path (Path): input image file path
        is_rgb (bool, optional): whether convert RGB format. Defaults to True.
        normalized (bool, optional): whether normalize loaded image. Defaults to False.

    Returns:
        np.ndarray: loaded image
    """
    # opencv cannot read Pathlib.Path format
    image = cv2.imread(str(path))
    if image is None:
        message = f'image path="{path}" cannot be loaded.'
        logger.error(message)
        raise LoadImageError(message)
    logger.info(f"Loaded Image: {path}")

    # convert image BGR to RGB
    if is_rgb:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # normalization
    if normalized:
        image = image / 255.0

    return image


def load_sample(
    X_path: Path,
    y_path: Path,
    input_image_shape: Tuple,
    mask_image: np.ndarray,
    is_rgb: bool,
    normalized: bool,
) -> Tuple:
    """Load samples for input into the model.
    Each sample is a set of image and label, with mask image applied as needed.

    Args:
        X_path (Path): x (input) path
        y_path (Path): y (label) path
        input_image_shape (Tuple): raw input image shape
        mask_image (np.ndarray): mask image array
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


def load_mask_image(
    mask_path: Optional[Path] = None, normalized: bool = True
) -> np.ndarray:
    """Load a binary mask image and normalizes the values as necessary.

    Args:
        mask_path (Path, optional): binary mask image path. Defaults to None.
        normalized (bool, optional): whether execute normalization. Defaults to True.

    Returns:
        np.ndarray: loaded binary masked image
    """
    if (mask_path is not None) and (mask_path.is_file()):
        # load binary mask image
        # opencv cannot read Pathlib.Path format
        mask = cv2.imread(str(mask_path))
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
    image: np.ndarray, mask: np.ndarray, channel: int = 3
) -> np.ndarray:
    """Apply mask processing to image data.

    Args:
        image (np.ndarray): image to be applied
        mask (np.ndarray): mask image
        channel (int, optional): channel number of applied image. Defaults to 3.

    Returns:
        np.ndarray: masked image
    """
    if mask is None:
        return image

    # apply mask to image
    if channel == 3:
        masked_image = image * mask
    else:
        masked_image = image * mask[:, :, 0]

    return masked_image


def extract_local_data(
    image: np.ndarray,
    density_map: Optional[np.ndarray],
    index_manager: IndexManager,
    is_flip: bool,
    index_list: Optional[List[int]] = None,
) -> Tuple:
    """Extract local image and density map from raw data

    Args:
        image (np.ndarray): raw image
        density_map (np.ndarray, optional): raw density map
        index_manager (IndexManager): index manager class of masked image
        is_flip (bool): whether image is flip or not
        index_list (List[int], optional): target index list of index_h and index_w

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
    pad_image = np.zeros((height + pad * 2, width + pad * 2, channel), dtype="uint8")
    pad_image[pad : pad + height, pad : pad + width] = image

    # get each axis index
    if is_flip:
        index_h = index_manager.flip_index_h
        index_w = index_manager.flip_index_w
    else:
        index_h = index_manager.index_h
        index_w = index_manager.index_w

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
