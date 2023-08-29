import os
from glob import glob
from pathlib import Path
from typing import List

import cv2
import numpy as np

from annotator.exceptions import LoadImageError, LoadVideoError, PathNotExistError
from annotator.logger import logger


def get_path_list(working_directory: Path, path: Path) -> List[Path]:
    """Input a file or directory path and creates a list of full paths.

    Args:
        working_directory (Path): current working directory
        path (Path): input path that file or directory

    Raises:
        PathNotExistError: not exist target file or directory

    Returns:
        List[Path]: list of full path
    """
    full_path = working_directory / path
    if os.path.isfile(full_path):
        path_list = [full_path]
    elif os.path.isdir(full_path):
        path_list = [
            Path(current_path)
            for current_path in glob(f"{full_path}/*")
            if os.path.isfile(current_path)
        ]
    else:
        message = f'path="{full_path}" is not exist.'
        logger.error(message)
        raise PathNotExistError(message)

    return path_list


def get_full_path_list(
    current_working_dirc: Path, relative_path_list: List[Path]
) -> List[Path]:
    """Join the current working directory name and relative path to get a list of full paths.

    Args:
        current_working_dirc (Path): current working directory name
        relative_path_list (List[Path]): list of relative path

    Returns:
        List[Path]: List of converted full path
    """
    full_path_list = [current_working_dirc / path for path in relative_path_list]
    return full_path_list


def get_input_data_type(path: Path) -> str:
    """Get the extension from the input data path and get the data processing format.
    The target format is images or videos.

    Args:
        path (Path): input file path

    Returns:
        str: processing data format
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


def load_image(path: Path) -> np.ndarray:
    """Loads image data from the input path and returns image in numpy array format.

    Args:
        path (Path): input image file path

    Raises:
        LoadImageError: loading error of image data

    Returns:
        np.ndarray: loaded image
    """
    # opencv cannot read Pathlib.Path format
    image = cv2.imread(str(path))
    if image is None:
        message = f'path="{path}" cannot read image file.'
        logger.error(message)
        raise LoadImageError(message)
    logger.info(f"Loaded Image: {path}")

    return image


def load_video(path: Path) -> cv2.VideoCapture:
    """Loads video data from the input path and returns video in cv2.VideoCapture format.

    Args:
        path (Path): input video file path

    Raises:
        LoadVideoError: loading error of video data

    Returns:
        cv2.VideoCapture: loaded video
    """
    video = cv2.VideoCapture(path)
    if not (video.isOpened()):
        message = f'path="{path}" cannot read video file.'
        logger.error(message)
        raise LoadVideoError(message)
    logger.info(f"Loaded Video: {path}")

    return video


def save_image(path: Path, image: np.ndarray) -> None:
    """Save the image data in numpy format in the target path.

    Args:
        path (Path): save path of image
        image (np.ndarray): target image
    """
    cv2.imwrite(path, image)
    logger.info(f"Saved Image: {path}")


def save_coordinate(path: Path, coordinate: np.ndarray) -> None:
    """Save the coordinate data (x, y) in numpy format in the target path.

    Args:
        path (Path): save path of coordinate
        coordinate (np.ndarray): coordinate of annotated points
    """
    np.savetxt(path, coordinate, delimiter=",", fmt="%d")
    logger.info(f"Saved Coordinate: {path}")


def save_density_map(path: Path, density_map: np.ndarray) -> None:
    """Save the density map data in numpy format in the target path.

    Args:
        path (Path): save path of density map
        density_map (np.ndarray): annotated density map
    """
    np.save(path, density_map)
    logger.info(f"Save Density Map: {path}")
