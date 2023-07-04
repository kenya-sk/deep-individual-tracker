import os
from glob import glob
from typing import List

import cv2
import hydra
import numpy as np
import pandas as pd
from detector.constants import (
    CONFIG_DIR,
    DATA_DIR,
    DETECTED_MOVIE_CONFIG_NAME,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    IMAGE_EXTENTION,
    LINE_TYPE,
    MOVIE_FPS,
    POINT_COLOR,
    POINT_RADIUS,
    POINT_THICKNESS,
)
from detector.logger import logger
from omegaconf import DictConfig
from tqdm import tqdm


def sort_by_frame_number(path_list: List) -> List:
    """Sort file path list by frame number.
    Frame numbers are extracted from file name.

    Args:
        path_list (List): file path list before sorting

    Returns:
        List: sorted list
    """

    def _extract_frame_number(path: str) -> int:
        """
        Extracted frame number from file path.
        procedures:
            1. extract file name
            2. remove extension
            3. split by "_" and get the last value
               in the list as frame number
        ex) path="./example/2022_03_19_234224.png"
            -> frame_number = 234224

        :param path: file path
        :return: frame number
        """
        base_name = os.path.basename(path)
        file_name = os.path.splitext(base_name)[0]
        frame_number = int(file_name.split("_")[-1])
        return frame_number

    df = pd.DataFrame({"raw_path": path_list})
    df.loc[:, "frame_number"] = df["raw_path"].map(_extract_frame_number)
    df = df.sort_values(by="frame_number")

    return df["raw_path"].to_list()


def draw_detection_points(image: np.ndarray, point_coord: np.ndarray) -> np.ndarray:
    """Draw detection points on the image.

    Returns:
        np.ndarray: image with detection points
    """
    if point_coord.shape == (2,):
        point_coord = np.array([point_coord])

    for point in point_coord:
        cv2.circle(
            image,
            (point[0], point[1]),
            radius=POINT_RADIUS,
            color=POINT_COLOR,
            thickness=POINT_THICKNESS,
            lineType=LINE_TYPE,
        )

    return image


def create_detected_point_movie(
    image_directory: str,
    point_coord_directory: str,
    movie_save_path: str,
) -> None:
    """Image data and coordinate data of detection points are load in pairs.
    The detection points are then plotted on the image,
    and the multiple data are connected in time series order to create a video.

    Args:
        image_directory (str): directory that the raw image are stored
        point_coord_directory (str): directory that the detected point coordinate are stored
        movie_save_path (str): path to save output movie
    """

    output_movie = cv2.VideoWriter(
        movie_save_path,
        cv2.VideoWriter_fourcc("m", "p", "4", "v"),
        MOVIE_FPS,
        (FRAME_WIDTH, FRAME_HEIGHT),
    )

    image_path_list = glob(f"{image_directory}/*{IMAGE_EXTENTION}")
    sorted_image_path_list = sort_by_frame_number(image_path_list)
    for path in tqdm(sorted_image_path_list, desc="Create Movie Data"):
        image = cv2.imread(path)
        # The file names of the image and the coordinate data must be the same
        cood_file_name = path.split("/")[-1].replace(IMAGE_EXTENTION, ".csv")
        detected_coord = np.loadtxt(
            f"{point_coord_directory}/{cood_file_name}", delimiter=","
        )

        # plot detected point on image
        drew_image = draw_detection_points(image, detected_coord)

        # writes as movie data
        output_movie.write(drew_image)

    cv2.destroyAllWindows()
    logger.info(f"Saved Movie Data in '{movie_save_path}'")


@hydra.main(
    config_path=str(CONFIG_DIR),
    config_name=DETECTED_MOVIE_CONFIG_NAME,
    version_base="1.1",
)
def main(cfg: DictConfig) -> None:
    """Create movie data based on the raw image data and
    the series of detected points data

    Args:
        cfg (DictConfig): config that loaded by @hydra.main()
    """

    logger.info(f"Loaded config: {cfg}")

    create_detected_point_movie(
        str(DATA_DIR / cfg["image_directory"]),
        str(DATA_DIR / cfg["point_coord_directory"]),
        str(DATA_DIR / cfg["movie_save_path"]),
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(e)
