import logging
import os
from glob import glob
from typing import List

import cv2
import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from tqdm import tqdm

# logging setting
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s"
)
logger = logging.getLogger(__name__)


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


def create_detected_point_movie(
    image_directory: str,
    point_coord_directory: str,
    movie_save_path: str,
    format_dict: dict,
) -> None:
    """Image data and coordinate data of detection points are load in pairs.
    The detection points are then plotted on the image,
    and the multiple data are connected in time series order to create a video.

    Args:
        image_directory (str): directory that the raw image are stored
        point_coord_directory (str): directory that the detected point coordinate are stored
        movie_save_path (str): path to save output movie
        format_dict (dict): dictionary of output movie format
    """

    output_movie = cv2.VideoWriter(
        movie_save_path,
        cv2.VideoWriter_fourcc("m", "p", "4", "v"),
        format_dict["fps"],
        (int(format_dict["width"]), int(format_dict["height"])),
    )

    image_path_list = glob(f"{image_directory}/*.png")
    sorted_image_path_list = sort_by_frame_number(image_path_list)
    for path in tqdm(sorted_image_path_list, desc="Create Movie Data"):
        image = cv2.imread(path)
        # The file names of the image and the coordinate data must be the same
        cood_file_name = path.split("/")[-1].replace(".png", ".csv")
        detected_coord = np.loadtxt(
            f"{point_coord_directory}/{cood_file_name}", delimiter=","
        )

        # plot detected point on image
        for cood in detected_coord:
            cv2.circle(
                image, (int(cood[0]), int(cood[1])), 3, (0, 0, 255), -1, cv2.LINE_AA
            )

        # writes as movie data
        output_movie.write(image)

    cv2.destroyAllWindows()
    logger.info("Saved Movie Data in '{movie_save_path}'")


@hydra.main(config_path="../conf", config_name="detected_point_movie")
def main(cfg: DictConfig) -> None:
    """Create movie data based on the raw image data and
    the series of detected points data

    Args:
        cfg (DictConfig): config that loaded by @hydra.main()
    """

    logger.info(f"Loaded config: {cfg}")

    # get path from config file
    image_directory = cfg.path.image_directory
    point_coord_directory = cfg.path.point_coord_directory
    movie_save_path = cfg.path.movie_save_path

    # get movie format from config file
    format_dict = cfg.movie_format

    create_detected_point_movie(
        image_directory, point_coord_directory, movie_save_path, format_dict
    )


if __name__ == "__main__":
    main()
