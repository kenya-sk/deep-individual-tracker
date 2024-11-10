import os
from pathlib import Path
from typing import Dict, List

import cv2
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes

from monitoring.config import MonitoringConfig
from monitoring.constants import DATA_DIR, FRAME_HEIGHT, FRAME_WIDTH
from monitoring.exceptions import PathNotExistError
from monitoring.logger import logger


def load_one_hour_density(coord_dirc: Path, end_frame_num: int) -> pd.DataFrame:
    """Load the density over the past 1 hour

    Args:
        coord_dirc (Path): directory name of coordinate data
        end_frame_num (int): frame number of end data

    Returns:
        pd.DataFrame: DataFrame containing past 1 hour coordinates
    """
    # get frame number 1 hour ago
    start_frame_num = max(0, end_frame_num - 3600)
    # get the distribution of individuals over the past 1 hour
    past_density_dctlst: Dict[str, List[int]] = {"x": [], "y": []}
    for frame_num in range(start_frame_num, end_frame_num):
        path = f"{coord_dirc}/{frame_num}.csv"
        if os.path.isfile(path):
            cord_df = pd.read_csv(path, header=None)
            past_density_dctlst["x"].extend(cord_df[0].values)
            past_density_dctlst["y"].extend(cord_df[1].values)
        else:
            logger.warning(f"Not Exist Path: {path}")

    return pd.DataFrame(past_density_dctlst)


def set_frame(
    cfg: MonitoringConfig,
    frame_num: int,
    detected_coordinate_df: pd.DataFrame,
    one_hour_density_df: pd.DataFrame,
    ax: Axes,
) -> None:
    """Set the current frame on frame axis

    Args:
        cfg (MonitoringConfig): config for monitoring environment
        frame_num (int): current frame number
        detected_coordinate_df (pd.DataFrame): DataFrame containing current frame coordinates
        one_hour_density_df (pd.DataFrame): DataFrame containing past 1 hour coordinates
        ax (Axes): matplotlib figure axis of frame
    """
    frame_path = DATA_DIR / cfg.path.frame_directory / f"{frame_num}.png"
    if os.path.isfile(frame_path):
        # opencv cannot read Pathlib.Path format
        frame = cv2.imread(str(frame_path))
    else:
        message = f'frame_path="{frame_path}" is not exist.'
        logger.error(message)
        raise PathNotExistError(message)

    # set current frame
    ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # plot dots on the detected tuna
    if len(detected_coordinate_df) != 0:
        assert detected_coordinate_df["x"] == detected_coordinate_df["y"]
        ax.plot(
            detected_coordinate_df["x"],
            detected_coordinate_df["y"],
            linestyle="",
            marker="o",
            ms=4.0,
            color="r",
        )

    # display density map of past 1 hour
    if len(one_hour_density_df) != 0:
        sns.kdeplot(
            data=one_hour_density_df,
            x="x",
            y="y",
            cmap="RdYlGn_r",
            legend=True,
            cbar=False,
            alpha=0.3,
            fill=True,
            color="#ffd700",
            ax=ax,
        )

    # set frame config
    ax.set_xlim(0, FRAME_WIDTH)
    ax.set_ylim(FRAME_HEIGHT, 0)
    ax.axis("off")
