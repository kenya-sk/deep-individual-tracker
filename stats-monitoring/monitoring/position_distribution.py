import os

# [FIXME] support displot or kdeplot
import warnings
from glob import glob

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from monitoring.constants import (
    DATA_DIR,
    FIGURE_HEIGHT,
    FIGURE_WIDTH,
    FRAME_HEIGHT,
    FRAME_WIDTH,
)
from monitoring.exceptions import PathNotExistError
from monitoring.logger import logger
from omegaconf import DictConfig
from tqdm import tqdm

warnings.resetwarnings()
warnings.simplefilter("ignore", FutureWarning)


def load_current_coordinate(cfg: DictConfig, frame_num: int) -> pd.DataFrame:
    """Load the current frame distribution

    Args:
        cfg (DictConfig): hydra config for monitoring environment
        frame_num (int): current frame number

    Returns:
        pd.DataFrame: DataFrame of current frame distribution
    """
    # Each coordinate is defined by the column names "x" and "y"
    coordinate_path = str(
        DATA_DIR / cfg["path"]["coordinate_directory"] / f"{frame_num}.csv"
    )
    if os.path.isfile(coordinate_path):
        coordinate_df = pd.read_csv(coordinate_path)
        coordinate_df.columns = ["x", "y"]
    else:
        message = f'coordinate_path="{coordinate_path}" is not exist.'
        logger.error(message)
        raise PathNotExistError(message)

    return coordinate_df


def set_histogram(
    cfg: DictConfig,
    current_coordinate_df: pd.DataFrame,
    x_ax: plt.axis,
    y_ax: plt.axis,
) -> None:
    """Set the individual distribution on histogram axis

    Args:
        cfg (DictConfig): hydra config for monitoring environment
        current_coordinate_df (pd.DataFrame): DataFrame containing current frame coordinates
        x_ax (plt.axis): matplotlib figure axis of X-histogram
        y_ax (plt.axis): matplotlib figure axis of Y-histogram
    """
    # plot X-axis histogram
    # set X-axis histogram bin number
    x_bins = int(FIGURE_WIDTH / cfg["histogram"]["x_bin_granularity"])

    # current X-axis distribution
    sns.distplot(
        current_coordinate_df["x"],
        bins=x_bins,
        color="#0000cd",
        axlabel=False,
        hist=False,
        kde=True,
        rug=False,
        ax=x_ax,
    )

    # set X-axis histogram parameters
    x_ax.set_xlim(0, FRAME_WIDTH)
    x_ax.tick_params(
        labelbottom=False,
        labelleft=False,
        labelright=False,
        labeltop=False,
        bottom=False,
        left=False,
        right=False,
        top=False,
    )

    # plot Y-axis histogram
    # set Y-axis histogram bin number
    y_bins = int(FIGURE_HEIGHT / cfg["histogram"]["y_bin_granularity"])

    # current Y-axis distribution
    sns.distplot(
        current_coordinate_df["y"],
        bins=y_bins,
        color="#0000cd",
        axlabel=False,
        hist=False,
        kde=True,
        rug=False,
        vertical=True,
        ax=y_ax,
    )

    # set Y-axis histogram parameters
    y_ax.set_ylim(0, FRAME_HEIGHT)
    y_ax.invert_xaxis()
    y_ax.invert_yaxis()
    y_ax.tick_params(
        labelbottom=False,
        labelleft=False,
        labelright=False,
        labeltop=False,
        bottom=False,
        left=False,
        right=False,
        top=False,
    )
