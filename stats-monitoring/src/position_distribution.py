import os

# [FIXME] support displot or kdeplot
import warnings
from glob import glob

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from constants import DATA_DIR, FIGURE_HEIGHT, FIGURE_WIDTH, FRAME_HEIGHT, FRAME_WIDTH
from exceptions import PathExistError
from logger import logger
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
        DATA_DIR / f"cfg['path']['coordinate_directory']/{frame_num}.csv"
    )
    if os.path.isfile(coordinate_path):
        coordinate_df = pd.read_csv(coordinate_path)
        coordinate_df.columns = ["x", "y"]
    else:
        message = f'coordinate_path="{coordinate_path}" is not exist.'
        logger.error(message)
        raise PathExistError(message)

    return coordinate_df


def load_past_coordinate(cfg: DictConfig) -> pd.DataFrame:
    """Load the past distribution used for comparison

    Args:
        cfg (DictConfig): hydra config for monitoring environment

    Returns:
        pd.DataFrame: DataFrame of past distribution
    """
    past_dir = str(DATA_DIR / cfg["path"]["past_coordinate_directory"])
    if os.path.isdir(past_dir):
        coordinate_path_lst = glob(f"{past_dir}/*.csv")
    else:
        message = f'past_dir="{past_dir}" is not exist.'
        logger.error(message)
        raise PathExistError(message)

    past_coordinate_dctlst = {"x": [], "y": []}
    for path in tqdm(coordinate_path_lst):
        cord_df = pd.read_csv(path, header=None)
        past_coordinate_dctlst["x"].extend(cord_df[0].values)
        past_coordinate_dctlst["y"].extend(cord_df[1].values)

    return pd.DataFrame(past_coordinate_dctlst)


def set_histogram(
    cfg: DictConfig,
    current_coordinate_df: pd.DataFrame,
    past_coordinate_df: pd.DataFrame,
    x_ax: plt.axis,
    y_ax: plt.axis,
) -> None:
    """Set the individual distribution on histogram axis

    Args:
        cfg (DictConfig): hydra config for monitoring environment
        current_coordinate_df (pd.DataFrame): DataFrame containing current frame coordinates
        past_coordinate_df (pd.DataFrame): DataFrame containing past coordinates for comparison
        x_ax (plt.axis): matplotlib figure axis of X-histogram
        y_ax (plt.axis): matplotlib figure axis of Y-histogram
    """
    # plot X-axis histogram
    # set X-axis histogram bin number
    x_bins = int(FIGURE_WIDTH / cfg["histogram"]["x_bin_granularity"])

    # past 1 hour distribution
    sns.distplot(
        past_coordinate_df["x"],
        bins=x_bins,
        color="#ff6347",
        axlabel=False,
        label="Past Distribution \n(1 hour)",
        hist=False,
        kde=True,
        rug=False,
        ax=x_ax,
    )

    # current X-axis distribution
    sns.distplot(
        current_coordinate_df["x"],
        bins=x_bins,
        color="#0000cd",
        axlabel=False,
        label="Current Distribution",
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

    # past 1 hour distribution
    sns.distplot(
        past_coordinate_df["y"],
        bins=y_bins,
        color="#ff6347",
        axlabel=False,
        label="Past \nDistribution \n(1 hour)",
        hist=False,
        kde=True,
        rug=False,
        vertical=True,
        ax=y_ax,
    )

    # current Y-axis distribution
    sns.distplot(
        current_coordinate_df["y"],
        bins=y_bins,
        color="#0000cd",
        axlabel=False,
        label="Current \nDistribution",
        hist=False,
        kde=True,
        rug=False,
        vertical=True,
        ax=y_ax,
    )

    # set Y-axis histogram parameters
    y_ax.legend(bbox_to_anchor=(0.5, 1.3), loc="upper center")
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
