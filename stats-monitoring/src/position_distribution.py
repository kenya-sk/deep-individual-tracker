import os
import sys

# [FIXME] support displot or kdeplot
import warnings
from glob import glob
from typing import NoReturn

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from omegaconf import DictConfig
from tqdm import tqdm

warnings.resetwarnings()
warnings.simplefilter("ignore", FutureWarning)


def load_current_coordinate(cfg: DictConfig, frame_num: int) -> pd.DataFrame:
    """load the current frame distribution

    :param cfg: hydra config for monitoring environment
    :param frame_num: current frame number
    :return: DataFrame of current frame distribution
    """
    # Each coordinate is defined by the column names "x" and "y"
    coordinate_path = os.path.join(
        cfg["path"]["coordinate_directory"], f"{frame_num}.csv"
    )
    if os.path.isfile(coordinate_path):
        coordinate_df = pd.read_csv(coordinate_path)
        coordinate_df.columns = ["x", "y"]
    else:
        print(f"[ERROR] Not Exist Path: {coordinate_path}")
        sys.exit(1)

    return coordinate_df


def load_past_coordinate(cfg: DictConfig) -> pd.DataFrame:
    """load the past distribution used for comparison

    :param cfg: hydra config for monitoring environment
    :return: DataFrame of past distribution
    """
    if os.path.isdir(cfg["path"]["past_coordinate_directory"]):
        coordinate_path_lst = glob(f'{cfg["path"]["past_coordinate_directory"]}/*.csv')
    else:
        print(
            f'[ERROR] Not Exist Directory: {cfg["path"]["past_coordinate_directory"]}'
        )
        sys.exit(1)

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
) -> NoReturn:
    """set the individual distribution on histogram axis

    :param cfg: hydra config for monitoring environment
    :param current_coordinate_df: DataFrame containing current frame coordinates
    :param past_coordinate_df: DataFrame containing past coordinates for comparison
    :param x_ax: matplotlib figure axis of X-histogram
    :param y_ax: matplotlib figure axis of Y-histogram
    :return: no return values
    """

    # plot X-axis histogram
    # set X-axis histogram bin number
    x_bins = int(cfg["figure"]["width"] / cfg["histogram"]["x_bin_granularity"])

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
    x_ax.set_xlim(0, cfg["frame"]["width"])
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
    y_bins = int(cfg["figure"]["height"] / cfg["histogram"]["y_bin_granularity"])

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
    y_ax.set_ylim(0, cfg["frame"]["height"])
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
