import logging
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import FuncAnimation
from omegaconf import DictConfig
from tqdm import trange

from constants import DATA_DIR
from frame import load_one_hour_density, set_frame
from position_distribution import (load_current_coordinate,
                                   load_past_coordinate, set_histogram)
from stats_metrics import load_statistics, set_stats_metrics

logger = logging.getLogger(__name__)


def update(
    frame_num: int,
    cfg: DictConfig,
    axs: List,
    past_coordinate_df: pd.DataFrame,
    stats_dict: Dict,
) -> None:
    """Update function for movie frame

    Args:
        frame_num (int): current frame number
        cfg (DictConfig): hydra config for monitoring environment
        axs (List): list of matplotlib axis
        past_coordinate_df (pd.DataFrame): DataFrame containing past coordinates for comparison
        stats_dict (Dict): dictionary containing each stats array
    """
    # clear all axis
    for ax in axs:
        ax.cla()

    # set distribution histogram of current frame
    current_coordinate_df = load_current_coordinate(cfg, frame_num)
    set_histogram(cfg, current_coordinate_df, past_coordinate_df, axs[1], axs[2])

    # set current frame
    if cfg["animation"]["display_dot"]:
        detected_coordinate_df = current_coordinate_df.copy()
    else:
        detected_coordinate_df = pd.DataFrame()
    # get density map 1 hour ago
    if cfg["animation"]["display_density"]:
        one_hour_density_df = load_one_hour_density(
            cfg["path"]["coordinate_directory"], frame_num
        )
    else:
        one_hour_density_df = pd.DataFrame()
    set_frame(cfg, frame_num, detected_coordinate_df, one_hour_density_df, axs[0])

    # set statistics metrics
    set_stats_metrics(cfg, frame_num, stats_dict, axs[3], axs[4], axs[5])


def generate_animation(cfg: DictConfig, fig: plt.figure, axs: List) -> None:
    """Generate a video for monitoring environment

    Args:
        cfg (DictConfig): hydra config for monitoring environment
        fig (plt.figure): matplotlib figure
        axs (List): list of matplotlib axis
    """
    logger.info("[START] Load Coordinate Data ...")
    past_coordinate_df = load_past_coordinate(cfg)
    logger.info("------------ [DONE] ------------")

    logger.info("[START] Load Statistics Data ...")
    stats_dict = load_statistics(cfg)
    logger.info("------------ [DONE] ------------")

    anim = FuncAnimation(
        fig,
        update,
        frames=trange(int(cfg["animation"]["frame_number"])),
        interval=cfg["animation"]["interval"],
        fargs=(cfg, axs, past_coordinate_df, stats_dict),
    )
    save_movie_path = DATA_DIR / cfg["path"]["save_movie_path"]
    anim.save(save_movie_path, writer=cfg["animation"]["format"])
    plt.close()

    logger.info(f"[END] Saved Animation in [{save_movie_path}]")
