from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import FuncAnimation
from matplotlib.figure import Figure
from tqdm import trange

from monitoring.config import MonitoringConfig
from monitoring.constants import DATA_DIR
from monitoring.frame import load_one_hour_density, set_frame
from monitoring.logger import logger
from monitoring.position_distribution import load_current_coordinate, set_histogram
from monitoring.schema import StatsData
from monitoring.stats_metrics import load_statistics, set_stats_metrics


def update(
    frame_num: int,
    cfg: MonitoringConfig,
    axs: List,
    stats_data: StatsData,
) -> None:
    """Update function for movie frame

    Args:
        frame_num (int): current frame number
        cfg (MonitoringConfig): config for monitoring environment
        axs (List): list of matplotlib axis
        stats_data (StatsData): instance of each statistics data
    """
    # clear all axis
    for ax in axs:
        ax.cla()

    # set distribution histogram of current frame
    coordinate_frame_num = int(frame_num - frame_num % cfg.animation.density_interval)
    current_coordinate_df = load_current_coordinate(cfg, coordinate_frame_num)
    set_histogram(cfg, current_coordinate_df, axs[1], axs[2])

    # set current frame
    if cfg.animation.display_dot:
        detected_coordinate_df = current_coordinate_df.copy()
    else:
        detected_coordinate_df = pd.DataFrame()
    # get density map 1 hour ago
    if cfg.animation.display_density:
        one_hour_density_df = load_one_hour_density(cfg.path.coordinate_directory, frame_num)
    else:
        one_hour_density_df = pd.DataFrame()
    set_frame(cfg, frame_num, detected_coordinate_df, one_hour_density_df, axs[0])

    # set statistics metrics
    set_stats_metrics(cfg, frame_num, stats_data, axs[3], axs[4])


def generate_animation(cfg: MonitoringConfig, fig: Figure, axs: List) -> None:
    """Generate a video for monitoring environment

    Args:
        cfg (MonitoringConfig): config for monitoring environment
        fig (Figure): matplotlib figure
        axs (List): list of matplotlib axis
    """
    logger.info("[START] Load Statistics Data ...")
    stats_data = load_statistics(cfg)
    logger.info("------------ [DONE] ------------")

    anim = FuncAnimation(
        fig,
        update,
        frames=trange(cfg.animation.frame_number),
        interval=cfg.animation.interval,
        fargs=(cfg, axs, stats_data),
    )
    save_movie_path = str(DATA_DIR / cfg.path.save_movie_path)
    anim.save(save_movie_path, writer=cfg.animation.format)
    plt.close()

    logger.info(f"[END] Saved Animation in [{save_movie_path}]")
