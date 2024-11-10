import os
from pathlib import Path
from typing import Dict

import numpy as np
from matplotlib.axes import Axes

from monitoring.config import MonitoringConfig
from monitoring.constants import (
    DATA_DIR,
    GRAPH_FONT_SIZE,
    LABEL_IDX,
    LABEL_IDX_MAX,
    TIME_LABELS,
)
from monitoring.exceptions import PathNotExistError, StatsKeyNotExistError
from monitoring.logger import logger


def load_statistics(cfg: MonitoringConfig) -> Dict[str, np.ndarray]:
    """Load the statistics used in the monitoring environment

    Args:
        cfg (MonitoringConfig): config for monitoring environment

    Returns:
        dict: dictionary containing each stats array
    """

    def load_value(path: Path) -> np.ndarray:
        """Load statistics value from CSV file

        Args:
            path (Path): CSV file path

        Returns:
            np.ndarray: target numpy array
        """
        if os.path.isfile(path):
            return np.loadtxt(path, delimiter=",")
        else:
            message = f'path="{path}" is not exist.'
            logger.error(message)
            raise PathNotExistError(message)

    stats_dict = {
        "mean": load_value(DATA_DIR / cfg.path.mean_speed_path),
        "past_mean": load_value(DATA_DIR / cfg.path.past_mean_speed_path),
        "acceleration": load_value(DATA_DIR / cfg.path.acceleration_count_path),
        "past_acceleration": load_value(DATA_DIR / cfg.path.past_acceleration_count_path),
    }

    return stats_dict


def load_array(stats_dict: Dict[str, np.ndarray], key: str) -> np.ndarray:
    """Load array from statistics dictionary

    Args:
        stats_dict (Dict): dictionary containing each stats array
        key (str): dictionary key

    Returns:
        np.ndarray: target stats array
    """
    if key in stats_dict.keys():
        return stats_dict[key]
    else:
        message = f'key="{key}" is not exist in stats_dict.'
        logger.error(message)
        raise StatsKeyNotExistError(message)


def set_stats_metrics(
    cfg: MonitoringConfig,
    frame_num: int,
    stats_dict: Dict[str, np.ndarray],
    mean_ax: Axes,
    acc_ax: Axes,
) -> None:
    """Set the stats for the current frame on each axis

    Args:
        cfg (MonitoringConfig): config for monitoring environment
        frame_num (int): current frame number
        stats_dict (Dict): dictionary containing each stats array
        mean_ax (Axes): matplotlib figure axis of mean speed
        acc_ax (Axes): matplotlib figure axis of acceleration count
    """
    mean_arr = load_array(stats_dict, "mean")
    x = [i for i in range(len(mean_arr))]

    # plot mean speed
    mean_ax.plot(x[: frame_num + 1], mean_arr[: frame_num + 1])
    mean_ax.set_xlim(0, LABEL_IDX_MAX)
    mean_ax.set_ylim(0, cfg.statistics.mean_max)
    mean_ax.set_ylabel("Mean moving \ndistance \n[pixel $s^{-1}$]")
    mean_ax.tick_params(labelbottom=False, bottom=False)
    mean_ax.axvline(frame_num, 0, 100, color="black", linestyle="dashed")

    # plot cumulate acceleration count
    acc_arr = load_array(stats_dict, "acceleration")
    acc_ax.plot(x[: frame_num + 1], acc_arr[: frame_num + 1])
    acc_ax.set_xlim(0, LABEL_IDX_MAX)
    acc_ax.set_ylim(0, cfg.statistics.acceleration_max)
    acc_ax.set_ylabel("Cumulative \nnumber of \nsudden acceleration \n[$d^{-1}$]")
    acc_ax.axvline(frame_num, 0, 100, color="black", linestyle="dashed")

    # set ticks only last graph
    acc_ax.set_xticks(LABEL_IDX)
    acc_ax.set_xticklabels(TIME_LABELS, fontsize=GRAPH_FONT_SIZE - 2)
