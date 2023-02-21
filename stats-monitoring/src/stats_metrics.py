import os
import sys
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig

from constants import DATA_DIR, TIME_LABELS

# if frame interval is 1000 milli second
# LABEL_IDX = [i for i in range(0, 28801, 1800)]
# if frame interval is 33 milli second (30FPS)
LABEL_IDX = [i for i in range(0, 864001, 54000)]


def load_statistics(cfg: DictConfig) -> dict:
    """Load the statistics used in the monitoring environment

    Args:
        cfg (DictConfig): hydra config for monitoring environment

    Returns:
        dict: dictionary containing each stats array
    """

    def load_value(path: str) -> np.array:
        """Load statistics value from CSV file

        Args:
            path (str): CSV file path

        Returns:
            np.array: target numpy array
        """
        if os.path.isfile(path):
            return np.loadtxt(path, delimiter=",")
        else:
            print(f"[ERROR] Not Exist File: {path}")
            sys.exit(1)

    stats_dict = {
        "mean": load_value(DATA_DIR / cfg["path"]["mean_speed_path"]),
        "past_mean": load_value(DATA_DIR / cfg["path"]["past_mean_speed_path"]),
        "count": load_value(DATA_DIR / cfg["path"]["individual_count_path"]),
        "past_count": load_value(DATA_DIR / cfg["path"]["past_individual_count_path"]),
        "acceleration": load_value(DATA_DIR / cfg["path"]["acceleration_count_path"]),
        "past_acceleration": load_value(
            DATA_DIR / cfg["path"]["past_acceleration_count_path"]
        ),
    }

    return stats_dict


def load_array(stats_dict: Dict, key: str) -> np.array:
    """Load array from statistics dictionary

    Args:
        stats_dict (Dict): dictionary containing each stats array
        key (str): dictionary key

    Returns:
        np.array: target stats array
    """
    if key in stats_dict.keys():
        return stats_dict[key]
    else:
        print(f"[ERROR] Not Exist Key: {key}")
        sys.exit(1)


def set_stats_metrics(
    cfg: DictConfig,
    frame_num: int,
    stats_dict: Dict,
    mean_ax: plt.axis,
    acc_ax: plt.axis,
    cnt_ax: plt.axis,
) -> None:
    """Set the stats for the current frame on each axis

    Args:
        cfg (DictConfig): hydra config for monitoring environment
        frame_num (int): current frame number
        stats_dict (Dict): dictionary containing each stats array
        mean_ax (plt.axis): matplotlib figure axis of mean speed
        acc_ax (plt.axis): matplotlib figure axis of acceleration count
        cnt_ax (plt.axis): matplotlib figure axis of individual count
    """
    mean_arr = load_array(stats_dict, "mean")
    x = [i for i in range(len(mean_arr))]

    # plot current mean speed
    # mean_arr[frame_num:] = None
    past_mean_arr = load_array(stats_dict, "past_mean")
    mean_ax.plot(x, past_mean_arr, color="#ff6347", alpha=0.6, label="Past")
    mean_ax.plot(
        x[: frame_num + 1], mean_arr[: frame_num + 1], color="#0000cd", label="Current"
    )
    mean_ax.set_ylim(0, cfg["statistics"]["mean_max"])
    # mean_ax.legend(loc='upper left')
    mean_ax.set_ylabel("Mean Speed")
    mean_ax.tick_params(labelbottom=False, bottom=False)
    mean_ax.axvline(frame_num, 0, 100, color="black", linestyle="dashed")

    # plot current acceleration count
    acc_arr = load_array(stats_dict, "acceleration")
    past_acc_arr = load_array(stats_dict, "past_acceleration")
    acc_ax.plot(x, past_acc_arr, color="#ff6347", alpha=0.6, label="Past")
    acc_ax.plot(
        x[: frame_num + 1], acc_arr[: frame_num + 1], color="#0000cd", label="Current"
    )
    acc_ax.set_ylim(0, cfg["statistics"]["acceleration_max"])
    # acc_ax.legend(loc='upper left')
    acc_ax.set_ylabel("Cumulative \nAcceleration \nCount")
    acc_ax.tick_params(labelbottom=False, bottom=False)
    acc_ax.axvline(frame_num, 0, 100, color="black", linestyle="dashed")

    # plot current individual count
    cnt_arr = load_array(stats_dict, "count")
    past_cnt_arr = load_array(stats_dict, "past_count")
    cnt_ax.plot(x, past_cnt_arr, color="#ff6347", alpha=0.6, label="Past")
    cnt_ax.plot(
        x[: frame_num + 1], cnt_arr[: frame_num + 1], color="#0000cd", label="Current"
    )
    cnt_ax.set_ylim(0, cfg["statistics"]["count_max"])
    # cnt_ax.legend(loc='lower left')
    cnt_ax.set_ylabel("Tuna Count")
    cnt_ax.axvline(frame_num, 0, 100, color="black", linestyle="dashed")

    # set ticks only last graph
    cnt_ax.set_xticks(LABEL_IDX)
    cnt_ax.set_xticklabels(TIME_LABELS, fontsize=int(cfg["graph"]["font_size"]) - 2)
