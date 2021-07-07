import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from omegaconf import DictConfig
from typing import NoReturn

# if frame interval is 1000 milli second
# LABEL_IDX = [i for i in range(0, 28801, 1800)]
# if frame interval is 33 milli second (30FPS)
LABEL_IDX = [i for i in range(0, 864001, 54000)]
LABELS = ['9:00', '9:30', '10:00', '10:30', '11:00', '11:30', '12:00', '12:30',
          '13:00', '13:30', '14:00', '14:30', '15:00', '15:30', '16:00', '16:30', '17:00']


def load_statistics(cfg: DictConfig) -> dict:
    """ Load the statistics used in the monitoring environment

    :param cfg: hydra config for monitoring environment
    :return: dictionary containing each stats array
    """
    def load_value(path: str) -> np.array:
        """ Load statistics value from CSV file

        :param path: CSV file path
        :return: target numpy array
        """
        if os.path.isfile(path):
            return np.loadtxt(path, delimiter=',')
        else:
            print(f'[ERROR] Not Exist File: {path}')
            sys.exit(1)

    stats_dict = {'mean': load_value(cfg['path']['mean_speed_path']),
                  'past_mean': load_value(cfg['path']['past_mean_speed_path']),
                  'count': load_value(cfg['path']['individual_count_path']),
                  'past_count': load_value(cfg['path']['past_individual_count_path']),
                  'acceleration': load_value(cfg['path']['acceleration_count_path']),
                  'past_acceleration': load_value(cfg['path']['past_acceleration_count_path'])}

    return stats_dict


def load_array(stats_dict: dict, key: str) -> np.array:
    """ load array from statistics dictionary

    :param stats_dict: dictionary containing each stats array
    :param key: dictionary key
    :return: target stats array
    """
    if key in stats_dict.keys():
        return stats_dict[key]
    else:
        print(f'[ERROR] Not Exist Key: {key}')
        sys.exit(1)


def set_stats_metrics(cfg: DictConfig, frame_num: int, stats_dict: dict,
                      mean_ax: plt.axis, acc_ax: plt.axis, cnt_ax: plt.axis) -> NoReturn:
    """ set the stats for the current frame on each axis

    :param cfg: hydra config for monitoring environment
    :param frame_num: current frame number
    :param stats_dict: dictionary containing each stats array
    :param mean_ax: matplotlib figure axis of mean speed
    :param acc_ax: matplotlib figure axis of acceleration count
    :param cnt_ax: matplotlib figure axis of individual count
    :return: no return value
    """
    mean_arr = load_array(stats_dict, 'mean')
    x = [i for i in range(len(mean_arr))]

    # plot current mean speed
    # mean_arr[frame_num:] = None
    past_mean_arr = load_array(stats_dict, 'past_mean')
    mean_ax.plot(x, past_mean_arr, color='#ff6347', alpha=0.6, label='Past')
    mean_ax.plot(x[:frame_num+1], mean_arr[:frame_num+1], color='#0000cd', label='Current')
    mean_ax.set_ylim(0, cfg['statistics']['mean_max'])
    # mean_ax.legend(loc='upper left')
    mean_ax.set_ylabel('Mean Speed')
    mean_ax.tick_params(labelbottom=False, bottom=False)
    mean_ax.axvline(frame_num, 0, 100, color='black', linestyle='dashed')

    # plot current acceleration count
    acc_arr = load_array(stats_dict, 'acceleration')
    past_acc_arr = load_array(stats_dict, 'past_acceleration')
    acc_ax.plot(x, past_acc_arr, color='#ff6347', alpha=0.6, label='Past')
    acc_ax.plot(x[:frame_num+1], acc_arr[:frame_num+1], color='#0000cd', label='Current')
    acc_ax.set_ylim(0, cfg['statistics']['acceleration_max'])
    # acc_ax.legend(loc='upper left')
    acc_ax.set_ylabel('Cumulative \nAcceleration \nCount')
    acc_ax.tick_params(labelbottom=False, bottom=False)
    acc_ax.axvline(frame_num, 0, 100, color='black', linestyle='dashed')

    # plot current individual count
    cnt_arr = load_array(stats_dict, 'count')
    past_cnt_arr = load_array(stats_dict, 'past_count')
    cnt_ax.plot(x, past_cnt_arr, color='#ff6347', alpha=0.6, label='Past')
    cnt_ax.plot(x[:frame_num+1], cnt_arr[:frame_num+1], color='#0000cd', label='Current')
    cnt_ax.set_ylim(0, cfg['statistics']['count_max'])
    # cnt_ax.legend(loc='lower left')
    cnt_ax.set_ylabel('Tuna Count')
    cnt_ax.axvline(frame_num, 0, 100, color='black', linestyle='dashed')

    # set ticks only last graph
    cnt_ax.set_xticks(LABEL_IDX)
    cnt_ax.set_xticklabels(LABELS, fontsize=int(cfg['graph']['font_size'])-2)
