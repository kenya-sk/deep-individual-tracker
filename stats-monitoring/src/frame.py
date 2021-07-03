import cv2
import os
import pandas as pd
import seaborn as sns
import sys
import matplotlib.pyplot as plt
from omegaconf import DictConfig
from typing import NoReturn


def load_one_hour_density(coord_dirc: str, end_frame_num: int) -> pd.DataFrame:
    """ load the density over the past 1 hour

    :param coord_dirc: directory name of coordinate data
    :param end_frame_num: frame number of end data
    :return: DataFrame containing past 1 hour coordinates
    """
    # get frame number 1 hour ago
    start_frame_num = max(0, end_frame_num - 3600)
    # get the distribution of individuals over the past 1 hour
    past_density_dctlst = {'x': [], 'y': []}
    for frame_num in range(start_frame_num, end_frame_num):
        path = f'{coord_dirc}/{frame_num}.csv'
        if os.path.isfile(path):
            cord_df = pd.read_csv(path, header=None)
            past_density_dctlst['x'].extend(cord_df[0].values)
            past_density_dctlst['y'].extend(cord_df[1].values)
        else:
            print(f'[WARNING] Not Exist Path: {path}')

        return pd.DataFrame(past_density_dctlst)


def set_frame(cfg: DictConfig, frame_num: int, detected_coordinate_df: pd.DataFrame,
              one_hour_density_df: pd.DataFrame, ax: plt.axis) -> NoReturn:
    """ set the current frame on frame axis

    :param cfg: hydra config for monitoring environment
    :param frame_num: current frame number
    :param detected_coordinate_df: DataFrame containing current frame coordinates
    :param one_hour_density_df: DataFrame containing past 1 hour coordinates
    :param ax: matplotlib figure axis of frame
    :return: no return value
    """
    frame_path = os.path.join(cfg['path']['frame_directory'], f'{frame_num}.png')
    if os.path.isfile(frame_path):
        frame = cv2.imread(frame_path)
    else:
        print(f'[ERROR] Not Exist Path: {frame_path}')
        sys.exit(1)

    # set current frame
    ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # plot dots on the detected tuna
    if len(detected_coordinate_df) != 0:
        assert detected_coordinate_df['x'] == detected_coordinate_df['y']
        ax.plot(detected_coordinate_df['x'], detected_coordinate_df['y'],
                linestyle='', marker='o', ms=4.0, color='r')

    # display density map of past 1 hour
    if len(one_hour_density_df) != 0:
        sns.kdeplot(data=one_hour_density_df, x='x', y='y', cmap='RdYlGn_r', legend='Tuna density',
                    cbar=False, alpha=0.3, fill=True, color="#ffd700", ax=ax)

    # set frame config
    ax.set_xlim(0, cfg['frame']['width'])
    ax.set_ylim(cfg['frame']['height'], 0)
    ax.axis('off')
