import matplotlib.pyplot as plt

from omegaconf import DictConfig, open_dict
from typing import Tuple


def get_init_figure(cfg: DictConfig) -> Tuple[plt.figure, list]:
    """ get the figure and the position of each graph
    used in the monitoring environment based on the config.

    :param cfg: hydra config for monitoring environment
    :return: matplotlib figure, list of each graph axis
    """
    # load config
    frame_width = cfg['frame']['width']
    frame_height = cfg['frame']['height']
    x_axis_height = cfg['graph']['x_axis_height']
    y_axis_height = cfg['graph']['y_axis_height']
    stats_graph_num = cfg['graph']['stats_graph_num']
    margin = cfg['graph']['margin']

    # set figure config
    plt.rcParams["font.size"] = cfg['graph']['font_size']
    figure_width = frame_width + y_axis_height + 5 * margin
    figure_height = frame_height + (x_axis_height + margin) * (stats_graph_num + 1) + 4 * margin
    with open_dict(cfg):
        cfg['figure']['width'] = figure_width
        cfg['figure']['height'] = figure_height
    fig = plt.figure(figsize=(figure_width / 100, figure_height / 100))

    # set each graph position
    # each axes set [left, bottom, width, height]
    frame_ax = plt.axes([
        (y_axis_height + 3 * margin) / figure_width,
        (stats_graph_num * y_axis_height + 5 * margin) / figure_height,
        frame_width / figure_width,
        frame_height / figure_height])

    x_hist_ax = plt.axes([
        (y_axis_height + 3 * margin) / figure_width,
        (stats_graph_num * x_axis_height + 6 * margin + frame_height) / figure_height,
        frame_width / figure_width,
        x_axis_height / figure_height])

    y_hist_ax = plt.axes([
        2 * margin / figure_width,
        (stats_graph_num * y_axis_height + 5 * margin) / figure_height,
        y_axis_height / figure_width,
        frame_height / figure_height])

    mean_graph_ax = plt.axes([
        (y_axis_height + 3 * margin) / figure_width,
        (2 * y_axis_height + 4 * margin) / figure_height,
        frame_width / figure_width,
        x_axis_height / figure_height])

    acc_graph_ax = plt.axes([
        (y_axis_height + 3 * margin) / figure_width,
        (y_axis_height + 3 * margin) / figure_height,
        frame_width / figure_width,
        x_axis_height / figure_height])

    cnt_graph_ax = plt.axes([
        (y_axis_height + 3 * margin) / figure_width,
        2 * margin / figure_height,
        frame_width / figure_width,
        x_axis_height / figure_height])

    axs = [frame_ax, x_hist_ax, y_hist_ax, mean_graph_ax, acc_graph_ax, cnt_graph_ax]

    return fig, axs
