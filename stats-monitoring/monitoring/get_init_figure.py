from typing import List, Tuple

import matplotlib.pyplot as plt
from monitoring.constants import (
    FIGURE_HEIGHT,
    FIGURE_WIDTH,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    GRAPH_FONT_SIZE,
    GRAPH_HEIGHT,
    GRAPH_MARGIN,
    GRAPH_NUM,
    GRAPH_WIDTH,
)


def get_init_figure() -> Tuple[plt.figure, List]:
    """Get the figure and the position of each graph
    used in the monitoring environment based on the constant values.

    Returns:
        Tuple[plt.figure, List]: matplotlib figure, list of each graph axis
    """

    # set figure config
    plt.rcParams["font.size"] = GRAPH_FONT_SIZE
    fig = plt.figure(figsize=(FIGURE_WIDTH / 100, FIGURE_HEIGHT / 100))

    # set each graph position
    # each axes set [left, bottom, width, height]
    frame_ax = plt.axes(
        [
            (GRAPH_WIDTH + 3 * GRAPH_MARGIN) / FIGURE_WIDTH,
            (GRAPH_NUM * GRAPH_WIDTH + 5 * GRAPH_MARGIN) / FIGURE_HEIGHT,
            FRAME_WIDTH / FIGURE_WIDTH,
            FRAME_HEIGHT / FIGURE_HEIGHT,
        ]
    )

    x_hist_ax = plt.axes(
        [
            (GRAPH_WIDTH + 3 * GRAPH_MARGIN) / FIGURE_WIDTH,
            (GRAPH_NUM * GRAPH_HEIGHT + 6 * GRAPH_MARGIN + FRAME_HEIGHT)
            / FIGURE_HEIGHT,
            FRAME_WIDTH / FIGURE_WIDTH,
            GRAPH_HEIGHT / FIGURE_HEIGHT,
        ]
    )

    y_hist_ax = plt.axes(
        [
            2 * GRAPH_MARGIN / FIGURE_WIDTH,
            (GRAPH_NUM * GRAPH_WIDTH + 5 * GRAPH_MARGIN) / FIGURE_HEIGHT,
            GRAPH_WIDTH / FIGURE_WIDTH,
            FRAME_HEIGHT / FIGURE_HEIGHT,
        ]
    )

    mean_graph_ax = plt.axes(
        [
            (GRAPH_WIDTH + 3 * GRAPH_MARGIN) / FIGURE_WIDTH,
            (2 * GRAPH_WIDTH + 4 * GRAPH_MARGIN) / FIGURE_HEIGHT,
            FRAME_WIDTH / FIGURE_WIDTH,
            GRAPH_HEIGHT / FIGURE_HEIGHT,
        ]
    )

    acc_graph_ax = plt.axes(
        [
            (GRAPH_WIDTH + 3 * GRAPH_MARGIN) / FIGURE_WIDTH,
            (GRAPH_WIDTH + 3 * GRAPH_MARGIN) / FIGURE_HEIGHT,
            FRAME_WIDTH / FIGURE_WIDTH,
            GRAPH_HEIGHT / FIGURE_HEIGHT,
        ]
    )

    cnt_graph_ax = plt.axes(
        [
            (GRAPH_WIDTH + 3 * GRAPH_MARGIN) / FIGURE_WIDTH,
            2 * GRAPH_MARGIN / FIGURE_HEIGHT,
            FRAME_WIDTH / FIGURE_WIDTH,
            GRAPH_HEIGHT / FIGURE_HEIGHT,
        ]
    )

    axs = [frame_ax, x_hist_ax, y_hist_ax, mean_graph_ax, acc_graph_ax, cnt_graph_ax]

    return fig, axs
