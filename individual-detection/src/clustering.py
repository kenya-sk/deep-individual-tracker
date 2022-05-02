import logging

import cv2
import matplotlib

matplotlib.use("Agg")
import glob

import numpy as np
from sklearn.cluster import MeanShift

from utils import get_current_time_str, get_frame_number_from_path

# logger setting
current_time = get_current_time_str()
log_path = f"./logs/clustering_{current_time}.log"
logging.basicConfig(
    filename=log_path,
    level=logging.DEBUG,
    format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
)
logger = logging.getLogger(__name__)


def apply_clustering_to_density_map(
    density_map: np.array, band_width: int, thresh: float = 0.5
) -> np.array:
    """Apply Mean-Shift Clustering to density maps

    Args:
        density_map (np.array): _description_
        band_width (int): _description_
        thresh (float, optional): _description_. Defaults to 0.5.

    Returns:
        np.array: _description_
    """
    # search high value coordinates
    while True:
        # point[0]: y  point[1]: x
        point = np.where(density_map > thresh)
        # X[:, 0]: x  X[:,1]: y
        X = np.vstack((point[1], point[0])).T
        if X.shape[0] > 0:
            break
        else:
            return np.zeros((0, 2))

    # MeanShift clustering
    ms = MeanShift(bandwidth=band_width, n_jobs=-1)
    ms.fit(X)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_
    labels_unique = np.unique(labels)
    n_clusters = len(labels_unique)
    centroid_array = np.zeros((n_clusters, 2))
    for k in range(n_clusters):
        centroid_array[k] = cluster_centers[k]

    return centroid_array.astype(np.int32)


def batch_clustering(
    density_map_path: str, band_width: int, thresh: float, save_directory: str
) -> None:
    """Apply clustering to multiple files individually

    Args:
        density_map_path (str): _description_
        band_width (int): _description_
        thresh (float): _description_
        save_directory (str): _description_
    """
    file_list = glob.glob(density_map_path)
    for i, path in enumerate(file_list):
        # apply clustering for predicted density map
        pred_density_map = np.load(path)
        centroid_array = apply_clustering_to_density_map(
            pred_density_map, band_width, thresh
        )

        # save clustering result
        frame_number = get_frame_number_from_path(path)
        save_path = f"{save_directory}/{frame_number}.csv"
        np.savetxt(
            save_path,
            centroid_array,
            fmt="%i",
            delimiter=",",
        )
        logger.info(f"Saved in '{save_path}', batch=[{i+1}/{len(file_list)}]")


def plot_prediction_box(
    image: np.array,
    centroid_array: np.array,
    hour: int,
    minute: int,
    output_directory: str,
    box_size: int = 12,
) -> None:
    """Plot the location and Box of detected individuals on the image

    Args:
        image (np.array): _description_
        centroid_array (np.array): _description_
        hour (int): _description_
        minute (int): _description_
        output_directory (str): _description_
        box_size (int, optional): _description_. Defaults to 12.

    Returns:
        _type_: _description_
    """

    # get coordinates of vertex(left top and right bottom)
    def get_rect_vertex(x, y, box_size):
        vertex = np.zeros((2, 2), dtype=np.uint16)
        shift = int(box_size / 2)
        # left top corner
        vertex[0][0] = x - shift
        vertex[0][1] = y - shift
        # right bottom corner
        vertex[1][0] = x + shift
        vertex[1][1] = y + shift

        return vertex

    cluster_number = centroid_array.shape[0]
    logger.info(f"Number of cluster: {cluster_number}")
    for i in range(cluster_number):
        x = int(centroid_array[i][0])
        y = int(centroid_array[i][1])
        img = cv2.circle(image, (x, y), 2, (0, 0, 255), -1, cv2.LINE_AA)
        vertex = get_rect_vertex(x, y, box_size)
        img = cv2.rectangle(
            img,
            (vertex[0][0], vertex[0][1]),
            (vertex[1][0], vertex[1][1]),
            (0, 0, 255),
            3,
        )

    # save plotted image
    save_path = f"{output_directory}/{hour}_{hour}.png"
    cv2.imwrite(save_path, img)
    logger.info(f"Saved in '{save_path}'. (data={hour}:{minute})")
