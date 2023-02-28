import cv2
import matplotlib

matplotlib.use("Agg")
import glob

import numpy as np
from detector.logger import logger
from sklearn.cluster import MeanShift
from utils import get_file_name_from_path


def apply_clustering_to_density_map(
    density_map: np.array, band_width: int, thresh: float = 0.5
) -> np.array:
    """Apply Mean-Shift Clustering to density maps.
    The center coordinates of the computed clusters are considered the final individual detection position.

    Args:
        density_map (np.array): predicted density map by trained CNN
        band_width (int): bandwidth used in the RBF kernel.
        thresh (float, optional): threshold value to be applied to the density map.
            Values below the threshold are ignored. Defaults to 0.5.

    Returns:
        np.array: array containing the cluster's centroid.
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
    ms = MeanShift(bandwidth=band_width)
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
    density_map_directory: str, band_width: int, thresh: float, save_directory: str
) -> None:
    """Apply clustering to multiple files individually.

    Args:
        density_map_directory (str): directory of density map
        band_width (int): bandwidth used in the RBF kernel.
        thresh (float): threshold value to be applied to the density map.
        save_directory (str): directory of save results
    """
    file_list = glob.glob(f"{density_map_directory}/*.npy")
    for i, path in enumerate(file_list):
        # apply clustering for predicted density map
        pred_density_map = np.load(path)
        centroid_array = apply_clustering_to_density_map(
            pred_density_map, band_width, thresh
        )

        # save clustering result
        frame_number = get_file_name_from_path(path)
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
    save_path: str,
    box_size: int = 12,
) -> None:
    """Plot the location and Box of detected individuals on the image.

    Args:
        image (np.array): raw image
        centroid_array (np.array): array of calculted centroid
        save_path (str): path of saved image
        box_size (int, optional): size of the box to be plotted. Defaults to 12.
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
    cv2.imwrite(save_path, img)
    logger.info(f"Saved in '{save_path}'.")
