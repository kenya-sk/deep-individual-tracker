import argparse
import logging

import cv2
import matplotlib

matplotlib.use("Agg")
import glob

import numpy as np
from sklearn.cluster import MeanShift

logger = logging.getLogger(__name__)


def clustering(dens_map, band_width, thresh=0):
    """
    clustering density map
    """
    # search high value coordinates
    while True:
        # point[0]: y  point[1]: x
        point = np.where(dens_map > thresh)
        # X[:, 0]: x  X[:,1]: y
        X = np.vstack((point[1], point[0])).T
        if X.shape[0] > 0:
            break
        else:
            if thresh > 0:
                thresh -= 0.05
            else:
                return np.zeros((0, 2))

    # MeanShift clustering
    logger.debug("START: clustering")
    ms = MeanShift(bandwidth=band_width, seeds=X)
    ms.fit(X)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_
    labels_unique = np.unique(labels)
    n_clusters = len(labels_unique)
    centroid_arr = np.zeros((n_clusters, 2))
    for k in range(n_clusters):
        centroid_arr[k] = cluster_centers[k]
    logger.debug("DONE: clustering")

    return centroid_arr.astype(np.int32)


def batch_clustering(args):
    """
    clustering multiple files
    """
    file_lst = glob.glob(args.dens_map_path)
    for i, file_path in enumerate(file_lst):
        logger.debug("current data: {0} / {1}".format(i + 1, len(file_lst)))
        est_dens_map = np.load(file_path)
        centroid_arr = clustering(est_dens_map, args.band_width, args.thresh)
        file_num = file_path.split("/")[-1][:-4]
        np.savetxt(
            "{0}/{1}.csv".format(args.out_clustering_dirc, file_num),
            centroid_arr,
            fmt="%i",
            delimiter=",",
        )


def plot_prediction_box(
    img, centroid_arr, hour, minute, out_pred_box_dirc, box_size=12
):
    """
    draw square box of predict point
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

    logger.debug("Number of cluster: {0}".format(centroid_arr.shape[0]))
    for i in range(centroid_arr.shape[0]):
        x = int(centroid_arr[i][0])
        y = int(centroid_arr[i][1])
        img = cv2.circle(img, (x, y), 2, (0, 0, 255), -1, cv2.LINE_AA)
        vertex = get_rect_vertex(x, y, box_size)
        img = cv2.rectangle(
            img,
            (vertex[0][0], vertex[0][1]),
            (vertex[1][0], vertex[1][1]),
            (0, 0, 255),
            3,
        )

    cv2.imwrite("{0}/{1}_{2}.png".format(out_pred_box_dirc, hour, minute), img)
    logger.debug("Done({0}:{1}): plot estimation box\n".format(hour, minute))


def make_clustering_parse():
    parser = argparse.ArgumentParser(
        prog="clustering.py",
        usage="clustering pred point",
        description="description",
        epilog="end",
        add_help=True,
    )

    # Data Argument
    parser.add_argument(
        "--dens_map_path",
        type=str,
        default="/Users/sakka/cnn_by_density_map/data/eval_image/pred/dens/*.npy",
    )
    parser.add_argument(
        "--out_clustering_dirc",
        type=str,
        default="/Users/sakka/cnn_by_density_map/data/eval_image/pred/cord2",
    )
    parser.add_argument(
        "--out_pred_box_dirc", type=str, default="/data/sakka/image/estBox"
    )

    # Parameter Argument
    parser.add_argument(
        "--band_width", type=int, default=10, help="band width of clustering"
    )
    parser.add_argument(
        "--thresh",
        type=float,
        default=0.45,
        help="threshold to be subjected to clustering",
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    logs_path = "/Users/sakka/cnn_by_density_map/logs/clustering.log"
    logging.basicConfig(
        filename=logs_path,
        level=logging.DEBUG,
        format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
    )
    args = make_clustering_parse()
    logger.debug("Running with args: {0}".format(args))
    batch_clustering(args)