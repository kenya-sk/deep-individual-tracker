import glob
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from scipy import optimize
from tqdm import tqdm

from detector.config import EvaluateConfig, load_config
from detector.constants import CONFIG_DIR, DATA_DIR, EVALUATE_CONFIG_NAME
from detector.evaluation_metrics import eval_metrics, output_evaluation_report
from detector.index_manager import IndexManager
from detector.logger import logger
from detector.process_dataset import load_mask_image


def get_ground_truth(
    ground_truth_path: Path, mask_image: Optional[np.ndarray]
) -> np.ndarray:
    """Plots the coordinates of answer label on the black image(all value 0)
    and creates a correct image for accuracy evaluation.

    Args:
        ground_truth_path (Path): coordinates of the estimated target (.csv)
        mask_image (np.ndarray, optional): binay mask image

    Returns:
        np.ndarray: array showing the positon of target
    """

    ground_truth_array = np.loadtxt(ground_truth_path, delimiter=",", dtype="int32")
    if mask_image is None:
        return ground_truth_array
    else:
        index_manager = IndexManager(mask_image)
        valid_ground_truth_list = []
        for i in range(ground_truth_array.shape[0]):
            index_w = np.where(index_manager.index_w == ground_truth_array[i][0])
            index_h = np.where(index_manager.index_h == ground_truth_array[i][1])
            intersect = np.intersect1d(index_w, index_h)
            if len(intersect) == 1:
                valid_ground_truth_list.append(
                    [
                        index_manager.index_w[intersect[0]],
                        index_manager.index_h[intersect[0]],
                    ]
                )
        return np.array(valid_ground_truth_list, dtype="int32")


def eval_detection(
    predcit_centroid_array: np.ndarray,
    ground_truth_array: np.ndarray,
    detection_threshold: int,
) -> Tuple:
    """A prediction is considered successful when the distance between
    the predicted detection point and the ground-truth is less than the "discriminant threshold".

    Args:
        predcit_centroid_array (np.ndarray): coordinates of predicted point
        ground_truth_array (np.ndarray): coordinates of ground truth
        detection_threshold (int): threshold value used to determine detection

    Returns:
        Tuple: evaluate results
    """

    # make distance matrix
    predict_centroid_number = len(predcit_centroid_array)
    ground_truth_number = len(ground_truth_array)
    sample_number = max(predict_centroid_number, ground_truth_number)
    dist_matrix = np.zeros((sample_number, sample_number))
    for i in range(len(predcit_centroid_array)):
        for j in range(len(ground_truth_array)):
            dist_cord = predcit_centroid_array[i] - ground_truth_array[j]
            dist_matrix[i][j] = np.linalg.norm(dist_cord)

    # calculate by hangarian algorithm
    row, col = optimize.linear_sum_assignment(dist_matrix)
    indexes = []
    for i in range(sample_number):
        indexes.append([row[i], col[i]])
    valid_index_length = min(len(predcit_centroid_array), len(ground_truth_array))
    valid_lst = [i for i in range(valid_index_length)]
    if len(predcit_centroid_array) >= len(ground_truth_array):
        match_indexes = list(filter((lambda index: index[1] in valid_lst), indexes))
    else:
        match_indexes = list(filter((lambda index: index[0] in valid_lst), indexes))

    true_positive = 0
    if predict_centroid_number > ground_truth_number:
        false_positive = predict_centroid_number - ground_truth_number
        false_negative = 0
    else:
        false_positive = 0
        false_negative = ground_truth_number - predict_centroid_number

    for i in range(len(match_indexes)):
        pred_index = match_indexes[i][0]
        truth_index = match_indexes[i][1]
        if dist_matrix[pred_index][truth_index] <= detection_threshold:
            true_positive += 1
        else:
            false_positive += 1
            false_negative += 1

    return true_positive, false_positive, false_negative, sample_number


def evaluate(
    predict_dirctory: Path,
    ground_truth_dirctory: Path,
    mask_image_path: Path,
    detection_threshold: int,
) -> None:
    """The evaluation indices of accuracy, precision, recall, and f measure are calculated for each image.
    Then, the average value is calculated and the overall evaluation is performed.

    Args:
        predict_dirctory (Path): directory name of prediction results
        ground_truth_dirctory (Path): directory name of ground truth
        mask_image_path (Path): path of mask image
        detection_threshold (int): threshold value used to determine detection

    Returns:
        None:
    """

    # initialization
    true_positive_list = []
    false_positive_list = []
    false_negative_list = []
    sample_number_list = []

    predict_path_list = glob.glob(f"{predict_dirctory}/*.csv")
    for path in tqdm(predict_path_list):
        file_name = path.split("/")[-1]
        predcit_centroid_array = np.loadtxt(path, delimiter=",", dtype="int32")
        if predcit_centroid_array.shape[0] == 0:
            # "not" exist detection point case
            logger.info(
                f"file name: {file_name}, Not found detection point of centroid. Accuracy is 0.0"
            )
            true_positive_list.append(0)
            false_positive_list.append(0)
            false_negative_list.append(0)
            sample_number_list.append(1)
        else:
            # exist detection point case
            mask_image = load_mask_image(mask_image_path)
            ground_truth_array = get_ground_truth(
                ground_truth_dirctory / file_name, mask_image
            )
            true_pos, false_pos, false_neg, sample_number = eval_detection(
                predcit_centroid_array, ground_truth_array, detection_threshold
            )
            true_positive_list.append(true_pos)
            false_positive_list.append(false_pos)
            false_negative_list.append(false_neg)
            sample_number_list.append(sample_number)

            # calculate evaluation metrics
            basic_metrics = eval_metrics(true_pos, false_pos, false_neg, sample_number)
            logger.info(
                f"""file_name: {file_name},Accuracy: {basic_metrics.accuracy:.2f},
                Precision: {basic_metrics.precision:.2f},Recall: {basic_metrics.recall:.2f},
                F-Measure: {basic_metrics.f_measure:.2f}"""
            )

    output_evaluation_report(
        true_positive_list, false_positive_list, false_negative_list, sample_number_list
    )


def main() -> None:
    cfg = load_config(CONFIG_DIR / EVALUATE_CONFIG_NAME, EvaluateConfig)
    logger.info(f"Loaded config: {cfg}")

    # evaluate predction results
    evaluate(
        DATA_DIR / cfg.predict_directory,
        DATA_DIR / cfg.ground_truth_directory,
        DATA_DIR / cfg.mask_path,
        cfg.detection_threshold,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(e)
