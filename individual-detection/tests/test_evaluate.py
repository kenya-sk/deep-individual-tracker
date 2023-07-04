from pathlib import Path

import numpy as np
from detector.evaluate import eval_detection, get_ground_truth
from numpy.testing import assert_array_equal


def test_get_ground_truth(tmp_path: Path) -> None:
    # create dummy ground truth data
    ground_truth_path = str(tmp_path / "ground_truth.csv")
    raw_ground_truth_array = np.array([[10, 50], [200, 450], [500, 120], [130, 125]])
    np.savetxt(ground_truth_path, raw_ground_truth_array, delimiter=",", fmt="%d")

    # mask_image is None case
    ground_truth_array = get_ground_truth(ground_truth_path, None)
    expected_ground_truth_array = raw_ground_truth_array.copy()
    assert_array_equal(ground_truth_array, expected_ground_truth_array)

    # mask_image is not None case
    mask_image = np.ones((256, 256))
    ground_truth_array = get_ground_truth(ground_truth_path, mask_image)
    expected_ground_truth_array = np.array([[10, 50], [130, 125]])
    assert_array_equal(ground_truth_array, expected_ground_truth_array)


def test_eval_detection() -> None:
    # detect all sample case
    predcit_centroid_array = np.array([[10, 50], [200, 450], [500, 120], [130, 125]])
    ground_truth_array = np.array([[10, 50], [200, 450], [500, 120], [130, 125]])
    detection_threshold = 10
    true_positive, false_positive, false_negative, sample_number = eval_detection(
        predcit_centroid_array, ground_truth_array, detection_threshold
    )
    assert true_positive == 4
    assert false_positive == 0
    assert false_negative == 0
    assert sample_number == 4

    # not detect all sample case
    predcit_centroid_array = np.array([[10, 50], [200, 450], [500, 120], [130, 125]])
    ground_truth_array = np.array(
        [[1000, 5000], [2000, 4500], [5000, 1200], [1300, 1250], [1200, 1200]]
    )
    detection_threshold = 10
    true_positive, false_positive, false_negative, sample_number = eval_detection(
        predcit_centroid_array, ground_truth_array, detection_threshold
    )
    assert true_positive == 0
    assert false_positive == 4
    assert false_negative == 5
    assert sample_number == 5

    # detect half sample case
    predcit_centroid_array = np.array(
        [[10, 50], [200, 450], [500, 120], [130, 125], [200, 300]]
    )
    ground_truth_array = np.array([[13, 51], [2000, 4500], [510, 120], [1300, 1250]])
    detection_threshold = 10
    true_positive, false_positive, false_negative, sample_number = eval_detection(
        predcit_centroid_array, ground_truth_array, detection_threshold
    )
    assert true_positive == 2
    assert false_positive == 3
    assert false_negative == 2
    assert sample_number == 5
