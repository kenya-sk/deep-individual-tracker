import numpy as np
from detector.constants import TEST_DATA_DIR
from detector.evaluate import eval_detection, get_ground_truth


def test_get_ground_truth():
    # mask_image is None case
    ground_truth_path = str(TEST_DATA_DIR / "ground_truth.csv")
    ground_truth_array = get_ground_truth(ground_truth_path, None)
    assert ground_truth_array.shape == (4, 2)
    assert ground_truth_array.dtype == np.int32
    assert ground_truth_array[0][0] == 10
    assert ground_truth_array[0][1] == 50
    assert ground_truth_array[1][0] == 200
    assert ground_truth_array[1][1] == 450
    assert ground_truth_array[2][0] == 500
    assert ground_truth_array[2][1] == 120
    assert ground_truth_array[3][0] == 130
    assert ground_truth_array[3][1] == 125

    # mask_image is not None case
    mask_image = np.ones((256, 256))
    ground_truth_array = get_ground_truth(ground_truth_path, mask_image)
    assert ground_truth_array.shape == (2, 2)
    assert ground_truth_array.dtype == np.int32
    assert ground_truth_array[0][0] == 10
    assert ground_truth_array[0][1] == 50
    assert ground_truth_array[1][0] == 130
    assert ground_truth_array[1][1] == 125


def test_eval_detection():
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
