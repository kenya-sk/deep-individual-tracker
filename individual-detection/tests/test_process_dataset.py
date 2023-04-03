import numpy as np
import pytest
from detector.exceptions import DatasetEmptyError
from detector.process_dataset import (
    get_masked_index,
    load_dataset,
    load_multi_date_datasets,
    save_dataset_path,
    split_dataset,
    split_dataset_by_date,
)


@pytest.fixture(scope="function")
def setup_folder(tmp_path):
    # Create image and label folder
    root_image_dirc = tmp_path / "image"
    root_label_dirc = tmp_path / "label"
    root_image_dirc.mkdir()
    root_label_dirc.mkdir()

    # Create date folders
    (root_image_dirc / "2023-01-01").mkdir()
    (root_image_dirc / "2023-01-02").mkdir()
    (root_image_dirc / "multi").mkdir()
    (root_label_dirc / "2023-01-01").mkdir()
    (root_label_dirc / "2023-01-02").mkdir()
    (root_label_dirc / "multi").mkdir()

    # add 5 dummy image and label files
    (root_image_dirc / "2023-01-01" / "20230101_1.png").touch()
    (root_image_dirc / "2023-01-01" / "20230101_2.png").touch()
    (root_image_dirc / "2023-01-02" / "20230102_1.png").touch()
    (root_image_dirc / "2023-01-02" / "20230102_2.png").touch()
    (root_image_dirc / "2023-01-02" / "20230102_3.png").touch()
    (root_image_dirc / "multi" / "20230301_1.png").touch()
    (root_image_dirc / "multi" / "20230302_1.png").touch()
    (root_image_dirc / "multi" / "20230303_1.png").touch()
    (root_image_dirc / "multi" / "20230304_1.png").touch()
    (root_image_dirc / "multi" / "20230305_1.png").touch()
    (root_label_dirc / "2023-01-01" / "20230101_1.npy").touch()
    (root_label_dirc / "2023-01-01" / "20230101_2.npy").touch()
    (root_label_dirc / "2023-01-02" / "20230102_1.npy").touch()
    (root_label_dirc / "2023-01-02" / "20230102_2.npy").touch()
    (root_label_dirc / "2023-01-02" / "20230102_3.npy").touch()
    (root_label_dirc / "multi" / "20230301_1.png").touch()
    (root_label_dirc / "multi" / "20230302_1.png").touch()
    (root_label_dirc / "multi" / "20230303_1.png").touch()
    (root_label_dirc / "multi" / "20230304_1.png").touch()
    (root_label_dirc / "multi" / "20230305_1.png").touch()

    return (tmp_path, root_image_dirc, root_label_dirc)


def test_load_dataset(setup_folder):
    # empty folder case
    with pytest.raises(DatasetEmptyError):
        _ = load_dataset("/home/not_exist/image", "/home/not_exist/label")

    # normal case
    _, root_image_dirc, root_label_dirc = setup_folder
    target_date = "2023-01-01"
    image_paths, label_paths = load_dataset(
        str(root_image_dirc / target_date), str(root_label_dirc / target_date)
    )
    expected_image_paths = [
        str(root_image_dirc / "2023-01-01" / "20230101_1.png"),
        str(root_image_dirc / "2023-01-01" / "20230101_2.png"),
    ]
    expected_label_paths = [
        str(root_label_dirc / "2023-01-01" / "20230101_1.npy"),
        str(root_label_dirc / "2023-01-01" / "20230101_2.npy"),
    ]
    assert len(image_paths) == 2
    assert len(label_paths) == 2
    assert sorted(image_paths) == sorted(expected_image_paths)
    assert sorted(label_paths) == sorted(expected_label_paths)


def test_load_multi_date_datasets(setup_folder):
    _, root_image_dirc, root_label_dirc = setup_folder
    date_list = ["20230301", "20230302"]
    image_paths, label_paths = load_multi_date_datasets(
        str(root_image_dirc / "multi"), str(root_label_dirc / "multi"), date_list
    )
    expected_image_paths = [
        str(root_image_dirc / "multi" / "20230301_1.png"),
        str(root_image_dirc / "multi" / "20230302_1.png"),
    ]
    expected_label_paths = [
        str(root_label_dirc / "multi" / "20230301_1.npy"),
        str(root_label_dirc / "multi" / "20230302_1.npy"),
    ]
    assert len(image_paths) == 2
    assert len(label_paths) == 2
    assert sorted(image_paths) == sorted(expected_image_paths)
    assert sorted(label_paths) == sorted(expected_label_paths)


def test_save_dataset_path(tmp_path):
    save_path = tmp_path / "dataset.csv"
    X_path_list = ["20230101_1.png", "20230101_2.png"]
    y_path_list = ["20230101_1.npy", "20230101_2.npy"]
    save_dataset_path(X_path_list, y_path_list, str(save_path))
    assert save_path.exists()


def test_split_dataset(tmp_path):
    X_path_list = [f"20230101_{i}.png" for i in range(10)]
    y_path_list = [f"20230101_{i}.npy" for i in range(10)]
    X_train, X_valid, X_test, y_train, y_valid, y_test = split_dataset(
        X_path_list, y_path_list, 0.2, str(tmp_path)
    )
    assert len(X_train) == 8
    assert len(X_valid) == 1
    assert len(X_test) == 1
    assert len(y_train) == 8
    assert len(y_valid) == 1
    assert len(y_test) == 1
    assert (tmp_path / "train_dataset.csv").exists()
    assert (tmp_path / "valid_dataset.csv").exists()
    assert (tmp_path / "test_dataset.csv").exists()


def test_split_dataset_by_date(setup_folder):
    root_dirc, root_image_dirc, root_label_dirc = setup_folder
    train_date_list = ["20230301", "20230302", "20230303"]
    valid_date_list = ["20230304"]
    test_date_list = ["20230305"]
    X_train, X_valid, X_test, y_train, y_valid, y_test = split_dataset_by_date(
        str(root_image_dirc / "multi"),
        str(root_label_dirc / "multi"),
        train_date_list,
        valid_date_list,
        test_date_list,
        str(root_dirc),
    )
    assert len(X_train) == 3
    assert len(X_valid) == 1
    assert len(X_test) == 1
    assert len(y_train) == 3
    assert len(y_valid) == 1
    assert len(y_test) == 1
    assert (root_dirc / "train_dataset.csv").exists()
    assert (root_dirc / "valid_dataset.csv").exists()
    assert (root_dirc / "test_dataset.csv").exists()


def test_get_masked_index():
    # create binary mask image (0: masked, 1: unmasked)
    # unmasked left top area
    mask = np.zeros((10, 10))
    mask[0:5, 0:5] = 1

    # normal case
    masked_index = get_masked_index(mask, horizontal_flip=False)
    expected_masked_index = [
        (0, 0),
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 4),
        (1, 0),
        (1, 1),
        (1, 2),
        (1, 3),
        (1, 4),
        (2, 0),
        (2, 1),
        (2, 2),
        (2, 3),
        (2, 4),
        (3, 0),
        (3, 1),
        (3, 2),
        (3, 3),
        (3, 4),
        (4, 0),
        (4, 1),
        (4, 2),
        (4, 3),
        (4, 4),
    ]
    assert sorted(masked_index) == sorted(expected_masked_index)

    # horizontal flip case
    masked_index = get_masked_index(mask, horizontal_flip=True)
    expected_masked_index = [
        (0, 5),
        (0, 6),
        (0, 7),
        (0, 8),
        (0, 9),
        (1, 5),
        (1, 6),
        (1, 7),
        (1, 8),
        (1, 9),
        (2, 5),
        (2, 6),
        (2, 7),
        (2, 8),
        (2, 9),
        (3, 5),
        (3, 6),
        (3, 7),
        (3, 8),
        (3, 9),
        (4, 5),
        (4, 6),
        (4, 7),
        (4, 8),
        (4, 9),
    ]
    assert sorted(masked_index) == sorted(expected_masked_index)
