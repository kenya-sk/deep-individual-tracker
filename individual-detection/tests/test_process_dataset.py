import pytest
from detector.exceptions import DatasetEmptyError
from detector.process_dataset import (
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
    assert image_paths.sort() == expected_image_paths.sort()
    assert image_paths.sort() == expected_label_paths.sort()


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
    assert image_paths.sort() == expected_image_paths.sort()
    assert image_paths.sort() == expected_label_paths.sort()


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
