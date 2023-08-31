from pathlib import Path
from typing import Tuple

import pytest

from detector.exceptions import DatasetEmptyError
from detector.process_dataset import (
    load_dataset,
    load_multi_date_datasets,
    save_dataset_path,
    split_dataset,
    split_dataset_by_date,
)

SetupFixture = Tuple[Path, Path, Path]


@pytest.fixture(scope="function")
def setup_folder(tmp_path: Path) -> SetupFixture:
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


def test_load_dataset(setup_folder: SetupFixture) -> None:
    # empty folder case
    with pytest.raises(DatasetEmptyError):
        _ = load_dataset(
            Path("/home/not_exist/image"), Path("/home/not_exist/label"), "*.png"
        )

    # normal case
    _, root_image_dirc, root_label_dirc = setup_folder
    target_date = "2023-01-01"
    image_paths, label_paths = load_dataset(
        root_image_dirc / target_date, root_label_dirc / target_date, "*.png"
    )
    expected_image_paths = [
        root_image_dirc / "2023-01-01" / "20230101_1.png",
        root_image_dirc / "2023-01-01" / "20230101_2.png",
    ]
    expected_label_paths = [
        root_label_dirc / "2023-01-01" / "20230101_1.npy",
        root_label_dirc / "2023-01-01" / "20230101_2.npy",
    ]
    assert len(image_paths) == 2
    assert len(label_paths) == 2
    assert sorted(image_paths) == sorted(expected_image_paths)
    assert sorted(label_paths) == sorted(expected_label_paths)


def test_load_multi_date_datasets(setup_folder: SetupFixture) -> None:
    _, root_image_dirc, root_label_dirc = setup_folder
    date_list = ["20230301", "20230302"]
    image_paths, label_paths = load_multi_date_datasets(
        root_image_dirc / "multi", root_label_dirc / "multi", date_list
    )
    expected_image_paths = [
        root_image_dirc / "multi" / "20230301_1.png",
        root_image_dirc / "multi" / "20230302_1.png",
    ]
    expected_label_paths = [
        root_label_dirc / "multi" / "20230301_1.npy",
        root_label_dirc / "multi" / "20230302_1.npy",
    ]
    assert len(image_paths) == 2
    assert len(label_paths) == 2
    assert sorted(image_paths) == sorted(expected_image_paths)
    assert sorted(label_paths) == sorted(expected_label_paths)


def test_save_dataset_path(tmp_path: Path) -> None:
    save_path = tmp_path / "dataset.csv"
    X_path_list = [Path("20230101_1.png"), Path("20230101_2.png")]
    y_path_list = [Path("20230101_1.npy"), Path("20230101_2.npy")]
    save_dataset_path(X_path_list, y_path_list, save_path)
    assert save_path.exists()


def test_split_dataset(tmp_path: Path) -> None:
    X_path_list = [Path(f"20230101_{i}.png") for i in range(10)]
    y_path_list = [Path(f"20230101_{i}.npy") for i in range(10)]
    X_train, X_valid, X_test, y_train, y_valid, y_test = split_dataset(
        X_path_list, y_path_list, 0.2, tmp_path
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


def test_split_dataset_by_date(setup_folder: SetupFixture) -> None:
    root_dirc, root_image_dirc, root_label_dirc = setup_folder
    train_date_list = ["20230301", "20230302", "20230303"]
    valid_date_list = ["20230304"]
    test_date_list = ["20230305"]
    X_train, X_valid, X_test, y_train, y_valid, y_test = split_dataset_by_date(
        root_image_dirc / "multi",
        root_label_dirc / "multi",
        train_date_list,
        valid_date_list,
        test_date_list,
        root_dirc,
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
