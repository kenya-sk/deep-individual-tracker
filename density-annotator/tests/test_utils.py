from pathlib import Path

import numpy as np
import pytest
from annotator.exceptions import PathNotExistError
from annotator.utils import (
    get_full_path_list,
    get_input_data_type,
    get_path_list,
    save_coordinate,
    save_density_map,
    save_image,
)


def test_get_path_list(tmp_path: Path) -> None:
    # file case
    test_file = "test_1.txt"
    test_file_path = tmp_path / test_file
    test_file_path.touch()
    path_list_1 = get_path_list(tmp_path, test_file)
    expected_1 = [test_file_path]
    assert path_list_1 == expected_1

    # directory case
    test_dir = "test_dir"
    test_dir_path = tmp_path / test_dir
    test_dir_path.mkdir()
    test_file_2 = "test_2.txt"
    test_file_2_path = test_dir_path / test_file_2
    test_file_2_path.touch()
    test_file_3 = "test_3.txt"
    test_file_3_path = test_dir_path / test_file_3
    test_file_3_path.touch()
    path_list_2 = get_path_list(tmp_path, test_dir)
    expected_2 = [test_file_2_path, test_file_3_path]
    assert path_list_2.sort() == expected_2.sort()

    # not exist case
    not_exist_path = str(tmp_path / "not_exist")
    with pytest.raises(PathNotExistError):
        _ = get_path_list(tmp_path, not_exist_path)


def test_full_path_list() -> None:
    current_working_dirc = Path("/home/test")
    relative_path_list = ["test_1.txt", "test_2.txt"]
    full_path_list = get_full_path_list(current_working_dirc, relative_path_list)
    expected = ["/home/test/test_1.txt", "/home/test/test_2.txt"]
    assert full_path_list.sort() == expected.sort()


def test_get_input_data_type(tmp_path: Path) -> None:
    # image case
    image_file = "test.png"
    image_path = tmp_path / image_file
    image_path.touch()
    assert get_input_data_type(str(image_path)) == "image"

    # video case
    video_file = "test.mp4"
    video_path = tmp_path / video_file
    video_path.touch()
    assert get_input_data_type(str(video_path)) == "video"

    # invalid case
    invalid_path = "/home/test.txt"
    assert get_input_data_type(invalid_path) == "invalid"


def test_save_image(tmp_path: Path) -> None:
    image = np.zeros((10, 10), dtype=np.uint8)
    image_path = tmp_path / "test.png"
    save_image(str(image_path), image)
    assert image_path.is_file()


def test_save_coordinate(tmp_path: Path) -> None:
    coordinate = np.zeros((10, 2), dtype=np.uint8)
    coordinate_path = tmp_path / "test.npy"
    save_coordinate(str(coordinate_path), coordinate)
    assert coordinate_path.is_file()


def test_save_density_map(tmp_path: Path) -> None:
    density_map = np.zeros((10, 10), dtype=np.uint8)
    density_map_path = tmp_path / "test.npy"
    save_density_map(str(density_map_path), density_map)
    assert density_map_path.is_file()
