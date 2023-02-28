import pytest
from annotator.exceptions import PathNotExistError
from annotator.utils import get_path_list


def test_get_path_list(tmp_path):
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
    not_exist_path = tmp_path / "not_exist"
    with pytest.raises(PathNotExistError):
        _ = get_path_list(tmp_path, not_exist_path)
