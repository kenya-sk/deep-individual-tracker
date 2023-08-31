import random
import re
import time
from pathlib import Path

from detector.utils import (
    get_directory_list,
    get_elapsed_time_str,
    get_file_name_from_path,
)


def test_get_elapsed_time_str() -> None:
    # chack specific time
    now = time.time()
    elapsed_str = get_elapsed_time_str(now - 5405)  # 1hour 30min 5sec
    assert elapsed_str == "1[hour] 30[min] 5[sec]"

    # check value range
    now = int(time.time())
    elapsed_str = get_elapsed_time_str(random.uniform(now // 1000, now))
    assert re.match(
        r"\d+\[hour\] [1-9]|[1-5][0-9]|60\[min\] [1-9]|[1-5][0-9]|60\[sec\]",
        elapsed_str,
    )


def test_get_directory_list(tmp_path: Path) -> None:
    (tmp_path / "test.txt").touch()
    (tmp_path / "coord").mkdir()
    (tmp_path / "dens").mkdir()
    directory_list = get_directory_list(tmp_path)
    expected = ["coord", "dens"]
    assert sorted(directory_list) == sorted(expected)


def test_get_file_name_from_path() -> None:
    test_path = Path("/home/data/20170416_20111.png")
    file_name = get_file_name_from_path(test_path)
    assert file_name == "20170416_20111"

    test_path = Path("/home/data/20170416/20111.png")
    file_name = get_file_name_from_path(test_path)
    assert file_name == "20111"
