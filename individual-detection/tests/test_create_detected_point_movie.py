import numpy as np
from detector.constants import POINT_COLOR
from detector.create_detected_point_movie import (
    draw_detection_points,
    sort_by_frame_number,
)


def test_sort_by_frame_number() -> None:
    path_list = [
        "./example/2022_03_19_234224.png",
        "./example/2022_03_19_234225.png",
        "./example/2022_03_19_234223.png",
    ]
    sorted_path_list = sort_by_frame_number(path_list)
    expected = [
        "./example/2022_03_19_234223.png",
        "./example/2022_03_19_234224.png",
        "./example/2022_03_19_234225.png",
    ]
    assert sorted_path_list == expected


def test_draw_detection_points() -> None:
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    # single point case
    point_coord = np.array([1, 1])
    drew_image = draw_detection_points(image, point_coord)
    assert image.shape == drew_image.shape
    assert tuple(drew_image[1, 1, :]) == POINT_COLOR

    # multiple points case
    point_coord = np.array([[1, 1], [2, 2], [5, 5]])
    drew_image = draw_detection_points(image, point_coord)
    assert image.shape == drew_image.shape
    assert tuple(drew_image[1, 1, :]) == POINT_COLOR
    assert tuple(drew_image[2, 2, :]) == POINT_COLOR
    assert tuple(drew_image[5, 5, :]) == POINT_COLOR
