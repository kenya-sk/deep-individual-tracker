import pytest
from detector.exceptions import IndexExtractionError
from detector.predict import extract_prediction_indices


def test_extract_prediction_indices():
    # example = 3x5 shape
    height_index_list = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3]
    width_index_list = [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5]

    # skip_pixel_interval = 0 case
    skip_pixel_interval = 0
    actual_list_1 = extract_prediction_indices(
        height_index_list, width_index_list, skip_pixel_interval
    )
    expected_list_1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    assert actual_list_1 == expected_list_1

    # index_extract_type = grid case
    actual_list_2 = extract_prediction_indices(
        height_index_list,
        width_index_list,
        skip_pixel_interval=2,
        index_extract_type="grid",
    )
    expected_list_2 = [1, 3, 5, 6, 7, 8, 9, 11, 13]
    assert actual_list_2 == expected_list_2

    # index_extract_type = intesect case
    actual_list_3 = extract_prediction_indices(
        height_index_list,
        width_index_list,
        skip_pixel_interval=2,
        index_extract_type="intersect",
    )
    expected_list_3 = [6, 8]
    assert actual_list_3 == expected_list_3

    # index_extract_type = vertical case
    actual_list_4 = extract_prediction_indices(
        height_index_list,
        width_index_list,
        skip_pixel_interval=2,
        index_extract_type="vertical",
    )
    expected_list_4 = [1, 3, 6, 8, 11, 13]
    assert actual_list_4 == expected_list_4

    # index_extract_type = horizontal case
    actual_list_5 = extract_prediction_indices(
        height_index_list,
        width_index_list,
        skip_pixel_interval=2,
        index_extract_type="horizontal",
    )
    expected_list_5 = [5, 6, 7, 8, 9]
    assert actual_list_5 == expected_list_5

    # invalid index_extract_type case
    with pytest.raises(IndexExtractionError):
        _ = extract_prediction_indices(
            height_index_list,
            width_index_list,
            skip_pixel_interval=2,
            index_extract_type="invalid",
        )
