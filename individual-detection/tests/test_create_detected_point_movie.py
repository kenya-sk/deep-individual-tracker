from detector.create_detected_point_movie import sort_by_frame_number


def test_sort_by_frame_number():
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
