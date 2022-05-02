import cv2
import numpy as np
import unittest

from utils import (
    get_path_list,
    get_full_path_list,
    get_input_data_type,
    load_image,
    load_video,
)


class TestPathList(unittest.TestCase):
    def test_get_path_list_file_case(self):
        exist_file_name = "demo.png"
        working_directory = "../data/demo/inputs"
        expected = ["../data/demo/inputs/demo.png"]
        actual = get_path_list(exist_file_name, working_directory)
        self.assertEqual(expected, actual, "It is not a list of expected files.")

    def test_get_path_list_directory_case(self):
        exist_directory_name = "inputs"
        working_directory = "../data/demo"
        expected = ["../data/demo/inputs/demo.png", "../data/demo/inputs/demo2.jpg"]
        actual = get_path_list(exist_directory_name, working_directory)
        # Change the order of elements to ensure a match
        expected = expected.sort()
        actual = actual.sort()
        self.assertEqual(expected, actual, "It is not a list of expected files.")

    def test_get_path_list_error_case(self):
        not_exist_file_name = "not_exist_file.png"
        working_directory = "../data/demo/inputs"
        with self.assertRaises(SystemExit):
            get_path_list(not_exist_file_name, working_directory)

    def test_full_path_list(self):
        current_working_dirc = "/home/ubuntu/demo"
        relative_path_list = [
            "./data/demo1.mp4",
            "./data/demo2.mp4",
            "./data/demo3.mp4",
        ]
        expected = [
            "/home/ubuntu/demo/data/demo1.mp4",
            "/home/ubuntu/demo/data/demo2.mp4",
            "/home/ubuntu/demo/data/demo3.mp4",
        ]
        actual = get_full_path_list(current_working_dirc, relative_path_list)
        # Change the order of elements to ensure a match
        expected = expected.sort()
        actual = actual.sort()
        self.assertEqual(expected, actual)


class TestFileType(unittest.TestCase):
    def test_get_input_data_type_image_case(self):
        png_image_path = "../data/demo/inputs/demo.png"
        jpg_image_path = "../data/demo/inputs/demo2.jpg"
        expected = "image"
        self.assertEqual(expected, get_input_data_type(png_image_path))
        self.assertEqual(expected, get_input_data_type(jpg_image_path))

    def test_get_input_data_type_video_case(self):
        video_path = "../data/demo/video/demo_video.mp4"
        expected = "video"
        self.assertEqual(expected, get_input_data_type(video_path))


class TestFileLoader(unittest.TestCase):
    def test_load_image(self):
        image_path = "../data/demo/inputs/demo.png"
        image = load_image(image_path)
        self.assertIs(type(image), np.ndarray)

    def test_load_video(self):
        video_path = "../data/demo/video/demo_video.mp4"
        video = load_video(video_path)
        self.assertIs(type(video), cv2.VideoCapture)


if __name__ == "__main__":
    unittest.main()
