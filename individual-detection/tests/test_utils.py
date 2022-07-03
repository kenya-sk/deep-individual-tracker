import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
from utils import (
    apply_masking_on_image,
    extract_local_data,
    get_directory_list,
    get_file_name_from_path,
    get_image_shape,
    get_masked_index,
    load_image,
    load_mask_image,
    load_sample,
    save_dataset_path,
)


class TestFileLoader(unittest.TestCase):
    def test_load_image(self):
        image_path = "./data/demo/demo_image.png"
        image = load_image(image_path, normalized=False)
        self.assertIs(type(image), np.ndarray)

        normalized_image = load_image(image_path, normalized=True)
        self.assertTrue(0.0 <= np.min(normalized_image))
        self.assertTrue(np.max(normalized_image) <= 1.0)

    def test_load_mask_image(self):
        mask_path = "./data/demo/demo_mask.png"
        mask = load_mask_image(mask_path, normalized=False)
        normalized_mask = load_mask_image(mask_path, normalized=True)
        none_mask = load_mask_image(None)

        self.assertIs(type(mask), np.ndarray)
        self.assertTrue(1 <= len(np.unique(normalized_mask)) <= 2)
        self.assertTrue(none_mask is None)

    def test_load_smple(self):
        x_path = "./data/demo/demo_image.png"
        y_path = "./data/demo/demo_label.npy"
        input_image_shape = (853, 1280, 3)
        mask_image = None
        X_image, y_dens = load_sample(
            x_path, y_path, input_image_shape, mask_image, is_rgb=True, normalized=False
        )

        self.assertIs(type(X_image), np.ndarray)
        self.assertIs(type(y_dens), np.ndarray)

        expected_X_channel = 3
        self.assertEqual(expected_X_channel, X_image.shape[2])
        self.assertEqual(2, len(y_dens.shape))  # expected 1 channel


class TestMaskImage(unittest.TestCase):
    def test_apply_masking_on_image(self):
        image_3channel = load_image("./data/demo/demo_image.png")
        image_1channel = image_3channel[:, :, 0]
        all_valid_mask = np.ones(image_3channel.shape)
        all_ignore_mask = np.zeros(image_3channel.shape)

        expected_all_valid__3channel_case = image_3channel.copy()
        expected_all_valid__1channel_case = image_1channel.copy()
        expected_all_ignore_3channel_case = np.zeros(image_3channel.shape)
        expected_all_ignore_1channel_case = np.zeros(image_1channel.shape)

        self.assertTrue(
            np.array_equal(
                expected_all_valid__3channel_case,
                apply_masking_on_image(image_3channel, all_valid_mask, channel=3),
            )
        )
        self.assertTrue(
            np.array_equal(
                expected_all_valid__1channel_case,
                apply_masking_on_image(image_1channel, all_valid_mask, channel=1),
            )
        )
        self.assertTrue(
            np.array_equal(
                expected_all_ignore_3channel_case,
                apply_masking_on_image(image_3channel, all_ignore_mask, channel=3),
            )
        )
        self.assertTrue(
            np.array_equal(
                expected_all_ignore_1channel_case,
                apply_masking_on_image(image_1channel, all_ignore_mask, channel=1),
            )
        )

    def test_get_masked_index(self):
        params_dict = {
            "image_height": 108,
            "image_width": 192,
            "analysis_image_height_min": 0,
            "analysis_image_height_max": 72,
            "analysis_image_width_min": 0,
            "analysis_image_width_max": 192,
        }

        mask_1 = None
        mask_2 = np.zeros((params_dict["image_height"], params_dict["image_width"]))
        mask_2[5:10, 5:10] = 1

        expected_1_index_h = []
        expected_1_index_w = []
        for h in range(
            params_dict["analysis_image_height_min"],
            params_dict["analysis_image_height_max"],
        ):
            for w in range(
                params_dict["analysis_image_width_min"],
                params_dict["analysis_image_width_max"],
            ):
                expected_1_index_h.append(h)
                expected_1_index_w.append(w)

        expected_2_index_h = []
        expected_2_index_w = []
        for h in range(5, 10):
            for w in range(5, 10):
                expected_2_index_h.append(h)
                expected_2_index_w.append(w)

        # horizontal flip case
        expected_3_index_h = []
        expected_3_index_w = []
        for h in range(5, 10):
            for w in range(5, 10):
                expected_3_index_h.append(h)
                expected_3_index_w.append(
                    params_dict["analysis_image_width_max"] - (w + 1)
                )

        actual_1_index_h, actual_1_index_w = get_masked_index(
            mask_1, params_dict, horizontal_flip=False
        )
        self.assertEqual(sorted(expected_1_index_h), sorted(actual_1_index_h))
        self.assertEqual(sorted(expected_1_index_w), sorted(actual_1_index_w))

        actual_2_index_h, actual_2_index_w = get_masked_index(
            mask_2, params_dict, horizontal_flip=False
        )
        self.assertEqual(sorted(expected_2_index_h), sorted(actual_2_index_h))
        self.assertEqual(sorted(expected_2_index_w), sorted(actual_2_index_w))

        # horizontal flip case
        actual_3_index_h, actual_3_index_w = get_masked_index(
            mask_2, params_dict, horizontal_flip=True
        )
        self.assertEqual(sorted(expected_3_index_h), sorted(actual_3_index_h))
        self.assertEqual(sorted(expected_3_index_w), sorted(actual_3_index_w))


class TestImageInfo(unittest.TestCase):
    def test_get_image_shape(self):
        image_path = "./data/demo/demo_image.png"
        image_3channel = load_image(image_path, normalized=False)
        image_1channel = image_3channel[:, :, 1]

        expected_3channel_shape = (853, 1280, 3)
        expected_1channel_shape = (853, 1280, 1)

        self.assertEqual(expected_3channel_shape, get_image_shape(image_3channel))
        self.assertEqual(expected_1channel_shape, get_image_shape(image_1channel))


class TestLocalImage(unittest.TestCase):
    def test_extract_local_data(self):
        params_dict = {
            "image_height": 108,
            "image_width": 192,
            "analysis_image_height_min": 0,
            "analysis_image_height_max": 72,
            "analysis_image_width_min": 0,
            "analysis_image_width_max": 192,
            "index_h": [10, 20, 30],
            "index_w": [15, 25, 35],
            "local_image_size": 32,
        }
        image = load_image("./data/demo/demo_image.png")
        density_map = np.load("./data/demo/demo_label.npy")
        local_image_list, local_density_list = extract_local_data(
            image, density_map, params_dict, is_flip=False
        )

        expected_local_data_num = 3
        self.assertEqual(expected_local_data_num, len(local_image_list))
        self.assertEqual(expected_local_data_num, len(local_density_list))

        expected_local_image_shape = (
            params_dict["local_image_size"],
            params_dict["local_image_size"],
            3,
        )
        self.assertEqual(expected_local_image_shape, local_image_list[0].shape)
        self.assertEqual("float32", local_image_list[0].dtype)


class TestDirectoryList(unittest.TestCase):
    def setUp(self):
        self.root_blank_directory = tempfile.mkdtemp()
        self.root_test_directory = tempfile.mkdtemp()
        os.makedirs(f"{self.root_test_directory}/test_1", exist_ok=True)
        os.makedirs(f"{self.root_test_directory}/test_2", exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.root_blank_directory)
        shutil.rmtree(self.root_test_directory)

    def test_get_directory_list(self):
        blank_expected = []
        self.assertEqual(blank_expected, get_directory_list(self.root_blank_directory))

        exist_expected = sorted(["test_1", "test_2"])
        self.assertEqual(
            exist_expected, sorted(get_directory_list(self.root_test_directory))
        )


class TestFileSaver(unittest.TestCase):
    def setUp(self):
        self.root_directory = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.root_directory)

    def test_save_dataset_path(self):
        X_path_list = ["test1.png", "test2.png", "test3.png"]
        y_path_list = ["test1.npy", "test2.npy", "test3.npy"]
        save_path = f"{self.root_directory}/test.csv"
        save_dataset_path(X_path_list, y_path_list, save_path)

        self.assertTrue(os.path.isfile(save_path))


class TestFileName(unittest.TestCase):
    def test_get_frame_number_from_path(self):
        path = "./demo/image/20170416_903321.png"
        expected = "20170416_903321"
        self.assertEqual(expected, get_file_name_from_path(path))


if __name__ == "__main__":
    unittest.main()
