from unittest.mock import Mock, patch

import numpy as np
import pytest
from annotator.config import AnnotatorConfig, load_config
from annotator.constants import ANNOTATOR_TEST_CONFIG_NAME, CONFIG_DIR, DATA_DIR
from annotator.density_annotator import DensityAnnotator


@pytest.fixture(scope="function")
def annotator_config() -> AnnotatorConfig:
    return load_config(CONFIG_DIR / ANNOTATOR_TEST_CONFIG_NAME, AnnotatorConfig)


@patch("cv2.namedWindow")
@patch("cv2.setMouseCallback")
def test_init_setting(
    window_mock: Mock, mouse_mock: Mock, annotator_config: AnnotatorConfig
) -> None:
    annotator = DensityAnnotator(annotator_config)
    assert annotator.sigma_pow == annotator_config.sigma_pow
    assert annotator.mouse_event_interval == annotator_config.mouse_event_interval
    assert len(annotator.input_file_path_list) == 1
    assert (
        annotator.save_raw_image_dir
        == DATA_DIR / annotator_config.path.save_raw_image_dir
    )
    assert (
        annotator.save_annotated_dir
        == DATA_DIR / annotator_config.path.save_annotated_dir
    )
    assert annotator.save_annotated_image_dir == str(
        DATA_DIR / annotator_config.path.save_annotated_dir / "image"
    )
    assert annotator.save_annotated_coord_dir == str(
        DATA_DIR / annotator_config.path.save_annotated_dir / "coord"
    )
    assert annotator.save_annotated_density_dir == str(
        DATA_DIR / annotator_config.path.save_annotated_dir / "dens"
    )


@patch("cv2.namedWindow")
@patch("cv2.setMouseCallback")
def test_annotator_initialization(
    window_mock: Mock, mouse_mock: Mock, annotator_config: AnnotatorConfig
) -> None:
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    annotator = DensityAnnotator(annotator_config)
    annotator.frame = frame
    annotator.annotator_initialization()

    assert annotator.width == frame.shape[1]
    assert annotator.height == frame.shape[0]

    assert annotator.coordinate_matrix.shape == (frame.shape[1], frame.shape[0], 2)
    assert list(annotator.coordinate_matrix[10][20]) == [10, 20]


@patch("cv2.namedWindow")
@patch("cv2.setMouseCallback")
def test_add_point(
    window_mock: Mock, mouse_mock: Mock, annotator_config: AnnotatorConfig
) -> None:
    annotator = DensityAnnotator(annotator_config)
    annotator.frame = np.zeros((100, 200, 3), dtype=np.uint8)
    annotator.frame_list = []
    annotator.features = np.array([], np.uint16)

    annotator.add_point(10, 20)
    annotator.add_point(100, 15)
    annotator.add_point(50, 30)
    assert annotator.features.shape == (3, 2)
    assert len(annotator.frame_list) == 3


@patch("cv2.namedWindow")
@patch("cv2.setMouseCallback")
@patch("cv2.imshow")
def test_delete_point(
    window_mock: Mock,
    mouse_mock: Mock,
    show_mock: Mock,
    annotator_config: AnnotatorConfig,
) -> None:
    annotator = DensityAnnotator(annotator_config)
    annotator.frame = np.zeros((100, 200, 3), dtype=np.uint8)
    annotator.frame_list = []
    annotator.features = np.array([], np.uint16)

    annotator.add_point(10, 20)
    annotator.add_point(100, 15)
    annotator.add_point(50, 30)
    annotator.delete_point()
    assert annotator.features.shape == (2, 2)
    assert list(annotator.features[0]) == [10, 20]
    assert list(annotator.features[1]) == [100, 15]
    assert len(annotator.frame_list) == 2
