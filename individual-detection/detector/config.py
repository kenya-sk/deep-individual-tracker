from pathlib import Path
from typing import Dict, List, Type, TypeVar, Union

import yaml
from pydantic.dataclasses import dataclass

T = TypeVar("T", bound="BaseConfig")


@dataclass(frozen=True)
class BaseConfig:
    pass


@dataclass(frozen=True)
class TrainConfig(BaseConfig):
    image_directory: Path
    density_directory: Path
    mask_path: Path
    pretrained_model_path: Path
    save_dataset_directory: Path
    tensorboard_directory: Path
    save_trained_model_directory: Path
    dataset_split_type: str
    train_date_list: List[str]
    validation_date_list: List[str]
    test_date_list: List[str]
    test_size: float
    batch_size: int
    n_epochs: int
    min_epochs: int
    early_stopping_patience: int
    flip_prob: float
    under_sampling_threshold: float
    dropout_rate: float
    hard_negative_mining_weight: float


@dataclass(frozen=True)
class PredictConfig(BaseConfig):
    target_date: str
    trained_model_directory: Path
    image_directory: Path
    video_directory: Path
    output_directory: Path
    mask_path: Path
    predict_data_type: str
    index_extract_type: str
    skip_pixel_interval: int
    start_hour: int
    end_hour: int
    predict_interval: int
    predict_batch_size: int
    is_saved_map: bool
    band_width: int
    cluster_threshold: int


@dataclass(frozen=True)
class EvaluateConfig(BaseConfig):
    predict_directory: Path
    ground_truth_directory: Path
    mask_path: Path
    detection_threshold: int


@dataclass(frozen=True)
class SearchParameterConfig(BaseConfig):
    dataset_path: Path
    mask_path: Path
    save_directory: Path
    trained_model_directory: Path
    cols_order: List[str]
    search_params: Dict[str, List[Union[int, float]]]
    sample_rate: float
    predict_batch_size: int
    band_width: int
    index_extract_type: str
    fixed_cluster_threshold: float
    detection_threshold: int


@dataclass(frozen=True)
class DetectedPointMovieConfig(BaseConfig):
    image_directory: Path
    point_coord_directory: Path
    movie_save_path: Path


def load_config(config_path: Path, config_class: Type[T]) -> T:
    """Load config file.
    Config class created by pydantic.dataclasses.dataclass from yaml file.
    Config file path defined in detector/constants.py.

    Args:
        config_path (Path): config file path
        config_class (Type[T]): config class

    Returns:
        T: dataclass of config
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config_class(**config)
