from pathlib import Path
from typing import List, Type, TypeVar

import yaml
from pydantic.dataclasses import dataclass

T = TypeVar("T", bound="BaseConfig")


@dataclass(frozen=True)
class BaseConfig:
    pass


@dataclass(frozen=True)
class AnnotatorPathConfig:
    input_file_path: Path
    save_raw_image_dir: Path
    save_annotated_dir: Path


@dataclass(frozen=True)
class AnnotatorConfig(BaseConfig):
    sigma_pow: int
    mouse_event_interval: int
    path: AnnotatorPathConfig


@dataclass(frozen=True)
class SamplerPathConfig:
    input_video_list: List[Path]
    save_frame_dirc: Path


@dataclass(frozen=True)
class SamplerConfig(BaseConfig):
    sampling_type: str
    sample_rate: int
    path: SamplerPathConfig


def load_config(config_path: Path, config_class: Type[T]) -> T:
    """Load config file.
    Config class created by pydantic.dataclasses.dataclass from yaml file.
    Config file path defined in annotator/constants.py.

    Args:
        config_path (Path): config file path
        config_class (Type[T]): config class

    Returns:
        T: dataclass of config
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config_class(**config)
