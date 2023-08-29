from pathlib import Path
from typing import List

import yaml
from pydantic.dataclasses import dataclass


@dataclass(frozen=True)
class AnnotatorPathConfig:
    input_file_path: Path
    save_raw_image_dir: Path
    save_annotated_dir: Path


@dataclass(frozen=True)
class AnnotatorConfig:
    sigma_pow: int
    mouse_event_interval: int
    path: AnnotatorPathConfig


@dataclass(frozen=True)
class SamplerPathConfig:
    input_video_list: List[Path]
    save_frame_dirc: Path


@dataclass(frozen=True)
class SamplerConfig:
    sampling_type: str
    sample_rate: int
    path: SamplerPathConfig


def load_annotator_config(config_path: Path) -> AnnotatorConfig:
    """Load annotator config file.
    Config class created by pydantic.dataclasses.dataclass from yaml file.
    Config file path defined in annotator/constants.py.

    Args:
        config_path (Path): config file path of annotator

    Returns:
        SamplerConfig: config of annotator
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return AnnotatorConfig(**config)


def load_sampler_config(config_path: Path) -> SamplerConfig:
    """Load sampler config file.
    Config class created by pydantic.dataclasses.dataclass from yaml file.
    Config file path defined in annotator/constants.py.

    Args:
        config_path (Path): config file path of sampler

    Returns:
        SamplerConfig: config of sampler
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return SamplerConfig(**config)
