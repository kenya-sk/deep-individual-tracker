from pathlib import Path

import yaml
from pydantic.dataclasses import dataclass


@dataclass(frozen=True)
class AnimationConfig:
    format: str
    frame_number: int
    interval: int
    density_interval: int
    display_dot: bool
    display_density: bool


@dataclass(frozen=True)
class HistogramConfig:
    x_bin_granularity: int
    y_bin_granularity: int


@dataclass(frozen=True)
class StatisticsConfig:
    mean_max: int
    acceleration_max: int


@dataclass(frozen=True)
class PathConfig:
    frame_directory: Path
    coordinate_directory: Path
    past_coordinate_directory: Path
    mean_speed_path: Path
    past_mean_speed_path: Path
    acceleration_count_path: Path
    past_acceleration_count_path: Path
    save_movie_path: Path


@dataclass(frozen=True)
class MonitoringConfig:
    animation: AnimationConfig
    histogram: HistogramConfig
    statistics: StatisticsConfig
    path: PathConfig


def load_config(config_path: Path) -> MonitoringConfig:
    """Load monitoring config file.
    Config class created by pydantic.dataclasses.dataclass from yaml file.
    Config file path defined in monitoring/constants.py.

    Args:
        config_path (Path): config file path of monitoring

    Returns:
        MonitoringConfig: config of monitoring
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return MonitoringConfig(**config)
