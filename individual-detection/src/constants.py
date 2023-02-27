import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

MODULE_HOME: Path = Path(__file__).resolve().parents[1]
MODUEL_SRC: Path = Path(__file__).resolve().parents[0]

CONFIG_DIR: Path = MODULE_HOME / "config"
LOG_DIR: Path = MODULE_HOME / "logs"
DATA_DIR: Path = MODULE_HOME / "data"

TODAY: str = datetime.today().strftime("%Y%m%d")
JST = timezone(timedelta(hours=+9), "JST")
EXECUTION_TIME: str = datetime.now(JST).strftime("%Y%m%d_%H%M%S")
LOGGER_NAME = "stats-monitoring"

TRAIN_CONFIG_NAME: str = "train"
PREDICT_CONFIG_NAME: str = "predict"
EVALUATE_CONFIG_NAME: str = "evaluate"
SEARCH_PARAMETER_CONFIG_NAME: str = "search_parameter"
DETECTED_MOVIE_CONFIG_NAME: str = "detected_point_movie"

# input video information
MOVIE_FPS: int = 30
IMAGE_EXTENTION: str = ".png"
FRAME_HEIGHT: int = 1080
FRAME_WIDTH: int = 1920
FRAME_CHANNEL: int = 3

# analysis information
ANALYSIS_HEIGHT_MIN: int = 0
ANALYSIS_HEIGHT_MAX: int = 720
ANALYSIS_WIDTH_MIN: int = 0
ANALYSIS_WIDTH_MAX: int = 1920
# square local image size: > 0
LOCAL_IMAGE_SIZE: int = 72

# ID of using GPU: 0-max number of available GPUs
GPU_DEVICE_ID: str = "0,1,2"
# using each GPU memory rate: 0.0-1.0
GPU_MEMORY_RATE: float = 0.9

RANDOM_SEED: int = 42
FLOAT_MAX: float = sys.float_info.max
