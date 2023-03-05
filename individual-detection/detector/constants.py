import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2

MODULE_HOME: Path = Path(__file__).resolve().parents[1]
MODUEL_SRC: Path = Path(__file__).resolve().parents[0]

CONFIG_DIR: Path = MODULE_HOME / "config"
LOG_DIR: Path = MODULE_HOME / "logs"
DATA_DIR: Path = MODULE_HOME / "data"
TEST_DATA_DIR: Path = DATA_DIR / "test"

TODAY: str = datetime.today().strftime("%Y%m%d")
JST = timezone(timedelta(hours=+9), "JST")
EXECUTION_TIME: str = datetime.now(JST).strftime("%Y%m%d_%H%M%S")
LOGGER_NAME = "individual-detection"

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
INPUT_IMAGE_SHAPE: tuple = (FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNEL)

# analysis information
ANALYSIS_HEIGHT_MIN: int = 0
ANALYSIS_HEIGHT_MAX: int = 720
ANALYSIS_WIDTH_MIN: int = 0
ANALYSIS_WIDTH_MAX: int = 1920
# square local image size: > 0
LOCAL_IMAGE_SIZE: int = 72

# ID of using GPU: 0-max number of available GPUs
# ex) "0,1,2"
GPU_DEVICE_ID: str = "0"
# using each GPU memory rate: 0.0-1.0
GPU_MEMORY_RATE: float = 0.9

RANDOM_SEED: int = 42
FLOAT_MAX: float = sys.float_info.max

# draw point parameter
POINT_RADIUS: int = 5
POINT_COLOR: tuple = (0, 0, 255)
POINT_THICKNESS: int = -1
LINE_TYPE: int = cv2.LINE_AA
