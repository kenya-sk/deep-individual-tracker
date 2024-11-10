from datetime import datetime
from pathlib import Path
from typing import List

MODULE_HOME: Path = Path(__file__).resolve().parents[1]
MODUEL_SRC: Path = Path(__file__).resolve().parents[0]

CONFIG_DIR: Path = MODULE_HOME / "config"
LOG_DIR: Path = MODULE_HOME / "logs"
DATA_DIR: Path = MODULE_HOME / "data"

TODAY: str = datetime.today().strftime("%Y%m%d")
LOGGER_NAME = "stats-monitoring"

MONITORING_CONFIG_NAME: str = "monitoring.yaml"

# frame size setting
FRAME_HEIGHT: int = 1080
FRAME_WIDTH: int = 1920

# graph size setting
GRAPH_HEIGHT: int = 240
GRAPH_WIDTH: int = 240
GRAPH_MARGIN: int = 20
GRAPH_NUM: int = 2
GRAPH_FONT_SIZE: int = 18

# zoom graph size setting
ZOOM_GRAPH_HEIGHT: int = 240
ZOOM_GRAPH_WIDTH: int = 480

# figure size setting
# it is caluculated by frame and graph size
FIGURE_HEIGHT: int = FRAME_HEIGHT + (GRAPH_HEIGHT + GRAPH_MARGIN) * (GRAPH_NUM + 1) + 4 * GRAPH_MARGIN
FIGURE_WIDTH: int = FRAME_WIDTH + GRAPH_WIDTH + 5 * GRAPH_MARGIN

# if frame interval is 1000 milli second
# LABEL_IDX = [i for i in range(0, 28801, 1800)]
# if frame interval is 33 milli second (30FPS)
LABEL_IDX_MAX: int = 864000
LABEL_IDX_INTERVAL: int = 54000
LABEL_IDX: List[int] = [i for i in range(0, LABEL_IDX_MAX + 1, LABEL_IDX_INTERVAL)]
TIME_LABELS: List[str] = [
    "9:00",
    "9:30",
    "10:00",
    "10:30",
    "11:00",
    "11:30",
    "12:00",
    "12:30",
    "13:00",
    "13:30",
    "14:00",
    "14:30",
    "15:00",
    "15:30",
    "16:00",
    "16:30",
    "17:00",
]
