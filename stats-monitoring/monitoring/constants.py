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

MONITORING_CONFIG_NAME: str = "monitoring"

FRAME_HEIGHT: int = 1080
FRAME_WIDTH: int = 1920

GRAPH_HEIGHT: int = 240
GRAPH_WIDTH: int = 240
GRAPH_MARGIN: int = 20
GRAPH_NUM: int = 2
GRAPH_FONT_SIZE: int = 18

FIGURE_HEIGHT: int = (
    FRAME_HEIGHT + (GRAPH_HEIGHT + GRAPH_MARGIN) * (GRAPH_NUM + 1) + 4 * GRAPH_MARGIN
)
FIGURE_WIDTH: int = FRAME_WIDTH + GRAPH_WIDTH + 5 * GRAPH_MARGIN

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
