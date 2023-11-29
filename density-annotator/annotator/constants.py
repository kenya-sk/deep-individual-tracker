from datetime import datetime
from pathlib import Path

MODULE_HOME: Path = Path(__file__).resolve().parents[1]
MODUEL_SRC: Path = Path(__file__).resolve().parents[0]

CONFIG_DIR: Path = MODULE_HOME / "config"
LOG_DIR: Path = MODULE_HOME / "logs"
DATA_DIR: Path = MODULE_HOME / "data"

TODAY: str = datetime.today().strftime("%Y%m%d")
LOGGER_NAME = "density-annotator"

IMAGE_EXTENTION: str = ".png"

SAMPLER_CONFIG_NAME: str = "frame_sampling.yaml"
ANNOTATOR_CONFIG_NAME: str = "annotator.yaml"
ANNOTATOR_TEST_CONFIG_NAME: str = "test_annotator.yaml"

# define GUI control key
Q_KEY = 0x71  # q key (end)
P_KEY = 0x70  # p key (pause)
D_KEY = 0x64  # d key (delete)
S_KEY = 0x73  # s key (save data and restart)
