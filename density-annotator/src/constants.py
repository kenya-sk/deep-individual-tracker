from pathlib import Path

MODULE_HOME: Path = Path(__file__).resolve().parents[1]
MODUEL_SRC: Path = Path(__file__).resolve().parents[0]

CONFIG_DIR: Path = MODULE_HOME / "config"
LOG_DIR: Path = MODULE_HOME / "logs"
DATA_DIR: Path = MODULE_HOME / "data"

IMAGE_EXTENTION: str = ".png"

SAMPLER_CONFIG_NAME: str = "frame_sampler"
ANNOTATOR_CONFIG_NAME: str = "annotator"

# define GUI control key
Q_KEY = 0x71  # q key (end)
P_KEY = 0x70  # p key (pause)
D_KEY = 0x64  # d key (delete)
S_KEY = 0x73  # s key (save data and restart)
