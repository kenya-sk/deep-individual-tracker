import logging

from monitoring.constants import LOG_DIR, LOGGER_NAME, TODAY


class ColoredStreamHandler(logging.StreamHandler):
    def __init__(self):
        super().__init__()
        self.level_map = {
            logging.DEBUG: (None, "cyan", False),
            logging.INFO: (None, "green", False),
            logging.WARNING: (None, "yellow", True),
            logging.ERROR: (None, "red", True),
            logging.CRITICAL: ("red", "white", True),
        }

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record)
            self.stream.write(message)
            self.stream.write(self.terminator)
            self.flush()
        except RecursionError:
            raise


def _setup_logger() -> logging.Logger:
    if not LOG_DIR.exists():
        LOG_DIR.mkdir(parents=True)

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "[%(levelname)s] %(asctime)s %(filename)s:%(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = ColoredStreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(
        LOG_DIR / f"{TODAY}_{LOGGER_NAME}.log", mode="a", encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    warning_handler = logging.FileHandler(
        LOG_DIR / f"{TODAY}_{LOGGER_NAME}_warning.log", mode="a", encoding="utf-8"
    )
    warning_handler.setLevel(logging.WARNING)
    warning_handler.setFormatter(formatter)
    logger.addHandler(warning_handler)

    return logger


logger = _setup_logger()
