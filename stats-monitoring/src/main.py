import logging
import time

import hydra
from omegaconf import DictConfig, OmegaConf

from constants import CONFIG_DIR, MONITORING_CONFIG_NAME
from generate_animation import generate_animation
from get_init_figure import get_init_figure

# logger setting
log_path = f"./logs/stats_monitoring_{time.time()}.log"
logging.basicConfig(
    filename=log_path,
    level=logging.DEBUG,
    format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
)
logger = logging.getLogger(__name__)


@hydra.main(config_path=CONFIG_DIR, config_name=MONITORING_CONFIG_NAME)
def main(cfg: DictConfig) -> None:
    # load parameter from hydra
    cfg = OmegaConf.to_container(cfg)
    logger.info(f"Loaded config: {cfg}")

    logger.info("[START] Set Figure and Each Axis ...")
    fig, axs = get_init_figure()
    logger.info("------------ [DONE] ------------")

    logger.info("[START] Generate Animation ...")
    generate_animation(cfg, fig, axs)
    logger.info("------------ [DONE] ------------")


if __name__ == "__main__":
    main()
