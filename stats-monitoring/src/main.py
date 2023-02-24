import logging
import time

import hydra
from constants import CONFIG_DIR, MONITORING_CONFIG_NAME
from generate_animation import generate_animation
from get_init_figure import get_init_figure
from logger import logger
from omegaconf import DictConfig, OmegaConf


@hydra.main(
    config_path=str(CONFIG_DIR), config_name=MONITORING_CONFIG_NAME, version_base="1.1"
)
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
