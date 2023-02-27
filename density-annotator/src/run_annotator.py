import hydra
from constants import ANNOTATOR_CONFIG_NAME, CONFIG_DIR
from density_annotator import DensityAnnotator
from logger import logger
from omegaconf import DictConfig


@hydra.main(
    config_path=str(CONFIG_DIR), config_name=ANNOTATOR_CONFIG_NAME, version_base="1.1"
)
def main(cfg: DictConfig) -> None:
    """
    Run DensityAnnotator according to the settings defined in the config file.

    :param cfg: config that loaded by @hydra.main()
    :return: None
    """
    annotator = DensityAnnotator(cfg)
    annotator.run()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(e)
