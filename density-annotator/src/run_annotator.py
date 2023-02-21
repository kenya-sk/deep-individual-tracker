import hydra
from constants import ANNOTATOR_CONFIG_NAME, CONFIG_DIR
from density_annotator import DensityAnnotator
from omegaconf import DictConfig


@hydra.main(config_path=CONFIG_DIR, config_name=ANNOTATOR_CONFIG_NAME)
def run_annotator(cfg: DictConfig) -> None:
    """
    Run DensityAnnotator according to the settings defined in the config file.

    :param cfg: config that loaded by @hydra.main()
    :return: None
    """
    annotator = DensityAnnotator(cfg)
    annotator.run()


if __name__ == "__main__":
    run_annotator()
