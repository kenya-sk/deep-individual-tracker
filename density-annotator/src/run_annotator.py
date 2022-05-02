import hydra
from omegaconf import DictConfig
from hydra.utils import get_original_cwd


from density_annotator import DensityAnnotator


@hydra.main(config_path="../conf", config_name="annotator")
def run_annotator(cfg: DictConfig) -> None:
    """
    Run DensityAnnotator according to the settings defined in the config file.

    :param cfg: config that loaded by @hydra.main()
    :return: None
    """
    original_cwd = get_original_cwd()
    annotator = DensityAnnotator(cfg, original_cwd)
    annotator.run()


if __name__ == "__main__":
    run_annotator()
