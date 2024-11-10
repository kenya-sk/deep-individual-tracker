from monitoring.config import load_config
from monitoring.constants import CONFIG_DIR, MONITORING_CONFIG_NAME
from monitoring.generate_animation import generate_animation
from monitoring.get_init_figure import get_init_figure
from monitoring.logger import logger


def main() -> None:
    cfg = load_config(CONFIG_DIR / MONITORING_CONFIG_NAME)
    logger.info(f"Loaded config: {cfg}")

    logger.info("[START] Set Figure and Each Axis ...")
    fig, axs = get_init_figure()
    logger.info("[DONE] Set Figure and Each Axis")

    logger.info("[START] Generate Animation ...")
    generate_animation(cfg, fig, axs)
    logger.info("[DONE] Generated Animation")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(e)
