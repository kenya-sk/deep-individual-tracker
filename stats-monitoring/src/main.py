from typing import NoReturn

from hydra.experimental import compose, initialize

from generate_animation import generate_animation
from get_init_figure import get_init_figure


def main() -> NoReturn:
    print("[START] Set Figure and Each Axis ...")
    fig, axs = get_init_figure(cfg)
    print("------------ [DONE] ------------")

    print("[START] Generate Animation ...")
    generate_animation(cfg, fig, axs)
    print("------------ [DONE] ------------")


if __name__ == "__main__":
    # load parameter from hydra
    initialize(config_path="../conf", job_name="stats-monitoring")
    cfg = compose(config_name="config")

    main()
