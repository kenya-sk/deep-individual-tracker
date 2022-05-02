import logging
import os
import random
from typing import List

import cv2
import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from tqdm import tqdm

from utils import get_full_path_list, load_video, save_image

# logging setting
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s"
)
logger = logging.getLogger(__name__)


def get_sampled_frame_number(total_frame_number: int, sample_rate: int):
    """
    Sample frame numbers randomly according to the sample rate.

    :param total_frame_number: total frame number of target video
    :param sample_rate: sampling rate for frame
    :return: list of sampled frame number
    """
    frame_number_list = []

    # sample target frame number
    start_idx = 0
    for start_idx in range(1, total_frame_number, sample_rate):
        end_idx = start_idx + sample_rate
        sampled_frame_number = random.randint(start_idx, end_idx)
        frame_number_list.append(sampled_frame_number)

    return frame_number_list


def frame_sampler(input_video_list: List, save_frame_dirc: str, sample_rate: int):
    """
    Load the video and save the sampled frames as image data.

    :param input_video_list: list of input video path
    :param save_frame_dirc: save sampled frame path
    :param sample_rate: sampling rate for frame
    :return: None
    """
    for video_path in input_video_list:
        file_name = os.path.splitext(os.path.basename(video_path))[0]
        video = load_video(video_path)
        total_frame_number = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        # get sample target frame number list
        frame_number_list = get_sampled_frame_number(total_frame_number, sample_rate)
        for frame_number in tqdm(frame_number_list):
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

            # load current frame
            ret, frame = video.read()
            if ret:
                save_file_name = f"{save_frame_dirc}/{file_name}_{frame_number}.png"
                save_image(save_file_name, frame)
            else:
                print(f"Error: cannot load {frame_number} frame")


@hydra.main(config_path="../conf", config_name="frame_sampling")
def run_sampler(cfg: DictConfig) -> None:
    """
    Run frame sampler according to the settings defined in the config file.

    :param cfg: config that loaded by @hydra.main()
    :return: None
    """
    logger.info(f"Loaded config: {cfg}")
    original_cwd = get_original_cwd()
    input_video_list = get_full_path_list(original_cwd, cfg.path.input_video_list)
    save_frame_dirc = os.path.join(original_cwd, cfg.path.save_frame_dirc)
    sample_rate = cfg.sample_rate

    # execute frame sampler
    frame_sampler(input_video_list, save_frame_dirc, sample_rate)


if __name__ == "__main__":
    run_sampler()
