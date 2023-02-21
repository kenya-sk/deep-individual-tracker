import logging
import math
import os
import sys
from glob import glob
from typing import List

import cv2
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from tensorflow.compat.v1 import InteractiveSession
from tqdm import tqdm

from clustering import apply_clustering_to_density_map
from constatns import (CONFIG_DIR, DATA_DIR, FRAME_HEIGHT, FRAME_WIDTH,
                       GPU_DEVICE_ID, GPU_MEMORY_RATE, PREDICT_CONFIG_NAME)
from model import DensityModel, load_model
from process_dataset import (apply_masking_on_image, extract_local_data,
                             get_masked_index, load_image, load_mask_image)
from utils import (display_data_info, get_current_time_str,
                   get_file_name_from_path, set_capture)

# logger setting
current_time = get_current_time_str()
log_path = f"./logs/predict_{current_time}.log"
logging.basicConfig(
    filename=log_path,
    level=logging.DEBUG,
    format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
)
logger = logging.getLogger(__name__)


def extract_prediction_indices(
    height_index_list: List,
    width_index_list: List,
    skip_pixel_interval: int,
    index_extract_type: str = "grid",
) -> List:
    """Sets the interval at which individual detection is performed.
    Extract the index list of targets to be detected according to the specified extraction method.

    Args:
        height_index_list (List): indices list of vertical axis
        width_index_list (List): indices list of horizontal axis
        skip_pixel_interval (int): pixel interval to be indexed
        index_extract_type (str, optional): index extract type. Defaults to "grid".

    Returns:
        List: List of extracted indices for predictions
    """
    if skip_pixel_interval == 0:
        return [i for i in range(len(height_index_list))]

    if index_extract_type == "grid":
        index_list = [
            i
            for i, (h, w) in enumerate(zip(height_index_list, width_index_list))
            if (h % skip_pixel_interval == 0) or (w % skip_pixel_interval == 0)
        ]
    elif index_extract_type == "intersect":
        index_list = [
            i
            for i, (h, w) in enumerate(zip(height_index_list, width_index_list))
            if (h % skip_pixel_interval == 0) and (w % skip_pixel_interval == 0)
        ]
    elif index_extract_type == "vertical":
        index_list = [
            i for i, w in enumerate(width_index_list) if w % skip_pixel_interval == 0
        ]
    elif index_extract_type == "horizontal":
        index_list = [
            i for i, h in enumerate(height_index_list) if h % skip_pixel_interval == 0
        ]
    else:
        logger.error(f"Invalid 'index_extract_type': {index_extract_type}")
        sys.exit(1)

    return index_list


def predict_density_map(
    model: DensityModel,
    tf_session: InteractiveSession,
    image: np.array,
    cfg: dict,
) -> np.array:
    # set horizontal index
    index_list = extract_prediction_indices(
        cfg["index_h"],
        cfg["index_w"],
        cfg["skip_pixel_interval"],
        cfg["index_extract_type"],
    )

    # load local images to be predicted
    X_local, _ = extract_local_data(image, None, cfg, False, index_list)

    # set prediction parameters
    pred_batch_size = cfg["predict_batch_size"]
    pred_n_batches = math.ceil(len(index_list) / pred_batch_size)
    pred_dens_map = np.zeros((FRAME_HEIGHT, FRAME_WIDTH), dtype="float32")

    for batch in range(pred_n_batches):
        # extract target indices
        start_idx = batch * pred_batch_size
        end_idx = batch * pred_batch_size + pred_batch_size
        target_idx_list = index_list[start_idx:end_idx]

        # predict each local image
        pred_array = tf_session.run(
            model.y,
            feed_dict={
                model.X: X_local[start_idx:end_idx],
                model.is_training: False,
                model.dropout_rate: 0.0,
            },
        ).reshape(len(target_idx_list))
        logger.debug(f"DONE: batch {batch+1}/{pred_n_batches}")

        pred_dens_map[
            cfg["index_h"][target_idx_list], cfg["index_w"][target_idx_list]
        ] = pred_array

    return pred_dens_map


def image_prediction(
    model: DensityModel,
    tf_session: InteractiveSession,
    image: np.array,
    frame_num: int,
    output_directory: str,
    cfg: dict,
) -> None:
    """Predictions are applied to single image data using trained model.

    Args:
        model (DensityModel): trained model
        tf_session (InteractiveSession): tensorflow session
        image (np.array): target raw image
        frame_num (int): target frame number
        output_directory (str): output directory name
        cfg (dict): config dictionary
    """
    logger.info("STSRT: predict density map (frame number= {0})".format(frame_num))

    # predict density map by trained model
    pred_dens_map = predict_density_map(model, tf_session, image, cfg)

    # save predicted data
    if cfg["is_saved_map"]:
        save_dens_path = f"{output_directory}/dens/{frame_num}.npy"
        np.save(save_dens_path, pred_dens_map)
        logger.info(f"predicted density map saved in '{save_dens_path}'.")

    # calculate centroid by mean shift clustering
    centroid_arr = apply_clustering_to_density_map(
        pred_dens_map, cfg["band_width"], cfg["cluster_thresh"]
    )
    save_coord_path = f"{output_directory}/coord/{frame_num}.csv"
    np.savetxt(
        save_coord_path,
        centroid_arr,
        fmt="%i",
        delimiter=",",
    )


def batch_prediction(
    model: DensityModel, tf_session: InteractiveSession, cfg: dict
) -> None:
    """Predictions are applied to multipule image data using trained model.

    Args:
        model (DensityModel): trained model
        tf_session (InteractiveSession): tensorflow session
        cfg (dict): config dictionary
    """
    # set path information
    input_image_path = f"{cfg['image_directory']}/{cfg['target_date']}/*.png"
    output_directory = f"{cfg['output_directory']}/{cfg['target_date']}"
    os.makedirs(output_directory, exist_ok=True)
    os.makedirs(f"{output_directory}/dens", exist_ok=True)
    os.makedirs(f"{output_directory}/coord", exist_ok=True)
    display_data_info(input_image_path, output_directory, cfg)

    # predcit for each image
    image_path_list = glob(input_image_path)
    for path in tqdm(image_path_list, desc="predit image data"):
        image = load_image(path, is_rgb=True, normalized=True)
        # apply mask on input image
        if cfg["mask_path"] is not None:
            mask_image = load_mask_image(cfg["mask_path"], normalized=True)
            image = apply_masking_on_image(image, mask_image)
        frame_num = get_file_name_from_path(path)
        image_prediction(model, tf_session, image, frame_num, output_directory, cfg)

    logger.info(f"Predicted {len(image_path_list)} images (path='{input_image_path}')")
    tf_session.close()


def video_prediction(
    model: DensityModel, tf_session: InteractiveSession, cfg: dict
) -> None:
    """Predictions are applied to video data using trained model.

    Args:
        model (DensityModel): trained model
        tf_session (InteractiveSession): tensorflow session
        cfg (dict): config dictionary
    """
    for hour in range(cfg["start_hour"], cfg["end_hour"]):
        output_directory = f"{cfg['output_directory']}/{cfg['target_date']}/{hour}"
        os.makedirs(output_directory, exist_ok=True)
        os.makedirs("{0}/dens".format(output_directory), exist_ok=True)
        os.makedirs("{0}/coord".format(output_directory), exist_ok=True)
        video_path = f"{cfg['video_directory']}/{cfg['target_date']}/{cfg['target_date']}{hour:0>2d}00.mp4"
        display_data_info(
            video_path,
            output_directory,
            cfg["skip_width"],
            cfg["predict_batch_size"],
            cfg["band_width"],
            cfg["cluster_thresh"],
            cfg["is_saved_map"],
        )

        # initializetion
        cap, _, _, _, _, _ = set_capture(video_path)
        frame_num = 0

        # predict for each frame at regular interval (config value=predict_interval)
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frame_num += 1
                if frame_num % cfg["predict_interval"] == 0:
                    # apply mask on input image
                    if cfg["mask_path"] is not None:
                        mask_image = load_mask_image(cfg["mask_path"], normalized=True)
                        frame = apply_masking_on_image(frame, mask_image)
                    image_prediction(
                        model, tf_session, frame, frame_num, output_directory, cfg
                    )
            else:
                # reached the last frame of the video
                break

        logger.info(f"Predicted video data (path='{video_path}')")

    # close all session
    cap.release()
    cv2.destoryAllWindows()
    tf_session.close()


@hydra.main(config_path=CONFIG_DIR, config_name=PREDICT_CONFIG_NAME)
def main(cfg: DictConfig) -> None:
    cfg = OmegaConf.to_container(cfg)
    logger.info(f"Loaded config: {cfg}")

    # set valid image index information
    mask_image = load_mask_image(DATA_DIR / cfg["mask_path"], normalized=True)
    index_h, index_w = get_masked_index(mask_image, cfg, horizontal_flip=False)
    # [TODO] create another config dictionary
    cfg["index_h"] = index_h
    cfg["index_w"] = index_w

    # load trained model
    model, tf_session = load_model(
        DATA_DIR / cfg["trained_model_directory"], GPU_DEVICE_ID, GPU_MEMORY_RATE
    )

    predict_data_type = cfg["predict_data_type"]
    if predict_data_type == "image":
        # predict from image data
        batch_prediction(model, tf_session, cfg)
    elif predict_data_type == "video":
        # predict from video data
        video_prediction(model, tf_session, cfg)
    else:
        logger.error(f"Error: not supported data type (={predict_data_type})")


if __name__ == "__main__":
    main()
