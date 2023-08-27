import math
import os
from glob import glob
from typing import List

import cv2
import hydra
import numpy as np
from detector.clustering import apply_clustering_to_density_map
from detector.constants import (
    CONFIG_DIR,
    DATA_DIR,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    PREDICT_CONFIG_NAME,
)
from detector.exceptions import IndexExtractionError, PredictionTypeError
from detector.index_manager import IndexManager
from detector.logger import logger
from detector.model import DensityModel, load_model
from detector.process_dataset import (
    apply_masking_on_image,
    extract_local_data,
    load_image,
    load_mask_image,
)
from detector.utils import display_data_info, get_file_name_from_path, set_capture
from omegaconf import DictConfig, OmegaConf
from tensorflow.compat.v1 import InteractiveSession
from tqdm import tqdm


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
        message = f'index_extract_type="{index_extract_type}" is not supported.'
        logger.error(message)
        raise IndexExtractionError(message)

    return index_list


def predict_density_map(
    model: DensityModel,
    tf_session: InteractiveSession,
    image: np.ndarray,
    index_manager: IndexManager,
    cfg: dict,
) -> np.ndarray:
    # set horizontal index
    index_list = extract_prediction_indices(
        list(index_manager.index_h),
        list(index_manager.index_w),
        cfg["skip_pixel_interval"],
        cfg["index_extract_type"],
    )

    # load local images to be predicted
    X_local, _ = extract_local_data(image, None, index_manager, False, index_list)

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
            index_manager.index_h[target_idx_list],
            index_manager.index_w[target_idx_list],
        ] = pred_array

    return pred_dens_map


def image_prediction(
    model: DensityModel,
    tf_session: InteractiveSession,
    image: np.ndarray,
    frame_num: int,
    output_directory: str,
    index_manager: IndexManager,
    cfg: dict,
) -> None:
    """Predictions are applied to single image data using trained model.

    Args:
        model (DensityModel): trained model
        tf_session (InteractiveSession): tensorflow session
        image (np.ndarray): target raw image
        frame_num (int): target frame number
        output_directory (str): output directory name
        index_manager (IndexManager): index manager class of masked image
        cfg (dict): config dictionary
    """
    logger.info("STSRT: predict density map (frame number= {0})".format(frame_num))

    # predict density map by trained model
    pred_dens_map = predict_density_map(model, tf_session, image, index_manager, cfg)

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
    model: DensityModel,
    tf_session: InteractiveSession,
    index_manager: IndexManager,
    cfg: dict,
) -> None:
    """Predictions are applied to multipule image data using trained model.

    Args:
        model (DensityModel): trained model
        tf_session (InteractiveSession): tensorflow session
        index_manager (IndexManager): index manager class of masked image
        cfg (dict): config dictionary
    """
    # set path information
    input_image_path = f"{DATA_DIR}/{cfg['image_directory']}/{cfg['target_date']}/*.png"
    output_directory = f"{DATA_DIR}/{cfg['output_directory']}/{cfg['target_date']}"
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
        frame_num = int(get_file_name_from_path(path))
        image_prediction(
            model, tf_session, image, frame_num, output_directory, index_manager, cfg
        )

    logger.info(f"Predicted {len(image_path_list)} images (path='{input_image_path}')")
    tf_session.close()


def video_prediction(
    model: DensityModel,
    tf_session: InteractiveSession,
    index_manager: IndexManager,
    cfg: dict,
) -> None:
    """Predictions are applied to video data using trained model.

    Args:
        model (DensityModel): trained model
        tf_session (InteractiveSession): tensorflow session
        index_manager (IndexManager): index manager class of masked image
        cfg (dict): config dictionary
    """
    for hour in range(cfg["start_hour"], cfg["end_hour"]):
        output_directory = (
            f"{DATA_DIR}/{cfg['output_directory']}/{cfg['target_date']}/{hour}"
        )
        os.makedirs(output_directory, exist_ok=True)
        os.makedirs(f"{output_directory}/dens", exist_ok=True)
        os.makedirs(f"{output_directory}/coord", exist_ok=True)
        video_path = f"{DATA_DIR}/{cfg['video_directory']}/{cfg['target_date']}/{cfg['target_date']}{hour:0>2d}00.mp4"
        display_data_info(video_path, output_directory, cfg)

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
                        model,
                        tf_session,
                        frame,
                        frame_num,
                        output_directory,
                        index_manager,
                        cfg,
                    )
            else:
                # reached the last frame of the video
                break

        logger.info(f"Predicted video data (path='{video_path}')")

    # close all session
    cap.release()
    cv2.destoryAllWindows()
    tf_session.close()


@hydra.main(
    config_path=str(CONFIG_DIR), config_name=PREDICT_CONFIG_NAME, version_base="1.1"
)
def main(cfg: DictConfig) -> None:
    cfg = OmegaConf.to_container(cfg)
    logger.info(f"Loaded config: {cfg}")

    # set valid image index information
    mask_image = load_mask_image(str(DATA_DIR / cfg["mask_path"]), normalized=True)
    index_manager = IndexManager(mask_image)

    # load trained model
    model, tf_session = load_model(str(DATA_DIR / cfg["trained_model_directory"]))

    predict_data_type = cfg["predict_data_type"]
    if predict_data_type == "image":
        # predict from image data
        batch_prediction(model, tf_session, index_manager, cfg)
    elif predict_data_type == "video":
        # predict from video data
        video_prediction(model, tf_session, index_manager, cfg)
    else:
        message = f'predict_data_type="{predict_data_type}" is not supported.'
        logger.error(message)
        raise PredictionTypeError(message)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(e)
