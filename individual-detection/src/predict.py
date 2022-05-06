import logging
import os
from curses import noraw
from glob import glob

import cv2
import hydra
import numpy as np
from cv2 import normalize
from omegaconf import DictConfig, OmegaConf
from tensorflow.compat.v1 import InteractiveSession

from clustering import apply_clustering_to_density_map
from model import DensityModel
from utils import (
    apply_masking_on_image,
    display_data_info,
    get_current_time_str,
    get_directory_list,
    get_frame_number_from_path,
    get_local_data,
    get_masked_index,
    load_image,
    load_mask_image,
    load_model,
    set_capture,
)

# logger setting
current_time = get_current_time_str()
log_path = f"./logs/predict_{current_time}.log"
logging.basicConfig(
    filename=log_path,
    level=logging.DEBUG,
    format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
)
logger = logging.getLogger(__name__)


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
    # load local images to be predicted
    X_local, _ = get_local_data(image, None, cfg, is_flip=False)

    # set horizontal index
    if cfg["skip_width"] == 0:
        index_lst = cfg["index_h"]
    else:
        index_lst = [
            i
            for i, (h, w) in enumerate(zip(cfg["index_h"], cfg["index_w"]))
            if (h % cfg["skip_width"] == 0) or (w % cfg["skip_width"] == 0)
        ]

    # set prediction parameters
    pred_batch_size = cfg["predict_batch_size"]
    pred_n_batches = int(len(index_lst) / pred_batch_size)
    pred_dens_map = np.zeros((cfg["image_height"], cfg["image_width"]), dtype="float32")

    logger.info("STSRT: predict density map (frame number= {0})".format(frame_num))
    for batch in range(pred_n_batches):
        # array of skipped local image
        X_skip = np.zeros(
            (
                pred_batch_size,
                cfg["local_image_size"],
                cfg["local_image_size"],
                cfg["image_channel"],
            )
        )
        for index_coord, index_local in enumerate(range(pred_batch_size)):
            current_index = index_lst[batch * pred_batch_size + index_local]
            X_skip[index_coord] = X_local[current_index]

        # predict each local image
        pred_array = tf_session.run(
            model.y,
            feed_dict={
                model.X: X_skip,
                model.is_training: False,
                model.dropout_rate: 0.0,
            },
        ).reshape(pred_batch_size)
        logger.debug(f"DONE: batch {batch+1}/{pred_n_batches}")

        for batch_idx in range(pred_batch_size):
            h_pred = cfg["index_h"][index_lst[batch * pred_batch_size + batch_idx]]
            w_pred = cfg["index_w"][index_lst[batch * pred_batch_size + batch_idx]]
            pred_dens_map[h_pred, w_pred] = pred_array[batch_idx]

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
    root_image_path = f"{cfg['image_directory']}/{cfg['target_date']}"
    image_directory_list = get_directory_list(root_image_path)

    for directory in image_directory_list:
        # set path information
        input_image_path = f"{root_image_path}/{directory}/*.png"
        image_path_list = glob(input_image_path)
        output_directory = f"{cfg['output_directory']}/{cfg['target_date']}"
        os.makedirs(output_directory, exist_ok=True)
        os.makedirs("{0}/dens".format(output_directory), exist_ok=True)
        os.makedirs("{0}/coord".format(output_directory), exist_ok=True)
        display_data_info(input_image_path, output_directory, cfg)

        # predcit for each image
        for path in image_path_list:
            image = load_image(path, is_rgb=True, normalized=True)
            # apply mask on input image
            if cfg["mask_path"] is not None:
                mask_image = load_mask_image(cfg["mask_path"], normalized=True)
                image = apply_masking_on_image(image, mask_image)
            frame_num = get_frame_number_from_path(path)
            image_prediction(model, tf_session, image, frame_num, output_directory, cfg)

        logger.info(
            f"Predicted {len(image_path_list)} images (path='{input_image_path}')"
        )


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


@hydra.main(config_path="../conf", config_name="predict")
def main(cfg: DictConfig) -> None:
    cfg = OmegaConf.to_container(cfg)
    logger.info(f"Loaded config: {cfg}")

    # set valid image index information
    mask_image = load_mask_image(cfg["mask_path"], normalized=True)
    index_h, index_w = get_masked_index(mask_image, cfg, horizontal_flip=False)
    cfg["index_h"] = index_h
    cfg["index_w"] = index_w

    # load trained model
    model, tf_session = load_model(
        cfg["trained_model_directory"], cfg["use_gpu_device"], cfg["use_memory_rate"]
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
