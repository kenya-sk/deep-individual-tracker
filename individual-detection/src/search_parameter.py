import logging
import time
from typing import List

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from tensorflow.compat.v1 import InteractiveSession
from tqdm import tqdm

from model import DensityModel
from clustering import apply_clustering_to_density_map
from evaluate import eval_detection, get_ground_truth, eval_metrics
from predict import predict_density_map
from utils import (
    load_image,
    get_current_time_str,
    load_mask_image,
    load_model,
    apply_masking_on_image,
)

# logger setting
current_time = get_current_time_str()
log_path = f"./logs/search_parameter_{current_time}.log"
logging.basicConfig(
    filename=log_path,
    level=logging.DEBUG,
    format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
)
logger = logging.getLogger(__name__)


def search_clustering_threshold(
    dataset_path_df: pd.DataFrame,
    patterns: List[float],
    mask_image: np.array,
    cfg: DictConfig,
) -> pd.DataFrame:
    """Try multiple patterns of threshold values used for clustering and evaluate the accuracy of each.

    Args:
        dataset_path_df (pd.DataFrame): DataFrame containing paths for input, label, and predicted density map
        patterns (List[float]): serach parameter patterns list
        mask_image (np.array): mask image
        cfg (DictConfig): config about analysis image region

    Returns:
        pd.DataFrame: DataFrame on accuracy for each parameter
    """
    result_dictlist = {
        "parameter": [],
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f_measure": [],
    }
    for param in tqdm(patterns, desc="search threshold patterns"):
        # initialized metrics
        accuracy, precision, recall, f_measure = 0.0, 0.0, 0.0, 0.0
        for predicted_path, label_path in zip(
            dataset_path_df["predicted"], dataset_path_df["label"]
        ):
            # load predicted density map and label
            ground_truth_array = get_ground_truth(label_path, mask_image, cfg)
            predicted_density_map = np.load(predicted_path)
            predicted_centroid_array = apply_clustering_to_density_map(
                predicted_density_map, cfg["band_width"], param
            )

            # evaluate current frame
            true_pos, false_pos, false_neg, sample_number = eval_detection(
                predicted_centroid_array, ground_truth_array, cfg["detection_threshold"]
            )
            (
                accuracy_per_image,
                precision_per_image,
                recall_per_image,
                f_measure_per_image,
            ) = eval_metrics(true_pos, false_pos, false_neg, sample_number)

            # upadate metrics
            accuracy += accuracy_per_image
            precision += precision_per_image
            recall += recall_per_image
            f_measure += f_measure_per_image

        # store current paramter results
        sample_num = len(dataset_path_df)
        result_dictlist["parameter"].append(param)
        result_dictlist["accuracy"].append(accuracy / sample_num)
        result_dictlist["precision"].append(precision / sample_num)
        result_dictlist["recall"].append(recall / sample_num)
        result_dictlist["f_measure"].append(f_measure / sample_num)

    return pd.DataFrame(result_dictlist)


def search_prediction_grid(
    dataset_path_df: pd.DataFrame,
    patterns: List[float],
    model: DensityModel,
    tf_session: InteractiveSession,
    mask_image: np.array,
    cfg: DictConfig,
) -> pd.DataFrame:
    """Try multiple patterns of parameters related to the interval at which
    predicts are made, and evaluate the accuracy of each.

    Args:
        dataset_path_df (pd.DataFrame): DataFrame containing paths for input, label, and predicted density map
        patterns (List[float]): serach parameter patterns list
        model (DensityModel): trained density model
        tf_session (InteractiveSession): tensorflow session
        mask_image (np.array): mask image
        cfg (DictConfig): config about analysis image region

    Returns:
        pd.DataFrame: DataFrame on accuracy for each parameter
    """
    result_dictlist = {
        "parameter": [],
        "calculation_time": [],
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f_measure": [],
    }
    for param in tqdm(patterns, desc="search prediciton grid patterns"):
        # initialized metrics
        calculation_time, accuracy, precision, recall, f_measure = (
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )
        for input_path, label_path in zip(
            dataset_path_df["input"], dataset_path_df["label"]
        ):
            # set timer
            start_time = time.time()

            # Predicts a density map with the specified grid size
            image = load_image(input_path, is_rgb=True)
            if mask_image is not None:
                image = apply_masking_on_image(image, mask_image)
            predicted_density_map = predict_density_map(model, tf_session, image, cfg)
            predicted_centroid_array = apply_clustering_to_density_map(
                predicted_density_map, cfg["band_width"], cfg["fixed_cluster_thresh"]
            )

            # load predicted density map and label
            ground_truth_array = get_ground_truth(label_path, mask_image, cfg)

            # evaluate current frame
            true_pos, false_pos, false_neg, sample_number = eval_detection(
                predicted_centroid_array, ground_truth_array, cfg["detection_threshold"]
            )
            (
                accuracy_per_image,
                precision_per_image,
                recall_per_image,
                f_measure_per_image,
            ) = eval_metrics(true_pos, false_pos, false_neg, sample_number)

            # upadate metrics
            calculation_time += time.time() - start_time
            accuracy += accuracy_per_image
            precision += precision_per_image
            recall += recall_per_image
            f_measure += f_measure_per_image

        # store current paramter results
        sample_num = len(dataset_path_df)
        result_dictlist["parameter"].append(param)
        result_dictlist["calculation_time"].append(calculation_time / sample_num)
        result_dictlist["accuracy"].append(accuracy / sample_num)
        result_dictlist["precision"].append(precision / sample_num)
        result_dictlist["recall"].append(recall / sample_num)
        result_dictlist["f_measure"].append(f_measure / sample_num)

    return pd.DataFrame(result_dictlist)


def search_parameter(
    dataset_path_df: pd.DataFrame,
    mask_image: np.array,
    model: DensityModel,
    tf_session: InteractiveSession,
    cfg: DictConfig,
):
    """Search for parameters that maximize accuracy. The search essentially uses a validation data set.
    The parameters that can be explored are "clustering threshold" and "prediction grid".

    Args:
        dataset_path_df (pd.DataFrame): DataFrame containing paths for input, label, and predicted density map
        mask_image (np.array): mask image
        model (DensityModel): trained density model
        tf_session (InteractiveSession): tensorflow session
        cfg (DictConfig): config about analysis image region
    """
    for param, patterns in cfg["search_params"].items():
        assert (
            type(patterns) == list
        ), "The search patterns are expected to be passed in a List type."
        logger.info(f"Search Parameter: {param}")
        logger.info(f"Search Patterns: {patterns}")

        if param == "threshold":
            resutl_df = search_clustering_threshold(
                dataset_path_df, patterns, mask_image, cfg
            )
        elif param == "prediction_grid":
            resutl_df = search_prediction_grid(
                dataset_path_df, patterns, model, tf_session, mask_image, cfg
            )
        else:
            logger.info(f"{param} is not defined.")
            continue

        # save search result
        save_path = f"{cfg['save_directory']}/search_result_{param}.csv"
        resutl_df.to_csv(save_path, index=False)
        logger.info(f'Searched Result Saved in "{save_path}".')


@hydra.main(config_path="../conf", config_name="search_parameter")
def main(cfg: DictConfig) -> None:
    cfg = OmegaConf.to_container(cfg)
    logger.info(f"Loaded config: {cfg}")

    # load input, label and predicted path list
    dataset_path_df = pd.read_csv(cfg["dataset_path"])
    # load mask image
    mask_image = (
        load_mask_image(cfg["mask_path"]) if cfg["mask_path"] is not None else None
    )
    # load trained model
    if "prediction_grid" in cfg["search_params"].keys():
        model, tf_session = load_model(
            cfg["trained_model_directory"],
            cfg["use_gpu_device"],
            cfg["use_memory_rate"],
        )
    else:
        model, tf_session = None, None

    # search best parameter
    search_parameter(dataset_path_df, mask_image, model, tf_session, cfg)


if __name__ == "__main__":
    main()
