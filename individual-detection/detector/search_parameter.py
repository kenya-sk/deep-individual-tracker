import dataclasses
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from tensorflow.compat.v1 import InteractiveSession
from tqdm import tqdm

from detector.clustering import apply_clustering_to_density_map
from detector.config import SearchParameterConfig, load_config
from detector.constants import (
    CONFIG_DIR,
    DATA_DIR,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    SEARCH_PARAMETER_CONFIG_NAME,
)
from detector.evaluate import eval_detection, eval_metrics, get_ground_truth
from detector.index_manager import IndexManager
from detector.logger import logger
from detector.model import DensityModel, load_model
from detector.predict import predict_density_map
from detector.process_dataset import apply_masking_on_image, load_image, load_mask_image


class ParameterStore:
    def __init__(self, cols_order: List[str]) -> None:
        """Initialize ParameterStore class

        Args:
            cols_order (List[str]): define columns name and order to save
        """
        self.cols_order = cols_order
        self.result_dictlist: Dict[str, List[float]] = {key: [] for key in cols_order}

        self.calculation_time_list: List[float] = []
        self.accuracy_list: List[float] = []
        self.precision_list: List[float] = []
        self.recall_list: List[float] = []
        self.f_measure_list: List[float] = []

    def init_per_image_metrics(self) -> None:
        """Initialize lists that store temporarily metrics value"""
        self.calculation_time_list.clear()
        self.accuracy_list.clear()
        self.precision_list.clear()
        self.recall_list.clear()
        self.f_measure_list.clear()

    def update_per_image_metrics(
        self,
        calculation_time: float,
        accuracy: float,
        precision: float,
        recall: float,
        f_measure: float,
    ) -> None:
        """Update lists that store temporarily metrics values

        Args:
            calculation_time (float): culculation time per image
            accuracy (float): accuracy per image
            precision (float): precision per image
            recall (float): recall per image
            f_measure (float): f-measure per image
        """
        self.calculation_time_list.append(calculation_time)
        self.accuracy_list.append(accuracy)
        self.precision_list.append(precision)
        self.recall_list.append(recall)
        self.f_measure_list.append(f_measure)

    def store_aggregation_results(self) -> None:
        """Calculate percentiles based on a list of each metric."""
        # calculate mean of metrics values
        self.result_dictlist["calculation_time_per_image_mean"].append(
            float(np.mean(self.calculation_time_list))
        )
        self.result_dictlist["accuracy_mean"].append(float(np.mean(self.accuracy_list)))
        self.result_dictlist["precision_mean"].append(
            float(np.mean(self.precision_list))
        )
        self.result_dictlist["recall_mean"].append(float(np.mean(self.recall_list)))
        self.result_dictlist["f_measure_mean"].append(
            float(np.mean(self.f_measure_list))
        )

        # calculate std of metrics values
        self.result_dictlist["calculation_time_per_image_std"].append(
            float(np.std(self.calculation_time_list))
        )
        self.result_dictlist["accuracy_std"].append(float(np.std(self.accuracy_list)))
        self.result_dictlist["precision_std"].append(float(np.std(self.precision_list)))
        self.result_dictlist["recall_std"].append(float(np.std(self.recall_list)))
        self.result_dictlist["f_measure_std"].append(float(np.std(self.f_measure_list)))

    def save_results(self, save_path: Path) -> None:
        """Save search result in "save_path".

        Args:
            save_path (Path): save path of CSV file
        """
        results_df = pd.DataFrame(self.result_dictlist)[self.cols_order]
        results_df.to_csv(save_path, index=False)
        logger.info(f'Searched Result Saved in "{save_path}".')


def search(
    search_param: str,
    dataset_path_df: pd.DataFrame,
    patterns: List[float],
    mask_image: Optional[np.ndarray],
    model: DensityModel,
    tf_session: InteractiveSession,
    index_manager: IndexManager,
    cfg: SearchParameterConfig,
) -> None:
    """Validate multiple combinations of parameters and evaluate accuracy, precision, recall, f-measure.

    Args:
        search_param (str): search parameter name
        dataset_path_df (pd.DataFrame): DataFrame containing paths for input, label, and predicted density map
        patterns (List[float]): serach parameter patterns list
        mask_image (np.ndarray, optional): mask image
        model (DensityModel): trained density model
        tf_session (InteractiveSession): tensorflow session
        index_manager (IndexManager): index manager class of masked image
        cfg (SearchParameterConfig): config about analysis image region
    """
    params_store = ParameterStore(cfg.cols_order)
    for param in tqdm(patterns, desc=f"search {search_param} patterns"):
        # store parameter value
        params_store.result_dictlist["parameter"].append(param)
        params_store.result_dictlist["sample_number"].append(len(dataset_path_df))
        # initialized per image metrics
        params_store.init_per_image_metrics()
        for i in range(len(dataset_path_df)):
            # set timer
            start_time = time.time()

            if search_param == "threshold":
                # load predicted density map
                predicted_density_map = np.load(dataset_path_df["predicted"][i])
                # clustering by target threshold
                predicted_centroid_array = apply_clustering_to_density_map(
                    predicted_density_map, cfg.band_width, param
                )
            elif search_param == "prediction_grid":
                # Predicts a density map with the specified grid size
                image = load_image(dataset_path_df["input"][i], is_rgb=True)
                if mask_image is not None:
                    image = apply_masking_on_image(image, mask_image)
                predicted_density_map = predict_density_map(
                    model,
                    tf_session,
                    image,
                    index_manager,
                    param,
                    cfg.index_extract_type,
                    cfg.predict_batch_size,
                )
                predicted_centroid_array = apply_clustering_to_density_map(
                    predicted_density_map,
                    cfg.band_width,
                    cfg.fixed_cluster_threshold,
                )
            else:
                logger.info(f"{param} is not defined.")
                return

            # load groud truth label
            ground_truth_array = get_ground_truth(
                dataset_path_df["label"][i], mask_image
            )

            # evaluate current frame
            true_pos, false_pos, false_neg, sample_number = eval_detection(
                predicted_centroid_array, ground_truth_array, cfg.detection_threshold
            )
            basic_metrics = eval_metrics(true_pos, false_pos, false_neg, sample_number)

            # upadate metrics
            calculation_time = time.time() - start_time
            params_store.update_per_image_metrics(
                calculation_time,
                basic_metrics.accuracy,
                basic_metrics.precision,
                basic_metrics.recall,
                basic_metrics.f_measure,
            )

        # store current paramter aggregation results
        params_store.store_aggregation_results()

    # save search result
    save_path = DATA_DIR / cfg.save_directory / f"search_result_{search_param}.csv"
    params_store.save_results(save_path)


def search_parameter(
    dataset_path_df: pd.DataFrame,
    mask_image: Optional[np.ndarray],
    model: DensityModel,
    tf_session: InteractiveSession,
    index_manager: IndexManager,
    cfg: SearchParameterConfig,
) -> None:
    """Search for parameters that maximize accuracy. The search essentially uses a validation data set.
    The parameters that can be explored are "clustering threshold" and "prediction grid".

    Args:
        dataset_path_df (pd.DataFrame): DataFrame containing paths for input, label, and predicted density map
        mask_image (np.ndarray, optional): mask image
        model (DensityModel): trained density model
        tf_session (InteractiveSession): tensorflow session
        index_manager (IndexManager): index manager class of masked image
        cfg (SearchParameterConfig): config about analysis image region
    """
    for param, patterns in cfg.search_params.items():
        assert (
            type(patterns) == list
        ), "The search patterns are expected to be passed in a List type."
        logger.info(f"Search Parameter: {param}")
        logger.info(f"Search Patterns: {patterns}")

        search(
            param,
            dataset_path_df,
            patterns,
            mask_image,
            model,
            tf_session,
            index_manager,
            cfg,
        )
        logger.info(f"Completed: {param}")


def main() -> None:
    cfg = load_config(CONFIG_DIR / SEARCH_PARAMETER_CONFIG_NAME, SearchParameterConfig)
    logger.info(f"Loaded config: {cfg}")

    # load input, label and predicted path list
    dataset_path_df = (
        pd.read_csv(DATA_DIR / cfg.dataset_path)
        .sample(frac=cfg.sample_rate)
        .reset_index(drop=True)
    )

    # load mask image
    if cfg.mask_path is not None:
        mask_image = load_mask_image(DATA_DIR / cfg.mask_path)
        index_manager = IndexManager(mask_image)
    else:
        mask_image = None
        index_manager = IndexManager(np.ones((FRAME_HEIGHT, FRAME_WIDTH)))

    # load trained model
    if "prediction_grid" in cfg.search_params.keys():
        model, tf_session = load_model(DATA_DIR / cfg.trained_model_directory)
    else:
        model, tf_session = None, None

    # search best parameter
    search_parameter(dataset_path_df, mask_image, model, tf_session, index_manager, cfg)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(e)
