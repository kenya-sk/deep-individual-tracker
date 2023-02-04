import logging
import sys
from dataclasses import dataclass
from typing import List

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BasicMetrics:
    accuracy: float
    precision: float
    recall: float
    f_measure: float

    def __init__(
        self, accuracy: float, precision: float, recall: float, f_measure: float
    ) -> None:
        if self.check_metrics_range(accuracy):
            object.__setattr__(self, "accuracy", accuracy)
        else:
            raise ValueError("Accuracy should be in the range 0.0-1.0.")

        if self.check_metrics_range(precision):
            object.__setattr__(self, "precision", precision)
        else:
            raise ValueError("Precision should be in the range 0.0-1.0.")

        if self.check_metrics_range(recall):
            object.__setattr__(self, "recall", recall)
        else:
            raise ValueError("Recall should be in the range 0.0-1.0.")

        if self.check_metrics_range(f_measure):
            object.__setattr__(self, "f_measure", f_measure)
        else:
            raise ValueError("F-Measure should be in the range 0.0-1.0.")

    @staticmethod
    def check_metrics_range(
        metrics: float, metrics_min: float = 0.0, metrics_max: float = 1.0
    ) -> bool:
        """Check if the metrics are within the range and return a bool value

        Args:
            metrics (float): the value of metrics
            metrics_min (float, optional): minimum value of metrics. Defaults to 0.0.
            metrics_max (float, optional): maximum value of metrics. Defaults to 1.0.

        Returns:
            bool: whether the metrics are within range
        """
        if (metrics >= metrics_min) and (metrics <= metrics_max):
            return True
        else:
            return False


@dataclass(frozen=True)
class MetricsSummury:
    total_sample_num: int
    mean_accuracy: float
    mean_precision: float
    mean_recall: float
    mean_f_measure: float
    std_accuracy: float
    std_precision: float
    std_recall: float
    std_f_measure: float
    min_accuracy: float
    min_precision: float
    min_recall: float
    min_f_measure: float
    max_accuracy: float
    max_precision: float
    max_recall: float
    max_f_measure: float

    def __init__(
        self,
        accuracy_list: List[float],
        precision_list: List[float],
        recall_list: List[float],
        f_measure_list: List[float],
    ) -> None:
        object.__setattr__(self, "total_sample_num", len(accuracy_list))
        object.__setattr__(self, "mean_accuracy", np.mean(accuracy_list))
        object.__setattr__(self, "mean_precision", np.mean(precision_list))
        object.__setattr__(self, "mean_recall", np.mean(recall_list))
        object.__setattr__(self, "mean_f_measure", np.mean(f_measure_list))
        object.__setattr__(self, "std_accuracy", np.std(accuracy_list))
        object.__setattr__(self, "std_precision", np.std(precision_list))
        object.__setattr__(self, "std_recall", np.std(recall_list))
        object.__setattr__(self, "std_f_measure", np.std(f_measure_list))
        object.__setattr__(self, "min_accuracy", np.min(accuracy_list))
        object.__setattr__(self, "min_precision", np.min(precision_list))
        object.__setattr__(self, "min_recall", np.min(recall_list))
        object.__setattr__(self, "min_f_measure", np.min(f_measure_list))
        object.__setattr__(self, "max_accuracy", np.max(accuracy_list))
        object.__setattr__(self, "max_precision", np.max(precision_list))
        object.__setattr__(self, "max_recall", np.max(recall_list))
        object.__setattr__(self, "max_f_measure", np.max(f_measure_list))


def eval_metrics(
    true_positive: int, false_positive: int, false_negative: int, sample_num: int
) -> BasicMetrics:
    """Calculate accuracy, precision, recall, and f_measure.

    Args:
        true_positive (int): the number of true positive
        false_positive (int): the number of false positive
        false_negative (int): the number of false negative
        sample_num (int): the number of sample

    Returns:
        BasicMetrics: value object of each metrics
    """
    if sample_num <= 0:
        logger.error("Sample is not existed. At least one sample is required.")
        sys.exit(1)

    # avoid zero division error
    accuracy = true_positive / sample_num

    if true_positive + false_positive != 0:
        precision = true_positive / (true_positive + false_positive)
    else:
        precision = 0.0

    if true_positive + false_negative != 0:
        recall = true_positive / (true_positive + false_negative)
    else:
        recall = 0.0

    if recall + precision != 0:
        f_measure = (2 * recall * precision) / (recall + precision)
    else:
        f_measure = 0.0

    return BasicMetrics(accuracy, precision, recall, f_measure)


def batch_evaluation(
    true_positive_list: List[int],
    false_positive_list: List[int],
    false_negative_list: List[int],
    sample_num_list: List[int],
) -> List[BasicMetrics]:
    """Takes a value list containing multiple sample values and calculate a metric for each

    Args:
        true_positive_list (List[int]): true positive number of each sample
        false_positive_list (List[int]): false positive number of each sample
        false_negative_list (List[int]): false negative number of each sample
        sample_num_list (List[int]): sample number of each sample

    Returns:
        List[BasicMetrics]: metrics of each sample
    """
    metrics_list = []
    data_size = len(true_positive_list)
    for i in range(data_size):
        basic_metrics = eval_metrics(
            true_positive_list[i],
            false_positive_list[i],
            false_negative_list[i],
            sample_num_list[i],
        )
        metrics_list.append(basic_metrics)

    return metrics_list


def calculate_metrics_summury(metrics_list: List[BasicMetrics]) -> MetricsSummury:
    accuracy_list = [metrics.accuracy for metrics in metrics_list]
    precision_list = [metrics.precision for metrics in metrics_list]
    recall_list = [metrics.recall for metrics in metrics_list]
    f_measure_list = [metrics.f_measure for metrics in metrics_list]

    return MetricsSummury(accuracy_list, precision_list, recall_list, f_measure_list)


def output_evaluation_report(
    true_positive_list: List[int],
    false_positive_list: List[int],
    false_negative_list: List[int],
    sample_num_list: List[int],
) -> None:
    """Outputs a formatted summary of evaluation results to the log.

    Args:
        true_positive_list (List[int]): list containing the number of true-positive in each frame
        false_positive_list (List[int]): list containing the number of false-positive in each frame
        false_negative_list (List[int]): list containing the number of false-negative in each frame
        sample_num_list (List[int]): list containing the number of sample in each frame
    """
    assert (
        len(true_positive_list)
        == len(false_positive_list)
        == len(false_negative_list)
        == len(sample_num_list)
    ), "List of each evaluation result are not same length."

    # Calculate each evaluation metric for multiple samples
    metrics_list = batch_evaluation(
        true_positive_list, false_positive_list, false_negative_list, sample_num_list
    )

    # Calculate statistical values for each evaluation metrics
    metrics_summury = calculate_metrics_summury(metrics_list)

    logger.info("\n**************************************************************")
    logger.info(f"Total data size: {metrics_summury.total_sample_num}")
    logger.info("----------------------------------------------------------------")
    logger.info(
        f"Mean Accuracy: {metrics_summury.mean_accuracy:.2f}±{metrics_summury.std_accuracy:.2f}"
    )
    logger.info(
        f"Mean Precision: {metrics_summury.mean_precision:.2f}±{metrics_summury.std_precision:.2f}"
    )
    logger.info(
        f"Mean Recall: {metrics_summury.mean_recall:.2f}±{metrics_summury.std_recall:.2f}"
    )
    logger.info(
        f"Mean F-measure: {metrics_summury.mean_f_measure:.2f}±{metrics_summury.std_f_measure:.2f}"
    )
    logger.info("----------------------------------------------------------------")
    logger.info(f"Min Accuracy: {metrics_summury.min_accuracy:.2f}")
    logger.info(f"Min Precision: {metrics_summury.min_precision:.2f}")
    logger.info(f"Min Recall: {metrics_summury.min_recall:.2f}")
    logger.info(f"Min F-measure: {metrics_summury.min_f_measure:.2f}")
    logger.info("----------------------------------------------------------------")
    logger.info(f"Max Accuracy: {metrics_summury.max_accuracy:.2f}")
    logger.info(f"Max Precision: {metrics_summury.max_precision:.2f}")
    logger.info(f"Max Recall: {metrics_summury.max_recall:.2f}")
    logger.info(f"Max F-measure: {metrics_summury.max_f_measure:.2f}")
    logger.info("****************************************************************")
