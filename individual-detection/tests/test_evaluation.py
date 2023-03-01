import pytest
from detector.evaluation_metrics import (
    BasicMetrics,
    batch_evaluation,
    calculate_metrics_summury,
    eval_metrics,
)
from detector.exceptions import DetectionSampleNumberError


def test_eval_metrics():
    # all detected case
    true_positive = 100
    false_positive = 0
    false_negative = 0
    sample_num = 100
    basic_metrics = eval_metrics(
        true_positive, false_positive, false_negative, sample_num
    )
    assert basic_metrics.accuracy == 1.0
    assert basic_metrics.precision == 1.0
    assert basic_metrics.recall == 1.0
    assert basic_metrics.f_measure == 1.0

    # all undetected case
    true_positive = 0
    false_positive = 0
    false_negative = 100
    sample_num = 100
    basic_metrics = eval_metrics(
        true_positive, false_positive, false_negative, sample_num
    )
    assert basic_metrics.accuracy == 0.0
    assert basic_metrics.precision == 0.0
    assert basic_metrics.recall == 0.0
    assert basic_metrics.f_measure == 0.0

    # half detected case
    true_positive = 50
    false_positive = 50
    false_negative = 50
    sample_num = 100
    basic_metrics = eval_metrics(
        true_positive, false_positive, false_negative, sample_num
    )
    assert basic_metrics.accuracy == 0.5
    assert basic_metrics.precision == 0.5
    assert basic_metrics.recall == 0.5
    assert basic_metrics.f_measure == 0.5

    # sample number error case
    true_positive = 50
    false_positive = 50
    false_negative = 50
    sample_num = 0
    with pytest.raises(DetectionSampleNumberError):
        basic_metrics = eval_metrics(
            true_positive, false_positive, false_negative, sample_num
        )


def test_batch_evaluation():
    true_positive_list = [100, 50, 0]
    false_positive_list = [0, 50, 0]
    false_negative_list = [0, 50, 100]
    sample_num_list = [100, 100, 100]
    metrics_list = batch_evaluation(
        true_positive_list, false_positive_list, false_negative_list, sample_num_list
    )
    assert len(metrics_list) == 3
    assert metrics_list[0].accuracy == 1.0
    assert metrics_list[0].precision == 1.0
    assert metrics_list[0].recall == 1.0
    assert metrics_list[0].f_measure == 1.0
    assert metrics_list[1].accuracy == 0.5
    assert metrics_list[1].precision == 0.5
    assert metrics_list[1].recall == 0.5
    assert metrics_list[1].f_measure == 0.5
    assert metrics_list[2].accuracy == 0.0
    assert metrics_list[2].precision == 0.0
    assert metrics_list[2].recall == 0.0
    assert metrics_list[2].f_measure == 0.0


def test_calculate_metrics_summury():
    metrics_list = [
        BasicMetrics(1.0, 1.0, 1.0, 1.0),
        BasicMetrics(0.5, 0.5, 0.5, 0.5),
        BasicMetrics(0.0, 0.0, 0.0, 0.0),
    ]
    metrics_summury = calculate_metrics_summury(metrics_list)
    assert metrics_summury.total_sample_num == 3
    assert metrics_summury.mean_accuracy == 0.5
    assert metrics_summury.mean_precision == 0.5
    assert metrics_summury.mean_recall == 0.5
    assert metrics_summury.mean_f_measure == 0.5
    assert round(metrics_summury.std_accuracy, 3) == 0.408
    assert round(metrics_summury.std_precision, 3) == 0.408
    assert round(metrics_summury.std_recall, 3) == 0.408
    assert round(metrics_summury.std_f_measure, 3) == 0.408
    assert metrics_summury.min_accuracy == 0.0
    assert metrics_summury.min_precision == 0.0
    assert metrics_summury.min_recall == 0.0
    assert metrics_summury.min_f_measure == 0.0
    assert metrics_summury.max_accuracy == 1.0
    assert metrics_summury.max_precision == 1.0
    assert metrics_summury.max_recall == 1.0
    assert metrics_summury.max_f_measure == 1.0
