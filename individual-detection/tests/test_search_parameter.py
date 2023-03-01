import pandas as pd
import pytest
from detector.search_parameter import ParameterStore


@pytest.fixture(scope="function")
def ps():
    cols_order = [
        "parameter",
        "sample_number",
        "calculation_time_per_image_mean",
        "accuracy_mean",
        "accuracy_min",
        "accuracy_q1",
        "accuracy_med",
        "accuracy_q3",
        "accuracy_max",
        "precision_mean",
        "precision_min",
        "precision_q1",
        "precision_med",
        "precision_q3",
        "precision_max",
        "recall_mean",
        "recall_min",
        "recall_q1",
        "recall_med",
        "recall_q3",
        "recall_max",
        "f_measure_mean",
        "f_measure_min",
        "f_measure_q1",
        "f_measure_med",
        "f_measure_q3",
        "f_measure_max",
    ]

    return ParameterStore(cols_order)


def test_init_per_image_metrics(ps):
    ps.init_per_image_metrics()
    assert ps.calculation_time_list == []
    assert ps.accuracy_list == []
    assert ps.precision_list == []
    assert ps.recall_list == []
    assert ps.f_measure_list == []


def test_update_per_image_metrics(ps):
    ps.update_per_image_metrics(100.2, 1.0, 0.8, 0.5, 0.6)
    assert ps.calculation_time_list == [100.2]
    assert ps.accuracy_list == [1.0]
    assert ps.precision_list == [0.8]
    assert ps.recall_list == [0.5]
    assert ps.f_measure_list == [0.6]


def test_update_summury_metrics(ps):
    # not exist key case
    prev_result_dictlist = ps.result_dictlist
    ps.update_summury_metrics("not_exist_key", "mean", [1.0, 0.5, 0.3], None)
    assert prev_result_dictlist == ps.result_dictlist

    # mean case
    ps.update_summury_metrics("accuracy_mean", "mean", [1.0, 0.5, 0.3], None)
    assert ps.result_dictlist["accuracy_mean"] == [0.6]

    # parcentile case
    ps.update_summury_metrics(
        "accuracy_med", "percentile", [1.0, 0.5, 0.4, 0.3, 0.2], 50
    )
    assert ps.result_dictlist["accuracy_med"] == [0.4]


def test_store_percentile_results(ps):
    ps.calculation_time_list = [100.2, 100.1, 100.3, 100.4, 100.5]
    ps.accuracy_list = [1.0, 0.5, 0.4, 0.3, 0.2]
    ps.precision_list = [0.8, 0.4, 0.3, 0.2, 0.1]
    ps.recall_list = [0.5, 0.2, 0.1, 0.05, 0.01]
    ps.f_measure_list = [0.6, 0.3, 0.2, 0.1, 0.05]

    ps.store_percentile_results()
    assert ps.result_dictlist["calculation_time_per_image_mean"][0] == pytest.approx(
        100.3, 0.01
    )
    assert ps.result_dictlist["accuracy_mean"][0] == pytest.approx(0.48, 0.001)
    assert ps.result_dictlist["precision_mean"][0] == pytest.approx(0.36, 0.001)
    assert ps.result_dictlist["recall_mean"][0] == pytest.approx(0.172, 0.0001)
    assert ps.result_dictlist["f_measure_mean"][0] == pytest.approx(0.25, 0.001)
    assert ps.result_dictlist["accuracy_min"][0] == 0.20
    assert ps.result_dictlist["precision_min"][0] == 0.10
    assert ps.result_dictlist["recall_min"][0] == 0.01
    assert ps.result_dictlist["f_measure_min"][0] == 0.05
    assert ps.result_dictlist["accuracy_q1"][0] == 0.30
    assert ps.result_dictlist["precision_q1"][0] == 0.20
    assert ps.result_dictlist["recall_q1"][0] == 0.05
    assert ps.result_dictlist["f_measure_q1"][0] == 0.10
    assert ps.result_dictlist["accuracy_med"][0] == 0.40
    assert ps.result_dictlist["precision_med"][0] == 0.30
    assert ps.result_dictlist["recall_med"][0] == 0.10
    assert ps.result_dictlist["f_measure_med"][0] == 0.20
    assert ps.result_dictlist["accuracy_q3"][0] == 0.50
    assert ps.result_dictlist["precision_q3"][0] == 0.40
    assert ps.result_dictlist["recall_q3"][0] == 0.20
    assert ps.result_dictlist["f_measure_q3"][0] == 0.30
    assert ps.result_dictlist["accuracy_max"][0] == 1.0
    assert ps.result_dictlist["precision_max"][0] == 0.8
    assert ps.result_dictlist["recall_max"][0] == 0.5
    assert ps.result_dictlist["f_measure_max"][0] == 0.6


def test_save_results(ps, tmp_path):
    cols_order = ps.cols_order
    ps.save_results(str(tmp_path / "test_parameter.csv"))

    # check file exists
    assert (tmp_path / "test_parameter.csv").is_file()

    # check columns order
    loaded_df = pd.read_csv(str(tmp_path / "test_parameter.csv"))
    assert list(loaded_df.columns) == cols_order
