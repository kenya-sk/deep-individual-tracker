from pathlib import Path

import pandas as pd
import pytest

from detector.search_parameter import ParameterStore


@pytest.fixture(scope="function")
def ps() -> ParameterStore:
    cols_order = [
        "parameter",
        "sample_number",
        "calculation_time_per_image_mean",
        "calculation_time_per_image_std",
        "accuracy_mean",
        "accuracy_std",
        "precision_mean",
        "precision_std",
        "recall_mean",
        "recall_std",
        "f_measure_mean",
        "f_measure_std",
    ]

    return ParameterStore(cols_order)


def test_init_per_image_metrics(ps: ParameterStore) -> None:
    ps.init_per_image_metrics()
    assert ps.calculation_time_list == []
    assert ps.accuracy_list == []
    assert ps.precision_list == []
    assert ps.recall_list == []
    assert ps.f_measure_list == []


def test_update_per_image_metrics(ps: ParameterStore) -> None:
    ps.update_per_image_metrics(100.2, 1.0, 0.8, 0.5, 0.6)
    assert ps.calculation_time_list == [100.2]
    assert ps.accuracy_list == [1.0]
    assert ps.precision_list == [0.8]
    assert ps.recall_list == [0.5]
    assert ps.f_measure_list == [0.6]


def test_store_percentile_results(ps: ParameterStore) -> None:
    ps.calculation_time_list = [100.2, 100.1, 100.3, 100.4, 100.5]
    ps.accuracy_list = [1.0, 0.5, 0.4, 0.3, 0.2]
    ps.precision_list = [0.8, 0.4, 0.3, 0.2, 0.1]
    ps.recall_list = [0.5, 0.2, 0.1, 0.05, 0.01]
    ps.f_measure_list = [0.6, 0.3, 0.2, 0.1, 0.05]
    ps.store_aggregation_results()

    # allow an error of 1% of std
    assert ps.result_dictlist["calculation_time_per_image_mean"][0] == pytest.approx(
        100.3, 0.01
    )
    assert ps.result_dictlist["accuracy_mean"][0] == pytest.approx(0.48, 0.001)
    assert ps.result_dictlist["precision_mean"][0] == pytest.approx(0.36, 0.001)
    assert ps.result_dictlist["recall_mean"][0] == pytest.approx(0.172, 0.0001)
    assert ps.result_dictlist["f_measure_mean"][0] == pytest.approx(0.25, 0.001)

    # allow an error of two decimal places
    assert ps.result_dictlist["calculation_time_per_image_std"][0] == pytest.approx(
        0.14, abs=0.05
    )
    assert ps.result_dictlist["accuracy_std"][0] == pytest.approx(0.28, abs=0.005)
    assert ps.result_dictlist["precision_std"][0] == pytest.approx(0.24, abs=0.005)
    assert ps.result_dictlist["recall_std"][0] == pytest.approx(0.18, abs=0.005)
    assert ps.result_dictlist["f_measure_std"][0] == pytest.approx(0.19, abs=0.005)


def test_save_results(ps: ParameterStore, tmp_path: Path) -> None:
    cols_order = ps.cols_order
    ps.save_results(tmp_path / "test_parameter.csv")

    # check file exists
    assert (tmp_path / "test_parameter.csv").is_file()

    # check columns order
    loaded_df = pd.read_csv(str(tmp_path / "test_parameter.csv"))
    assert list(loaded_df.columns) == cols_order
