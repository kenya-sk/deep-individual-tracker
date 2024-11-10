from typing import Annotated, Any

import numpy as np
from matplotlib.axes import Axes
from pydantic import BaseModel, PlainSerializer, PlainValidator


def validate(v: Any) -> np.ndarray:
    if isinstance(v, np.ndarray):
        return v
    else:
        raise TypeError(f"Expected numpy array, got {type(v)}")


def serialize(v: np.ndarray) -> list[list[float]]:
    return v.tolist()


DataArray = Annotated[
    np.ndarray,
    PlainValidator(validate),
    PlainSerializer(serialize),
]


class StatsData(BaseModel):
    mean: DataArray
    past_mean: DataArray
    acceleration: DataArray
    past_acceleration: DataArray


def validate_axes(v: Any) -> Axes:
    if isinstance(v, Axes):
        return v
    else:
        raise TypeError(f"Expected matplotlib.axes.Axes, got {type(v)}")


def serialize_axes(v: Axes) -> dict:
    return {"info": "Axes object"}


AxesField = Annotated[
    Axes,
    PlainValidator(validate_axes),
    PlainSerializer(serialize_axes),
]


class MonitoringAxes(BaseModel):
    frame_ax: AxesField
    x_hist_ax: AxesField
    y_hist_ax: AxesField
    mean_graph_ax: AxesField
    zoom_mean_graph_ax: AxesField
    acc_graph_ax: AxesField
