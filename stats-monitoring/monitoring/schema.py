from typing import Annotated, Any

import numpy as np
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
