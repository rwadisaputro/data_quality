from __future__ import annotations

import numpy as np
import pytest

from data_quality.cardinality import HyperLogLog
from data_quality.cardinality.diagnostics import (
    UnsupportedCardinalityDtypeError,
    driver_stream_fallback_warning,
)


def test_hll_add_register_array_validation_and_update() -> None:
    sketch = HyperLogLog(precision=4)

    with pytest.raises(ValueError, match="one-dimensional"):
        sketch.add_register_array(np.asarray([[0]]), np.asarray([1]))
    with pytest.raises(ValueError, match="same length"):
        sketch.add_register_array(np.asarray([0, 1]), np.asarray([1]))

    sketch.add_register_array(np.asarray([], dtype=np.int64), np.asarray([], dtype=np.int64))

    with pytest.raises(TypeError, match="register_indexes"):
        sketch.add_register_array(np.asarray([0.0]), np.asarray([1]))
    with pytest.raises(TypeError, match="ranks"):
        sketch.add_register_array(np.asarray([0]), np.asarray([1.0]))
    with pytest.raises(ValueError, match="out-of-range"):
        sketch.add_register_array(np.asarray([-1]), np.asarray([1]))
    with pytest.raises(ValueError, match="out-of-range"):
        sketch.add_register_array(np.asarray([sketch.register_count]), np.asarray([1]))
    with pytest.raises(ValueError, match="ranks must"):
        sketch.add_register_array(np.asarray([0]), np.asarray([-1]))
    with pytest.raises(ValueError, match="ranks must"):
        sketch.add_register_array(np.asarray([0]), np.asarray([100]))

    sketch.add_register_array(np.asarray([0, 0, 2]), np.asarray([1, 4, 3]))
    assert sketch.registers[0] == 4
    assert sketch.registers[2] == 3


def test_distributed_diagnostics_payloads() -> None:
    error = UnsupportedCardinalityDtypeError(
        backend="polars",
        column="nested",
        native_dtype="List(Int64)",
        dtype_family="nested",
    )
    assert error.as_dict()["code"] == "unsupported_cardinality_dtype"

    warning = driver_stream_fallback_warning(
        backend="pyspark",
        connection_mode="remote",
    )
    assert warning.as_dict()["code"] == "distributed_driver_stream_fallback"
