"""Pandas adapter for stable HyperLogLog distinct-cardinality estimation."""

from __future__ import annotations

import operator
from collections.abc import Iterator
from importlib import import_module
from typing import Any, cast

from data_quality.cardinality.hashing import (
    DEFAULT_HASH_CONFIGURATION,
    HashConfiguration,
    canonicalize_scalar,
    hash_canonical_bytes,
)
from data_quality.cardinality.hll import DEFAULT_PRECISION
from data_quality.cardinality.sketch import HashedHyperLogLog

DEFAULT_PANDAS_CHUNK_SIZE = 65_536


def iter_pandas_hashes(
    series: object,
    *,
    configuration: HashConfiguration = DEFAULT_HASH_CONFIGURATION,
    chunk_size: int = DEFAULT_PANDAS_CHUNK_SIZE,
) -> Iterator[int]:
    """Yield stable hashes for non-null values in a pandas Series.

    Null-like values recognised by ``Series.isna()`` are excluded. Processing is
    chunked so temporary null masks stay bounded instead of allocating additional
    full-column objects solely for HLL ingestion.
    """

    pandas_module, numpy_module, pandas_series = _validate_pandas_series(series)
    chunk_size = _validate_chunk_size(chunk_size)

    for start in range(0, len(pandas_series), chunk_size):
        chunk = pandas_series.iloc[start : start + chunk_size]
        null_mask = chunk.isna().to_numpy(dtype=bool, copy=False)
        values = chunk.array

        for position, is_null in enumerate(null_mask):
            if is_null:
                continue
            canonical_value = _canonicalize_pandas_scalar(
                values[position],
                pandas_module=pandas_module,
                numpy_module=numpy_module,
            )
            yield hash_canonical_bytes(
                canonical_value,
                configuration=configuration,
            )


def pandas_hll_sketch(
    series: object,
    *,
    precision: int = DEFAULT_PRECISION,
    configuration: HashConfiguration = DEFAULT_HASH_CONFIGURATION,
    chunk_size: int = DEFAULT_PANDAS_CHUNK_SIZE,
) -> HashedHyperLogLog:
    """Build a configured HLL sketch from non-null values of a pandas Series."""

    sketch = HashedHyperLogLog(
        precision=precision,
        hash_configuration=configuration,
    )
    sketch.add_hashes(
        iter_pandas_hashes(
            series,
            configuration=configuration,
            chunk_size=chunk_size,
        )
    )
    return sketch


def pandas_hll_nunique(
    series: object,
    *,
    precision: int = DEFAULT_PRECISION,
    configuration: HashConfiguration = DEFAULT_HASH_CONFIGURATION,
    chunk_size: int = DEFAULT_PANDAS_CHUNK_SIZE,
) -> float:
    """Estimate non-null distinct cardinality for a pandas Series using HLL."""

    return pandas_hll_sketch(
        series,
        precision=precision,
        configuration=configuration,
        chunk_size=chunk_size,
    ).estimate()


def _canonicalize_pandas_scalar(
    value: object,
    *,
    pandas_module: Any,
    numpy_module: Any,
) -> bytes:
    """Normalise pandas/NumPy scalar wrappers into the shared canonical format."""

    if isinstance(value, pandas_module.Timestamp):
        if value.tzinfo is not None:
            value = value.tz_convert("UTC")
            return b"ta" + value.isoformat().encode("ascii")
        return b"tn" + value.isoformat().encode("ascii")

    if isinstance(value, pandas_module.Timedelta):
        return b"u" + str(int(value.value)).encode("ascii")

    if isinstance(value, pandas_module.Period):
        return (
            b"p"
            + str(value.freqstr).encode("ascii")
            + b":"
            + str(value.ordinal).encode("ascii")
        )

    if isinstance(value, pandas_module.Interval):
        left = _canonicalize_pandas_scalar(
            value.left,
            pandas_module=pandas_module,
            numpy_module=numpy_module,
        )
        right = _canonicalize_pandas_scalar(
            value.right,
            pandas_module=pandas_module,
            numpy_module=numpy_module,
        )
        closed = str(value.closed).encode("ascii")
        return (
            b"v"
            + len(left).to_bytes(4, "big")
            + left
            + len(right).to_bytes(4, "big")
            + right
            + closed
        )

    if isinstance(value, numpy_module.datetime64):
        return _canonicalize_pandas_scalar(
            pandas_module.Timestamp(value),
            pandas_module=pandas_module,
            numpy_module=numpy_module,
        )

    if isinstance(value, numpy_module.timedelta64):
        return _canonicalize_pandas_scalar(
            pandas_module.Timedelta(value),
            pandas_module=pandas_module,
            numpy_module=numpy_module,
        )

    if isinstance(value, numpy_module.bool_):
        value = bool(value)
    elif isinstance(value, numpy_module.integer):
        value = int(value)
    elif isinstance(value, numpy_module.floating):
        value = float(value)
    elif isinstance(value, numpy_module.complexfloating):
        value = complex(value)
    elif isinstance(value, numpy_module.str_):
        value = str(value)
    elif isinstance(value, numpy_module.bytes_):
        value = bytes(value)

    return canonicalize_scalar(value)


def _validate_pandas_series(series: object) -> tuple[Any, Any, Any]:
    """Load optional pandas dependencies and validate the adapter input."""

    try:
        pandas_module = import_module("pandas")
        numpy_module = import_module("numpy")
    except ImportError as exc:
        raise ImportError(
            "Pandas cardinality support requires the optional 'pandas' dependency"
        ) from exc

    if not isinstance(series, pandas_module.Series):
        raise TypeError("Pandas HLL functions require a pandas.Series")

    return pandas_module, numpy_module, cast(Any, series)


def _validate_chunk_size(chunk_size: int) -> int:
    """Validate the positive integer chunk size used for temporary pandas masks."""

    if isinstance(chunk_size, bool):
        raise TypeError("chunk_size must be an integer, not a boolean")

    try:
        value = operator.index(chunk_size)
    except TypeError as exc:
        raise TypeError("chunk_size must be integer-like") from exc

    if value <= 0:
        raise ValueError("chunk_size must be greater than zero")
    return value
