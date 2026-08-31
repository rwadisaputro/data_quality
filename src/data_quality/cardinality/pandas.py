"""Pandas backend adapter for dtype-specialised adaptive cardinality profiling."""

from __future__ import annotations

import hashlib
import math
import operator
from collections.abc import Iterator
from dataclasses import dataclass, replace
from enum import Enum
from importlib import import_module
from typing import Any, cast

from data_quality.cardinality.adaptive import (
    AdaptiveCardinalityConfig,
    AdaptiveCardinalityHandler,
    AdaptiveCardinalityResult,
)
from data_quality.cardinality.diagnostics import (
    CardinalityWarning,
    UnsupportedCardinalityValueError,
    scalar_fallback_warning,
)
from data_quality.cardinality.hashing import (
    CANONICAL_BYTES_KERNEL_ID,
    DEFAULT_HASH_CONFIGURATION,
    HashConfiguration,
    canonicalize_scalar,
    hash_configuration_for_kernel,
    hash_ndarray,
)
from data_quality.cardinality.hll import DEFAULT_PRECISION
from data_quality.cardinality.report import (
    ColumnCardinalityOutcome,
    DataFrameCardinalityResult,
)
from data_quality.cardinality.sketch import HashedHyperLogLog

DEFAULT_PANDAS_CHUNK_SIZE = 65_536


class PandasDtypeFamily(str, Enum):
    """Backend-neutral family assigned from one native pandas dtype."""

    BOOLEAN = "boolean"
    INTEGER = "integer"
    FLOATING = "floating"
    COMPLEX = "complex"
    STRING = "string"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    DATETIME_TZ = "datetime_tz"
    TIMEDELTA = "timedelta"
    PERIOD = "period"
    INTERVAL = "interval"
    OBJECT = "object"
    OTHER = "other"


@dataclass(frozen=True, slots=True)
class PandasDtypeIdentity:
    """JSON-safe classification derived from a pandas column's native dtype."""

    family: PandasDtypeFamily
    numpy_kind: str | None
    is_extension_dtype: bool
    canonicalisation_strategy: str

    def as_dict(self) -> dict[str, object]:
        return {
            "dtype_family": self.family.value,
            "numpy_kind": self.numpy_kind,
            "is_extension_dtype": self.is_extension_dtype,
            "canonicalisation_strategy": self.canonicalisation_strategy,
        }


@dataclass(frozen=True, slots=True)
class PandasHashKernel:
    """Native pandas dtype kernel selected before reading row values."""

    kernel_id: str
    exact_representation: str
    scalar_fallback: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "kernel_id": self.kernel_id,
            "exact_representation": self.exact_representation,
            "scalar_fallback": self.scalar_fallback,
            "hash_execution": "numpy_ndarray_bulk",
        }


def identify_pandas_dtype(dtype: object) -> PandasDtypeIdentity:
    """Classify one pandas native dtype entirely from dataframe metadata."""

    try:
        pandas_module = import_module("pandas")
    except ImportError as exc:
        raise ImportError(
            "Pandas cardinality support requires the optional 'pandas' dependency"
        ) from exc

    pandas_types = pandas_module.api.types
    extension_type = pandas_module.api.extensions.ExtensionDtype

    if isinstance(dtype, pandas_module.CategoricalDtype):
        family = PandasDtypeFamily.CATEGORICAL
    elif isinstance(dtype, pandas_module.DatetimeTZDtype):
        family = PandasDtypeFamily.DATETIME_TZ
    elif isinstance(dtype, pandas_module.PeriodDtype):
        family = PandasDtypeFamily.PERIOD
    elif isinstance(dtype, pandas_module.IntervalDtype):
        family = PandasDtypeFamily.INTERVAL
    elif isinstance(dtype, pandas_module.StringDtype):
        family = PandasDtypeFamily.STRING
    elif pandas_types.is_bool_dtype(dtype):
        family = PandasDtypeFamily.BOOLEAN
    elif pandas_types.is_integer_dtype(dtype):
        family = PandasDtypeFamily.INTEGER
    elif pandas_types.is_float_dtype(dtype):
        family = PandasDtypeFamily.FLOATING
    elif pandas_types.is_complex_dtype(dtype):
        family = PandasDtypeFamily.COMPLEX
    elif pandas_types.is_datetime64_dtype(dtype):
        family = PandasDtypeFamily.DATETIME
    elif pandas_types.is_timedelta64_dtype(dtype):
        family = PandasDtypeFamily.TIMEDELTA
    elif pandas_types.is_object_dtype(dtype):
        family = PandasDtypeFamily.OBJECT
    elif pandas_types.is_string_dtype(dtype):
        family = PandasDtypeFamily.STRING
    else:
        family = PandasDtypeFamily.OTHER

    if family in {PandasDtypeFamily.OBJECT, PandasDtypeFamily.OTHER}:
        strategy = f"pandas_scalar_fallback_{family.value}"
    else:
        strategy = f"pandas_dtype_kernel_{family.value}"

    numpy_kind = getattr(dtype, "kind", None)
    return PandasDtypeIdentity(
        family=family,
        numpy_kind=None if numpy_kind is None else str(numpy_kind),
        is_extension_dtype=isinstance(dtype, extension_type),
        canonicalisation_strategy=strategy,
    )


def iter_pandas_hash_arrays(
    dataframe: object,
    column: object,
    *,
    configuration: HashConfiguration = DEFAULT_HASH_CONFIGURATION,
    chunk_size: int = DEFAULT_PANDAS_CHUNK_SIZE,
) -> Iterator[Any]:
    """Yield dtype-specialised ``np.uint64`` hash batches for one DataFrame column."""

    resolution = _resolve_pandas_dataframe_column(dataframe, column)
    pandas_series = resolution[3]
    dtype_identity = resolution[5]
    native_dtype = resolution[7]
    kernel = _pandas_hash_kernel(native_dtype, dtype_identity)
    for tokens, _ in _iter_pandas_canonical_arrays(
        pandas_series,
        chunk_size=chunk_size,
        dtype_identity=dtype_identity,
        column=column,
        native_dtype=native_dtype,
    ):
        yield hash_ndarray(tokens, kernel_id=kernel.kernel_id, configuration=configuration)


def iter_pandas_hashes(
    dataframe: object,
    column: object,
    *,
    configuration: HashConfiguration = DEFAULT_HASH_CONFIGURATION,
    chunk_size: int = DEFAULT_PANDAS_CHUNK_SIZE,
) -> Iterator[int]:
    """Compatibility iterator over a DataFrame column's ndarray hash stream."""

    for hash_values in iter_pandas_hash_arrays(
        dataframe,
        column,
        configuration=configuration,
        chunk_size=chunk_size,
    ):
        yield from (int(value) for value in hash_values)


def pandas_hll_sketch(
    dataframe: object,
    column: object,
    *,
    precision: int = DEFAULT_PRECISION,
    configuration: HashConfiguration = DEFAULT_HASH_CONFIGURATION,
    chunk_size: int = DEFAULT_PANDAS_CHUNK_SIZE,
) -> HashedHyperLogLog:
    """Build a dtype-compatible HLL sketch for one pandas DataFrame column."""

    resolution = _resolve_pandas_dataframe_column(dataframe, column)
    pandas_series = resolution[3]
    dtype_identity = resolution[5]
    native_dtype = resolution[7]
    kernel = _pandas_hash_kernel(native_dtype, dtype_identity)
    effective_configuration = hash_configuration_for_kernel(
        configuration,
        kernel_id=kernel.kernel_id,
    )
    sketch = HashedHyperLogLog(
        precision=precision,
        hash_configuration=effective_configuration,
    )
    for tokens, _ in _iter_pandas_canonical_arrays(
        pandas_series,
        chunk_size=chunk_size,
        dtype_identity=dtype_identity,
        column=column,
        native_dtype=native_dtype,
    ):
        sketch.add_hash_array(
            hash_ndarray(
                tokens,
                kernel_id=kernel.kernel_id,
                configuration=effective_configuration,
            )
        )
    return sketch


def pandas_hll_nunique(
    dataframe: object,
    column: object,
    *,
    precision: int = DEFAULT_PRECISION,
    configuration: HashConfiguration = DEFAULT_HASH_CONFIGURATION,
    chunk_size: int = DEFAULT_PANDAS_CHUNK_SIZE,
) -> int:
    """Estimate non-null distinct cardinality as an integer."""

    return round(
        pandas_hll_sketch(
            dataframe,
            column,
            precision=precision,
            configuration=configuration,
            chunk_size=chunk_size,
        ).estimate()
    )


def pandas_adaptive_cardinality(
    dataframe: object,
    column: object,
    *,
    config: AdaptiveCardinalityConfig | None = None,
    chunk_size: int = DEFAULT_PANDAS_CHUNK_SIZE,
    backend_metadata: dict[str, object] | None = None,
    pipeline_metadata: dict[str, object] | None = None,
) -> AdaptiveCardinalityResult:
    """Profile one pandas DataFrame column through exact-then-HLL adaptive state."""

    resolution = _resolve_pandas_dataframe_column(dataframe, column)
    return _pandas_adaptive_cardinality_resolved(
        resolution,
        config=config,
        chunk_size=chunk_size,
        backend_metadata=backend_metadata,
        pipeline_metadata=pipeline_metadata,
    )


def pandas_dataframe_cardinality(
    dataframe: object,
    *,
    config: AdaptiveCardinalityConfig | None = None,
    chunk_size: int = DEFAULT_PANDAS_CHUNK_SIZE,
    backend_metadata: dict[str, object] | None = None,
    pipeline_metadata: dict[str, object] | None = None,
) -> DataFrameCardinalityResult:
    """Profile every pandas column while isolating unsupported object columns."""

    _, _, pandas_dataframe = _validate_pandas_dataframe(dataframe)
    chunk_size = _validate_chunk_size(chunk_size)
    if backend_metadata is None:
        from data_quality.intake import identify_backend

        backend_metadata = identify_backend(pandas_dataframe).as_dict()

    outcomes: list[ColumnCardinalityOutcome] = []
    for position in range(len(pandas_dataframe.columns)):
        resolution = _resolve_pandas_dataframe_column_position(pandas_dataframe, position)
        column_label = pandas_dataframe.columns[position]
        try:
            result = _pandas_adaptive_cardinality_resolved(
                resolution,
                config=config,
                chunk_size=chunk_size,
                backend_metadata=backend_metadata,
                pipeline_metadata=pipeline_metadata
                or {
                    "entrypoint": "profile_cardinality",
                    "backend_detection": "data_quality.intake.identify_backend",
                    "adapter": "PandasCardinalityAdapter",
                    "dtype_source": "pandas.DataFrame.dtypes",
                    "input_boundary": "dataframe",
                    "column_selection": "all_columns",
                },
            )
        except UnsupportedCardinalityValueError as error:
            outcomes.append(
                ColumnCardinalityOutcome(
                    column=column_label,
                    position=position,
                    result=None,
                    error=error.warning,
                )
            )
        else:
            outcomes.append(
                ColumnCardinalityOutcome(
                    column=column_label,
                    position=position,
                    result=result,
                )
            )

    return DataFrameCardinalityResult(
        backend=dict(backend_metadata),
        total_row_count=len(pandas_dataframe),
        column_count=len(pandas_dataframe.columns),
        columns=tuple(outcomes),
    )


def pandas_adaptive_cardinality_json(
    dataframe: object,
    column: object,
    *,
    config: AdaptiveCardinalityConfig | None = None,
    chunk_size: int = DEFAULT_PANDAS_CHUNK_SIZE,
    indent: int | None = 2,
) -> str:
    """Return one DataFrame-column adaptive result and lineage as JSON."""

    return pandas_adaptive_cardinality(
        dataframe,
        column,
        config=config,
        chunk_size=chunk_size,
    ).to_json(indent=indent)


def _pandas_adaptive_cardinality_resolved(
    resolution: tuple[Any, Any, Any, Any, int, PandasDtypeIdentity, dict[str, object], Any],
    *,
    config: AdaptiveCardinalityConfig | None,
    chunk_size: int,
    backend_metadata: dict[str, object] | None,
    pipeline_metadata: dict[str, object] | None,
) -> AdaptiveCardinalityResult:
    (
        _pandas_module,
        numpy_module,
        pandas_dataframe,
        pandas_series,
        _position,
        dtype_identity,
        column_metadata,
        native_dtype,
    ) = resolution
    chunk_size = _validate_chunk_size(chunk_size)
    base_config = config or AdaptiveCardinalityConfig()
    kernel = _pandas_hash_kernel(native_dtype, dtype_identity)
    effective_hash_configuration = hash_configuration_for_kernel(
        base_config.hash_configuration,
        kernel_id=kernel.kernel_id,
    )
    effective_config = replace(
        base_config,
        hash_configuration=effective_hash_configuration,
    )

    def array_hasher(tokens: object) -> object:
        return hash_ndarray(
            tokens,
            kernel_id=kernel.kernel_id,
            configuration=effective_hash_configuration,
        )

    handler = AdaptiveCardinalityHandler(effective_config, array_hasher=array_hasher)
    chunk_count = 0
    column_label = pandas_dataframe.columns[_position]
    for tokens, row_positions in _iter_pandas_canonical_arrays(
        pandas_series,
        chunk_size=chunk_size,
        dtype_identity=dtype_identity,
        column=column_label,
        native_dtype=native_dtype,
    ):
        chunk_count += 1
        handler.add_canonical_array(tokens, row_positions=row_positions)

    null_count = 0 if len(pandas_series) == 0 else int(pandas_series.isna().sum())
    if backend_metadata is None:
        from data_quality.intake import identify_backend

        backend_metadata = identify_backend(pandas_dataframe).as_dict()

    warnings = _pandas_result_warnings(
        kernel=kernel,
        column=column_label,
        native_dtype=native_dtype,
        dtype_identity=dtype_identity,
    )
    lineage = {
        "backend": backend_metadata,
        "column": column_metadata,
        "pipeline": pipeline_metadata
        or {
            "entrypoint": "pandas_adaptive_cardinality",
            "backend_detection": "data_quality.intake.identify_backend",
            "adapter": "pandas_dataframe",
            "dtype_source": "pandas.DataFrame.dtypes",
            "input_boundary": "dataframe",
        },
        "numpy_version": str(numpy_module.__version__),
        "dataframe_type": (
            f"{type(pandas_dataframe).__module__}.{type(pandas_dataframe).__qualname__}"
        ),
        "dataframe_column_count": len(pandas_dataframe.columns),
        "input_dtype": str(pandas_series.dtype),
        "dtype_family": dtype_identity.family.value,
        "hash_kernel": kernel.as_dict(),
        "scalar_fallback_used": kernel.scalar_fallback,
        "index_type": (
            f"{type(pandas_dataframe.index).__module__}."
            f"{type(pandas_dataframe.index).__qualname__}"
        ),
        "input_row_count": len(pandas_dataframe),
        "processed_non_empty_chunks": chunk_count,
        "input_chunk_count": (
            0 if len(pandas_dataframe) == 0 else math.ceil(len(pandas_dataframe) / chunk_size)
        ),
        "hash_execution": "numpy_ndarray_bulk",
        "canonicalisation_execution": dtype_identity.canonicalisation_strategy,
        "hash_engine": effective_hash_configuration.engine_id,
        "hash_algorithm": effective_hash_configuration.algorithm_id,
    }
    return handler.build_result(
        total_row_count=len(pandas_dataframe),
        null_count=null_count,
        lineage=lineage,
        execution_parameters={
            "batch_size": chunk_size,
            "pandas_chunk_size": chunk_size,
        },
        exact_state_representation=kernel.exact_representation,
        warnings=warnings,
    )


def _iter_pandas_canonical_arrays(
    series: object,
    *,
    chunk_size: int,
    dtype_identity: PandasDtypeIdentity | None = None,
    column: object | None = None,
    native_dtype: object | None = None,
) -> Iterator[tuple[Any, Any]]:
    """Yield non-null dtype-kernel tokens and original positional row indexes."""

    pandas_module, numpy_module, pandas_series = _validate_pandas_series(series)
    chunk_size = _validate_chunk_size(chunk_size)
    effective_dtype = dtype_identity or identify_pandas_dtype(pandas_series.dtype)
    dtype_value = pandas_series.dtype if native_dtype is None else native_dtype

    for start in range(0, len(pandas_series), chunk_size):
        chunk = pandas_series.iloc[start : start + chunk_size]
        null_mask = chunk.isna().to_numpy(dtype=bool, copy=False)
        included_mask = ~null_mask
        if not bool(included_mask.any()):
            continue

        values = chunk.array[included_mask]
        row_positions = start + numpy_module.flatnonzero(included_mask)
        tokens = _canonicalize_pandas_ndarray(
            values,
            dtype_identity=effective_dtype,
            pandas_module=pandas_module,
            numpy_module=numpy_module,
            row_positions=row_positions,
            column=column if column is not None else pandas_series.name,
            native_dtype=dtype_value,
        )
        yield tokens, row_positions


def _canonicalize_pandas_ndarray(
    values: object,
    *,
    dtype_identity: PandasDtypeIdentity,
    pandas_module: Any,
    numpy_module: Any,
    row_positions: object | None = None,
    column: object | None = None,
    native_dtype: object | None = None,
) -> Any:
    """Build exact-state tokens with dtype-specialised NumPy/pandas kernels."""

    array = numpy_module.asarray(values)
    if array.ndim != 1:
        raise ValueError("Pandas cardinality values must be one-dimensional")
    if array.size == 0:
        return numpy_module.empty(0, dtype=array.dtype)

    family = dtype_identity.family
    if family is PandasDtypeFamily.BOOLEAN:
        return numpy_module.asarray(array, dtype=numpy_module.bool_)

    if family is PandasDtypeFamily.INTEGER:
        return numpy_module.asarray(array)

    if family is PandasDtypeFamily.FLOATING:
        result = numpy_module.asarray(array, dtype=numpy_module.float64).copy()
        result[result == 0.0] = 0.0
        return result

    if family is PandasDtypeFamily.COMPLEX:
        result = numpy_module.asarray(array, dtype=numpy_module.complex128).copy()
        real = result.real
        imag = result.imag
        real[real == 0.0] = 0.0
        imag[imag == 0.0] = 0.0
        return result

    if family is PandasDtypeFamily.STRING:
        # pandas.hash_array does not accept NumPy Unicode (``U``) buffers on all
        # supported pandas versions. Object conversion is vectorised and still avoids
        # the scalar canonicalizer.
        return numpy_module.asarray(array, dtype=object)

    if family is PandasDtypeFamily.CATEGORICAL:
        codes = getattr(values, "codes", None)
        if codes is None:  # pragma: no cover - pandas Categorical invariant
            return numpy_module.asarray(array, dtype=object)
        return numpy_module.asarray(codes, dtype=numpy_module.int64)

    if family in {
        PandasDtypeFamily.DATETIME,
        PandasDtypeFamily.DATETIME_TZ,
        PandasDtypeFamily.TIMEDELTA,
        PandasDtypeFamily.PERIOD,
    }:
        asi8 = getattr(values, "asi8", None)
        if asi8 is not None:
            return numpy_module.asarray(asi8, dtype=numpy_module.int64)
        if family is PandasDtypeFamily.PERIOD:
            return numpy_module.asarray(
                [cast(Any, value).ordinal for value in array],
                dtype=numpy_module.int64,
            )
        temporal = pandas_module.Series(array)
        return numpy_module.asarray(temporal.array.asi8, dtype=numpy_module.int64)

    if family is PandasDtypeFamily.INTERVAL:
        return numpy_module.asarray(array, dtype=object)

    return _scalar_fallback_array(
        array,
        pandas_module=pandas_module,
        numpy_module=numpy_module,
        row_positions=row_positions,
        column=column,
        native_dtype=native_dtype,
    )


def _scalar_fallback_array(
    array: Any,
    *,
    pandas_module: Any,
    numpy_module: Any,
    row_positions: object | None,
    column: object | None,
    native_dtype: object | None,
) -> Any:
    positions = (
        None
        if row_positions is None
        else numpy_module.asarray(row_positions, dtype=numpy_module.int64)
    )
    canonical = numpy_module.empty(len(array), dtype=object)
    for index, value in enumerate(array):
        try:
            canonical[index] = _canonicalize_pandas_scalar(
                value,
                pandas_module=pandas_module,
                numpy_module=numpy_module,
            )
        except TypeError as error:
            row_position = None if positions is None else int(positions[index])
            raise UnsupportedCardinalityValueError(
                backend="pandas",
                column=column,
                native_dtype=str(native_dtype),
                row_position=row_position,
                value=value,
            ) from error
    return canonical


def _canonicalize_pandas_scalar(
    value: object,
    *,
    pandas_module: Any,
    numpy_module: Any,
) -> bytes:
    """Normalise pandas/NumPy scalar wrappers for mixed/object fallback only."""

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


def _pandas_hash_kernel(
    native_dtype: object,
    dtype_identity: PandasDtypeIdentity,
) -> PandasHashKernel:
    family = dtype_identity.family
    if family is PandasDtypeFamily.BOOLEAN:
        return PandasHashKernel("pandas-boolean-v1", "numpy_bool_array")
    if family is PandasDtypeFamily.INTEGER:
        return PandasHashKernel("pandas-integer-v1", "numpy_integer_array")
    if family is PandasDtypeFamily.FLOATING:
        return PandasHashKernel("pandas-float64-v1", "numpy_float64_array")
    if family is PandasDtypeFamily.COMPLEX:
        return PandasHashKernel("pandas-complex128-v1", "numpy_complex128_array")
    if family is PandasDtypeFamily.STRING:
        return PandasHashKernel("pandas-string-v1", "numpy_object_string_array")
    if family is PandasDtypeFamily.CATEGORICAL:
        category_signature = _categorical_signature(native_dtype)
        return PandasHashKernel(
            f"pandas-categorical-codes-v1:{category_signature}",
            "numpy_int64_categorical_codes",
        )
    if family is PandasDtypeFamily.DATETIME:
        return PandasHashKernel("pandas-datetime-ns-v1", "numpy_int64_nanoseconds")
    if family is PandasDtypeFamily.DATETIME_TZ:
        return PandasHashKernel("pandas-datetime-utc-ns-v1", "numpy_int64_utc_nanoseconds")
    if family is PandasDtypeFamily.TIMEDELTA:
        return PandasHashKernel("pandas-timedelta-ns-v1", "numpy_int64_nanoseconds")
    if family is PandasDtypeFamily.PERIOD:
        frequency = str(getattr(native_dtype, "freq", native_dtype))
        return PandasHashKernel(
            f"pandas-period-ordinal-v1:{frequency}",
            "numpy_int64_period_ordinals",
        )
    if family is PandasDtypeFamily.INTERVAL:
        subtype = str(getattr(native_dtype, "subtype", "unknown"))
        closed = str(getattr(native_dtype, "closed", "unknown"))
        return PandasHashKernel(
            f"pandas-interval-v1:{subtype}:{closed}",
            "numpy_object_interval_array",
        )
    return PandasHashKernel(
        CANONICAL_BYTES_KERNEL_ID,
        "numpy_object_array_of_canonical_bytes",
        scalar_fallback=True,
    )


def _categorical_signature(native_dtype: object) -> str:
    categories = getattr(native_dtype, "categories", None)
    ordered = bool(getattr(native_dtype, "ordered", False))
    payload = f"{repr(categories)}|ordered={ordered}".encode("utf-8")
    return hashlib.blake2b(payload, digest_size=8, person=b"dq-cat-v1").hexdigest()


def _pandas_result_warnings(
    *,
    kernel: PandasHashKernel,
    column: object,
    native_dtype: object,
    dtype_identity: PandasDtypeIdentity,
) -> tuple[CardinalityWarning, ...]:
    if not kernel.scalar_fallback:
        return ()
    return (
        scalar_fallback_warning(
            backend="pandas",
            column=column,
            native_dtype=str(native_dtype),
            dtype_family=dtype_identity.family.value,
        ),
    )


def _resolve_pandas_dataframe_column(
    dataframe: object,
    column: object,
) -> tuple[Any, Any, Any, Any, int, PandasDtypeIdentity, dict[str, object], Any]:
    """Validate a pandas DataFrame and resolve one uniquely-labelled column."""

    pandas_module, numpy_module, pandas_dataframe = _validate_pandas_dataframe(dataframe)
    try:
        location = pandas_dataframe.columns.get_loc(column)
    except (KeyError, TypeError) as error:
        raise KeyError(f"Pandas dataframe has no column {column!r}") from error

    if isinstance(location, int):
        position = location
    else:
        try:
            position = operator.index(location)
        except TypeError as error:
            raise ValueError(
                f"Pandas column label {column!r} is not unique; cardinality requires one column"
            ) from error
    return _resolve_pandas_dataframe_column_position(pandas_dataframe, position)


def _resolve_pandas_dataframe_column_position(
    dataframe: object,
    position: int,
) -> tuple[Any, Any, Any, Any, int, PandasDtypeIdentity, dict[str, object], Any]:
    pandas_module, numpy_module, pandas_dataframe = _validate_pandas_dataframe(dataframe)
    if not 0 <= position < len(pandas_dataframe.columns):
        raise IndexError("Pandas column position is out of range")
    pandas_series = pandas_dataframe.iloc[:, position]
    native_dtype = pandas_dataframe.dtypes.iloc[position]
    dtype_identity = identify_pandas_dtype(native_dtype)
    column_metadata = {
        "name": str(pandas_dataframe.columns[position]),
        "position": position,
        "native_dtype": str(native_dtype),
        "native_dtype_class": (
            f"{type(native_dtype).__module__}.{type(native_dtype).__qualname__}"
        ),
        "dtype_source": "pandas.DataFrame.dtypes",
        **dtype_identity.as_dict(),
    }
    return (
        pandas_module,
        numpy_module,
        pandas_dataframe,
        pandas_series,
        position,
        dtype_identity,
        column_metadata,
        native_dtype,
    )


def _validate_pandas_dataframe(dataframe: object) -> tuple[Any, Any, Any]:
    """Load optional pandas dependencies and validate the public DataFrame input."""

    try:
        pandas_module = import_module("pandas")
        numpy_module = import_module("numpy")
    except ImportError as exc:
        raise ImportError(
            "Pandas cardinality support requires the optional 'pandas' dependency"
        ) from exc

    if not isinstance(dataframe, pandas_module.DataFrame):
        raise TypeError("Pandas cardinality functions require a pandas.DataFrame")
    return pandas_module, numpy_module, cast(Any, dataframe)


def _validate_pandas_series(series: object) -> tuple[Any, Any, Any]:
    """Load optional pandas dependencies and validate the internal adapter input."""

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
    """Validate the positive integer chunk size used for temporary pandas arrays."""

    if isinstance(chunk_size, bool):
        raise TypeError("chunk_size must be an integer, not a boolean")
    try:
        value = operator.index(chunk_size)
    except TypeError as exc:
        raise TypeError("chunk_size must be integer-like") from exc
    if value <= 0:
        raise ValueError("chunk_size must be greater than zero")
    return value
