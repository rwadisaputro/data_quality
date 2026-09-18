"""Polars cardinality adapter kernels for eager and lazy frames."""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass, replace
from enum import Enum
from importlib import import_module
from typing import Any, Iterator

from data_quality.cardinality.adaptive import (
    AdaptiveCardinalityConfig,
    AdaptiveCardinalityHandler,
    AdaptiveCardinalityResult,
)
from data_quality.cardinality.diagnostics import (
    CardinalityWarning,
    UnsupportedCardinalityDtypeError,
    UnsupportedCardinalityValueError,
    scalar_fallback_warning,
)
from data_quality.cardinality.hashing import (
    CANONICAL_BYTES_KERNEL_ID,
    HashConfiguration,
    bind_hash_configuration,
    canonicalize_scalar,
)
from data_quality.cardinality.report import (
    ColumnCardinalityOutcome,
    DataFrameCardinalityResult,
)
from data_quality.models import BackendIdentity, FrameKind

DEFAULT_POLARS_BATCH_SIZE = 65_536
POLARS_HASH_ALGORITHM_ID = "polars-series-hash-64-v1"
POLARS_HASH_ENGINE_ID = "polars.Series.hash"
POLARS_HASH_RUNTIME_ID = "polars"
POLARS_CANONICALISATION_VERSION = "polars-dtype-kernels-v1"


class PolarsDtypeFamily(str, Enum):
    """Logical Polars dtype families used to select canonical ndarray kernels."""

    BOOLEAN = "boolean"
    INTEGER = "integer"
    FLOATING = "floating"
    STRING = "string"
    BINARY = "binary"
    DECIMAL = "decimal"
    DATE = "date"
    DATETIME = "datetime"
    DURATION = "duration"
    TIME = "time"
    CATEGORICAL = "categorical"
    ENUM = "enum"
    OBJECT = "object"
    NULL = "null"
    NESTED = "nested"
    OTHER = "other"


@dataclass(frozen=True, slots=True)
class PolarsDtypeIdentity:
    """Native Polars dtype classification and execution metadata."""

    family: PolarsDtypeFamily
    canonicalisation_strategy: str
    is_nested: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "dtype_family": self.family.value,
            "canonicalisation_strategy": self.canonicalisation_strategy,
            "is_nested": self.is_nested,
        }


@dataclass(frozen=True, slots=True)
class PolarsHashKernel:
    """Vectorised exact-token/hash kernel selected for one Polars dtype."""

    kernel_id: str
    exact_representation: str
    scalar_fallback: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "kernel_id": self.kernel_id,
            "exact_representation": self.exact_representation,
            "scalar_fallback": self.scalar_fallback,
        }


def identify_polars_dtype(dtype: object) -> PolarsDtypeIdentity:
    """Classify a native Polars dtype without inspecting column values."""

    polars_module = _load_polars()
    dtype_name = _polars_dtype_name(dtype)
    lowered = dtype_name.lower()

    is_nested = bool(getattr(dtype, "is_nested", lambda: False)())
    if is_nested:
        family = PolarsDtypeFamily.NESTED
    elif dtype_name == "Boolean":
        family = PolarsDtypeFamily.BOOLEAN
    elif bool(getattr(dtype, "is_integer", lambda: False)()):
        family = PolarsDtypeFamily.INTEGER
    elif bool(getattr(dtype, "is_float", lambda: False)()):
        family = PolarsDtypeFamily.FLOATING
    elif bool(getattr(dtype, "is_decimal", lambda: False)()):
        family = PolarsDtypeFamily.DECIMAL
    elif dtype_name in {"String", "Utf8"}:
        family = PolarsDtypeFamily.STRING
    elif dtype_name == "Binary":
        family = PolarsDtypeFamily.BINARY
    elif dtype_name == "Date":
        family = PolarsDtypeFamily.DATE
    elif lowered.startswith("datetime"):
        family = PolarsDtypeFamily.DATETIME
    elif lowered.startswith("duration"):
        family = PolarsDtypeFamily.DURATION
    elif dtype_name == "Time":
        family = PolarsDtypeFamily.TIME
    elif lowered.startswith("categorical"):
        family = PolarsDtypeFamily.CATEGORICAL
    elif lowered.startswith("enum"):
        family = PolarsDtypeFamily.ENUM
    elif dtype_name == "Object" or bool(getattr(dtype, "is_object", lambda: False)()):
        family = PolarsDtypeFamily.OBJECT
    elif dtype_name == "Null":
        family = PolarsDtypeFamily.NULL
    else:
        family = PolarsDtypeFamily.OTHER

    if family is PolarsDtypeFamily.OBJECT:
        strategy = "polars_scalar_fallback_object"
    elif family in {PolarsDtypeFamily.NESTED, PolarsDtypeFamily.OTHER}:
        strategy = "unsupported"
    else:
        strategy = f"polars_dtype_kernel_{family.value}"

    # Access the module so optional dependency errors are raised here rather than later.
    _ = polars_module
    return PolarsDtypeIdentity(
        family=family,
        canonicalisation_strategy=strategy,
        is_nested=is_nested,
    )


def polars_adaptive_cardinality(
    dataframe: object,
    column: object,
    *,
    config: AdaptiveCardinalityConfig | None = None,
    batch_size: int = DEFAULT_POLARS_BATCH_SIZE,
    backend_metadata: dict[str, object] | None = None,
    pipeline_metadata: dict[str, object] | None = None,
) -> AdaptiveCardinalityResult:
    """Profile one eager/lazy Polars column through the shared adaptive handler."""

    polars_module, frame, backend = _validate_polars_frame(dataframe, backend_metadata)
    schema, dtype_source = _polars_schema(frame)
    column_name, position, native_dtype = _resolve_polars_column(schema, column)
    return _polars_profile_columns(
        polars_module,
        frame,
        backend,
        [(column_name, position, native_dtype)],
        config=config,
        batch_size=batch_size,
        dtype_source=dtype_source,
        pipeline_metadata=pipeline_metadata,
        isolate_errors=False,
    )[0].result  # type: ignore[return-value]


def polars_dataframe_cardinality(
    dataframe: object,
    *,
    config: AdaptiveCardinalityConfig | None = None,
    batch_size: int = DEFAULT_POLARS_BATCH_SIZE,
    backend_metadata: dict[str, object] | None = None,
    pipeline_metadata: dict[str, object] | None = None,
) -> DataFrameCardinalityResult:
    """Profile every eager/lazy Polars column in one batch-streaming pass."""

    polars_module, frame, backend = _validate_polars_frame(dataframe, backend_metadata)
    schema, dtype_source = _polars_schema(frame)
    columns = [
        (name, position, schema[name])
        for position, name in enumerate(_schema_names(schema))
    ]
    outcomes = _polars_profile_columns(
        polars_module,
        frame,
        backend,
        columns,
        config=config,
        batch_size=batch_size,
        dtype_source=dtype_source,
        pipeline_metadata=pipeline_metadata,
        isolate_errors=True,
    )
    row_count = 0
    if outcomes:
        successful = next((outcome.result for outcome in outcomes if outcome.result), None)
        if successful is not None:
            row_count = successful.total_row_count
        elif _is_polars_dataframe(polars_module, frame):
            row_count = len(frame)
        else:
            row_count = _polars_lazy_row_count(polars_module, frame)
    elif _is_polars_dataframe(polars_module, frame):
        row_count = len(frame)
    else:
        row_count = _polars_lazy_row_count(polars_module, frame)
    return DataFrameCardinalityResult(
        backend=dict(backend),
        total_row_count=row_count,
        column_count=len(columns),
        columns=tuple(outcomes),
    )


def _polars_profile_columns(
    polars_module: Any,
    frame: Any,
    backend_metadata: dict[str, object],
    columns: list[tuple[str, int, object]],
    *,
    config: AdaptiveCardinalityConfig | None,
    batch_size: int,
    dtype_source: str,
    pipeline_metadata: dict[str, object] | None,
    isolate_errors: bool,
) -> list[ColumnCardinalityOutcome]:
    batch_size = _validate_batch_size(batch_size)
    base_config = config or AdaptiveCardinalityConfig()
    states: dict[int, dict[str, Any]] = {}
    outcomes: dict[int, ColumnCardinalityOutcome] = {}
    polars_version = str(getattr(polars_module, "__version__", "unknown"))

    for name, position, native_dtype in columns:
        identity = identify_polars_dtype(native_dtype)
        if identity.family in {PolarsDtypeFamily.NESTED, PolarsDtypeFamily.OTHER}:
            error = UnsupportedCardinalityDtypeError(
                backend="polars",
                column=name,
                native_dtype=str(native_dtype),
                dtype_family=identity.family.value,
            )
            if not isolate_errors:
                raise error
            outcomes[position] = ColumnCardinalityOutcome(
                column=name,
                position=position,
                result=None,
                error=error.warning,
            )
            continue
        kernel = _polars_hash_kernel(native_dtype, identity)
        effective_hash = _polars_hash_configuration(
            base_config.hash_configuration,
            kernel=kernel,
            polars_version=polars_version,
        )
        effective_config = replace(base_config, hash_configuration=effective_hash)

        def array_hasher(
            tokens: object,
            *,
            _kernel: PolarsHashKernel = kernel,
            _configuration: HashConfiguration = effective_hash,
        ) -> object:
            return _hash_polars_tokens(
                polars_module,
                tokens,
                kernel=_kernel,
                configuration=_configuration,
            )

        states[position] = {
            "name": name,
            "native_dtype": native_dtype,
            "identity": identity,
            "kernel": kernel,
            "config": effective_config,
            "handler": AdaptiveCardinalityHandler(
                effective_config,
                array_hasher=array_hasher,
            ),
            "null_count": 0,
            "warnings": _polars_result_warnings(
                kernel=kernel,
                column=name,
                native_dtype=native_dtype,
                identity=identity,
            ),
        }

    row_offset = 0
    batch_count = 0
    selected_names = [name for name, position, _ in columns if position in states]
    if selected_names:
        for batch in _iter_polars_batches(
            polars_module,
            frame,
            selected_names,
            batch_size=batch_size,
        ):
            batch_count += 1
            batch_len = len(batch)
            for name, position, _ in columns:
                state = states.get(position)
                if state is None:
                    continue
                series = batch.get_column(name)
                try:
                    tokens, positions, null_count = _canonicalize_polars_series(
                        polars_module,
                        series,
                        identity=state["identity"],
                        native_dtype=state["native_dtype"],
                        column=name,
                        row_offset=row_offset,
                    )
                except UnsupportedCardinalityValueError as error:
                    if not isolate_errors:
                        raise
                    outcomes[position] = ColumnCardinalityOutcome(
                        column=name,
                        position=position,
                        result=None,
                        error=error.warning,
                    )
                    states.pop(position, None)
                    continue
                state["null_count"] += null_count
                if len(tokens):
                    state["handler"].add_canonical_array(tokens, row_positions=positions)
            row_offset += batch_len
    elif _is_polars_dataframe(polars_module, frame):
        row_offset = len(frame)
    else:
        row_offset = _polars_lazy_row_count(polars_module, frame)

    for name, position, native_dtype in columns:
        if position in outcomes:
            continue
        state = states[position]
        identity = state["identity"]
        kernel = state["kernel"]
        column_metadata = {
            "name": name,
            "position": position,
            "native_dtype": str(native_dtype),
            "native_dtype_class": (
                f"{type(native_dtype).__module__}.{type(native_dtype).__qualname__}"
            ),
            "dtype_source": dtype_source,
            **identity.as_dict(),
        }
        lineage = {
            "backend": backend_metadata,
            "column": column_metadata,
            "pipeline": pipeline_metadata
            or {
                "entrypoint": "polars_adaptive_cardinality",
                "backend_detection": "data_quality.intake.identify_backend",
                "adapter": "PolarsCardinalityAdapter",
                "dtype_source": dtype_source,
                "input_boundary": "dataframe",
            },
            "polars_version": polars_version,
            "dataframe_type": f"{type(frame).__module__}.{type(frame).__qualname__}",
            "input_row_count": row_offset,
            "dtype_family": identity.family.value,
            "hash_kernel": kernel.as_dict(),
            "scalar_fallback_used": kernel.scalar_fallback,
            "processed_non_empty_batches": batch_count,
            "hash_execution": "polars_series_hash_bulk",
            "canonicalisation_execution": identity.canonicalisation_strategy,
            "hash_engine": state["config"].hash_configuration.engine_id,
            "hash_algorithm": state["config"].hash_configuration.algorithm_id,
        }
        result = state["handler"].build_result(
            total_row_count=row_offset,
            null_count=state["null_count"],
            lineage=lineage,
            execution_parameters={
                "batch_size": batch_size,
                "polars_batch_size": batch_size,
                "polars_execution": (
                    "eager_iter_slices"
                    if _is_polars_dataframe(polars_module, frame)
                    else "lazy_collect_batches_streaming"
                ),
            },
            exact_state_representation=kernel.exact_representation,
            warnings=state["warnings"],
        )
        outcomes[position] = ColumnCardinalityOutcome(
            column=name,
            position=position,
            result=result,
        )

    return [outcomes[position] for _, position, _ in columns]


def _canonicalize_polars_series(
    polars_module: Any,
    series: Any,
    *,
    identity: PolarsDtypeIdentity,
    native_dtype: object,
    column: object,
    row_offset: int,
) -> tuple[Any, Any, int]:
    numpy_module = _load_numpy()
    family = identity.family
    null_mask = series.is_null().to_numpy()
    excluded_mask = numpy_module.asarray(null_mask, dtype=bool)
    if family is PolarsDtypeFamily.FLOATING:
        nan_mask = series.is_nan().fill_null(False).to_numpy()
        excluded_mask |= numpy_module.asarray(nan_mask, dtype=bool)
    included_mask = ~excluded_mask
    null_count = int(excluded_mask.sum())
    if not bool(included_mask.any()):
        return (
            numpy_module.empty(0, dtype=numpy_module.uint8),
            numpy_module.empty(0, dtype=numpy_module.int64),
            null_count,
        )

    positions = row_offset + numpy_module.flatnonzero(included_mask)
    filtered = series.filter(polars_module.Series(included_mask))

    if family is PolarsDtypeFamily.BOOLEAN:
        tokens = filtered.to_numpy().astype(numpy_module.bool_, copy=False)
    elif family is PolarsDtypeFamily.INTEGER:
        tokens = filtered.to_numpy()
    elif family is PolarsDtypeFamily.FLOATING:
        tokens = filtered.to_numpy().astype(numpy_module.float64, copy=True)
        tokens[tokens == 0.0] = 0.0
    elif family in {PolarsDtypeFamily.STRING, PolarsDtypeFamily.BINARY}:
        tokens = numpy_module.asarray(filtered.to_numpy(), dtype=object)
    elif family in {PolarsDtypeFamily.CATEGORICAL, PolarsDtypeFamily.ENUM}:
        tokens = numpy_module.asarray(filtered.cast(polars_module.String).to_numpy(), dtype=object)
    elif family is PolarsDtypeFamily.DECIMAL:
        tokens = numpy_module.asarray(filtered.cast(polars_module.String).to_numpy(), dtype=object)
    elif family is PolarsDtypeFamily.DATE:
        tokens = filtered.cast(polars_module.Int32).to_numpy().astype(numpy_module.int64)
    elif family in {
        PolarsDtypeFamily.DATETIME,
        PolarsDtypeFamily.DURATION,
        PolarsDtypeFamily.TIME,
    }:
        tokens = filtered.cast(polars_module.Int64).to_numpy().astype(numpy_module.int64)
    else:
        values = numpy_module.asarray(filtered.to_numpy(), dtype=object)
        tokens = numpy_module.empty(len(values), dtype=object)
        for index, value in enumerate(values):
            try:
                tokens[index] = canonicalize_scalar(value)
            except TypeError as error:
                raise UnsupportedCardinalityValueError(
                    backend="polars",
                    column=column,
                    native_dtype=str(native_dtype),
                    row_position=int(positions[index]),
                    value=value,
                ) from error
    return tokens, positions, null_count


def _hash_polars_tokens(
    polars_module: Any,
    tokens: object,
    *,
    kernel: PolarsHashKernel,
    configuration: HashConfiguration,
) -> Any:
    numpy_module = _load_numpy()
    array = numpy_module.asarray(tokens)
    if array.ndim != 1:
        raise ValueError("Polars cardinality tokens must be one-dimensional")
    if array.size == 0:
        return numpy_module.empty(0, dtype=numpy_module.uint64)
    seeds = _polars_hash_seeds(configuration.seed)
    hashed = polars_module.Series("__dq_cardinality__", array).hash(
        seed=seeds[0],
        seed_1=seeds[1],
        seed_2=seeds[2],
        seed_3=seeds[3],
    )
    return numpy_module.asarray(hashed.to_numpy(), dtype=numpy_module.uint64)


def _polars_hash_configuration(
    base: HashConfiguration,
    *,
    kernel: PolarsHashKernel,
    polars_version: str,
) -> HashConfiguration:
    return bind_hash_configuration(
        base,
        runtime_id=POLARS_HASH_RUNTIME_ID,
        runtime_version=polars_version,
        algorithm_id=POLARS_HASH_ALGORITHM_ID,
        engine_id=POLARS_HASH_ENGINE_ID,
        canonicalisation_version=POLARS_CANONICALISATION_VERSION,
        kernel_id=kernel.kernel_id,
    )


def _polars_hash_kernel(
    native_dtype: object,
    identity: PolarsDtypeIdentity,
) -> PolarsHashKernel:
    family = identity.family
    dtype_text = str(native_dtype)
    if family is PolarsDtypeFamily.BOOLEAN:
        return PolarsHashKernel("polars-boolean-v1", "numpy_bool_array")
    if family is PolarsDtypeFamily.INTEGER:
        return PolarsHashKernel(f"polars-integer-v1:{dtype_text}", "numpy_integer_array")
    if family is PolarsDtypeFamily.FLOATING:
        return PolarsHashKernel("polars-float64-v1", "numpy_float64_array")
    if family is PolarsDtypeFamily.STRING:
        return PolarsHashKernel("polars-string-v1", "numpy_object_string_array")
    if family is PolarsDtypeFamily.BINARY:
        return PolarsHashKernel("polars-binary-v1", "numpy_object_binary_array")
    if family is PolarsDtypeFamily.DECIMAL:
        return PolarsHashKernel(f"polars-decimal-string-v1:{dtype_text}", "decimal_string_array")
    if family is PolarsDtypeFamily.DATE:
        return PolarsHashKernel("polars-date-days-v1", "numpy_int64_epoch_days")
    if family is PolarsDtypeFamily.DATETIME:
        return PolarsHashKernel(f"polars-datetime-v1:{dtype_text}", "numpy_int64_datetime_units")
    if family is PolarsDtypeFamily.DURATION:
        return PolarsHashKernel(f"polars-duration-v1:{dtype_text}", "numpy_int64_duration_units")
    if family is PolarsDtypeFamily.TIME:
        return PolarsHashKernel("polars-time-v1", "numpy_int64_time_units")
    if family in {PolarsDtypeFamily.CATEGORICAL, PolarsDtypeFamily.ENUM}:
        return PolarsHashKernel(
            f"polars-logical-string-v1:{family.value}",
            "numpy_object_logical_string_array",
        )
    if family is PolarsDtypeFamily.NULL:
        return PolarsHashKernel("polars-null-v1", "empty_array")
    return PolarsHashKernel(
        CANONICAL_BYTES_KERNEL_ID,
        "numpy_object_array_of_canonical_bytes",
        scalar_fallback=True,
    )


def _polars_result_warnings(
    *,
    kernel: PolarsHashKernel,
    column: object,
    native_dtype: object,
    identity: PolarsDtypeIdentity,
) -> tuple[CardinalityWarning, ...]:
    if not kernel.scalar_fallback:
        return ()
    return (
        scalar_fallback_warning(
            backend="polars",
            column=column,
            native_dtype=str(native_dtype),
            dtype_family=identity.family.value,
        ),
    )


def _validate_polars_frame(
    dataframe: object,
    backend_metadata: dict[str, object] | None,
) -> tuple[Any, Any, dict[str, object]]:
    polars_module = _load_polars()
    if not isinstance(dataframe, (polars_module.DataFrame, polars_module.LazyFrame)):
        raise TypeError("Polars cardinality functions require a polars.DataFrame or LazyFrame")
    if backend_metadata is None:
        from data_quality.intake import identify_backend

        backend_metadata = identify_backend(dataframe).as_dict()
    return polars_module, dataframe, dict(backend_metadata)


def _polars_schema(frame: Any) -> tuple[Any, str]:
    polars_module = _load_polars()
    if _is_polars_dataframe(polars_module, frame):
        return frame.schema, "polars.DataFrame.schema"
    return frame.collect_schema(), "polars.LazyFrame.collect_schema"


def _resolve_polars_column(schema: Any, column: object) -> tuple[str, int, object]:
    names = _schema_names(schema)
    if not isinstance(column, str):
        raise TypeError("Polars cardinality column selectors must be column-name strings")
    try:
        position = names.index(column)
    except ValueError as error:
        raise KeyError(f"Polars dataframe has no column {column!r}") from error
    return column, position, schema[column]


def _iter_polars_batches(
    polars_module: Any,
    frame: Any,
    selected_names: list[str],
    *,
    batch_size: int,
) -> Iterator[Any]:
    if _is_polars_dataframe(polars_module, frame):
        selected = frame.select(selected_names)
        yield from selected.iter_slices(n_rows=batch_size)
        return
    selected = frame.select(selected_names)
    collect_batches = getattr(selected, "collect_batches", None)
    if collect_batches is not None:
        yield from collect_batches(
            chunk_size=batch_size,
            maintain_order=True,
            engine="streaming",
        )
        return
    collected = selected.collect(engine="streaming")
    yield from collected.iter_slices(n_rows=batch_size)


def _polars_lazy_row_count(polars_module: Any, frame: Any) -> int:
    counted = frame.select(polars_module.len().alias("__dq_rows__")).collect(engine="streaming")
    return int(counted.item(0, 0))


def _schema_names(schema: Any) -> list[str]:
    names_method = getattr(schema, "names", None)
    if callable(names_method):
        return list(names_method())
    return list(schema.keys())


def _polars_dtype_name(dtype: object) -> str:
    base_type = getattr(dtype, "base_type", None)
    if callable(base_type):
        base = base_type()
        return getattr(base, "__name__", str(base))
    return getattr(dtype, "__name__", type(dtype).__name__)


def _is_polars_dataframe(polars_module: Any, frame: object) -> bool:
    return isinstance(frame, polars_module.DataFrame)


def _polars_hash_seeds(seed: int) -> tuple[int, int, int, int]:
    mask = (1 << 64) - 1
    return (
        seed & mask,
        (seed ^ 0x9E3779B97F4A7C15) & mask,
        (seed ^ 0xBF58476D1CE4E5B9) & mask,
        (seed ^ 0x94D049BB133111EB) & mask,
    )


def _validate_batch_size(batch_size: int) -> int:
    if isinstance(batch_size, bool):
        raise TypeError("batch_size must be an integer, not a boolean")
    try:
        value = operator.index(batch_size)
    except TypeError as error:
        raise TypeError("batch_size must be integer-like") from error
    if value <= 0:
        raise ValueError("batch_size must be greater than zero")
    return value


def _load_polars() -> Any:
    try:
        return import_module("polars")
    except ImportError as error:
        raise ImportError(
            "Polars cardinality support requires the optional 'polars' dependency"
        ) from error


def _load_numpy() -> Any:
    try:
        return import_module("numpy")
    except ImportError as error:
        raise ImportError("Polars cardinality support requires NumPy") from error
