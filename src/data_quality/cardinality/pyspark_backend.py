"""PySpark SQL cardinality profiling using distributed exact/HLL stages."""

from __future__ import annotations

import math
import operator
import sys
from dataclasses import dataclass
from enum import Enum
from importlib import import_module
from typing import Any

from data_quality.cardinality.adaptive import (
    AdaptiveCardinalityConfig,
    AdaptiveCardinalityResult,
    CardinalityMode,
    PromotionMetadata,
    PromotionReason,
)
from data_quality.cardinality.diagnostics import UnsupportedCardinalityDtypeError
from data_quality.cardinality.hashing import (
    HASH_WIDTH_BITS,
    HashConfiguration,
    bind_hash_configuration,
)
from data_quality.cardinality.report import (
    ColumnCardinalityOutcome,
    DataFrameCardinalityResult,
)
from data_quality.cardinality.sketch import HashedHyperLogLog

DEFAULT_PYSPARK_BATCH_SIZE = 65_536
PYSPARK_HASH_ALGORITHM_ID = "spark-xxhash64-column-v1"
PYSPARK_HASH_ENGINE_ID = "pyspark.sql.functions.xxhash64"
PYSPARK_HASH_RUNTIME_ID = "pyspark"
PYSPARK_CANONICALISATION_VERSION = "spark-sql-native-dtype-v1"


class PySparkDtypeFamily(str, Enum):
    """Spark SQL dtype families supported by the cardinality adapter."""

    BOOLEAN = "boolean"
    INTEGER = "integer"
    FLOATING = "floating"
    DECIMAL = "decimal"
    STRING = "string"
    BINARY = "binary"
    DATE = "date"
    TIMESTAMP = "timestamp"
    TIMESTAMP_NTZ = "timestamp_ntz"
    DAYTIME_INTERVAL = "daytime_interval"
    YEARMONTH_INTERVAL = "yearmonth_interval"
    NULL = "null"
    NESTED = "nested"
    OTHER = "other"


@dataclass(frozen=True, slots=True)
class PySparkDtypeIdentity:
    """Native Spark SQL dtype classification and hash-kernel metadata."""

    family: PySparkDtypeFamily
    canonicalisation_strategy: str
    is_nested: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "dtype_family": self.family.value,
            "canonicalisation_strategy": self.canonicalisation_strategy,
            "is_nested": self.is_nested,
        }


@dataclass(frozen=True, slots=True)
class PySparkHashKernel:
    """Spark SQL native hash/register kernel selected for one column dtype."""

    kernel_id: str
    exact_representation: str

    def as_dict(self) -> dict[str, object]:
        return {
            "kernel_id": self.kernel_id,
            "exact_representation": self.exact_representation,
            "scalar_fallback": False,
        }


def identify_pyspark_dtype(dtype: object) -> PySparkDtypeIdentity:
    """Classify one ``pyspark.sql.types.DataType`` without sampling rows."""

    types_module = _load_pyspark_types()
    nested_types = _existing_types(types_module, "ArrayType", "MapType", "StructType")
    integer_types = _existing_types(
        types_module,
        "ByteType",
        "ShortType",
        "IntegerType",
        "LongType",
    )
    floating_types = _existing_types(types_module, "FloatType", "DoubleType")
    string_types = _existing_types(types_module, "StringType", "CharType", "VarcharType")

    if nested_types and isinstance(dtype, nested_types):
        family = PySparkDtypeFamily.NESTED
    elif isinstance(dtype, _require_type(types_module, "BooleanType")):
        family = PySparkDtypeFamily.BOOLEAN
    elif integer_types and isinstance(dtype, integer_types):
        family = PySparkDtypeFamily.INTEGER
    elif floating_types and isinstance(dtype, floating_types):
        family = PySparkDtypeFamily.FLOATING
    elif isinstance(dtype, _require_type(types_module, "DecimalType")):
        family = PySparkDtypeFamily.DECIMAL
    elif string_types and isinstance(dtype, string_types):
        family = PySparkDtypeFamily.STRING
    elif isinstance(dtype, _require_type(types_module, "BinaryType")):
        family = PySparkDtypeFamily.BINARY
    elif isinstance(dtype, _require_type(types_module, "DateType")):
        family = PySparkDtypeFamily.DATE
    elif _is_existing_instance(dtype, types_module, "TimestampNTZType"):
        family = PySparkDtypeFamily.TIMESTAMP_NTZ
    elif isinstance(dtype, _require_type(types_module, "TimestampType")):
        family = PySparkDtypeFamily.TIMESTAMP
    elif _is_existing_instance(dtype, types_module, "DayTimeIntervalType"):
        family = PySparkDtypeFamily.DAYTIME_INTERVAL
    elif _is_existing_instance(dtype, types_module, "YearMonthIntervalType"):
        family = PySparkDtypeFamily.YEARMONTH_INTERVAL
    elif isinstance(dtype, _require_type(types_module, "NullType")):
        family = PySparkDtypeFamily.NULL
    else:
        family = PySparkDtypeFamily.OTHER

    strategy = (
        "unsupported"
        if family in {PySparkDtypeFamily.NESTED, PySparkDtypeFamily.OTHER}
        else f"spark_sql_native_{family.value}"
    )
    return PySparkDtypeIdentity(
        family=family,
        canonicalisation_strategy=strategy,
        is_nested=family is PySparkDtypeFamily.NESTED,
    )


def pyspark_adaptive_cardinality(
    dataframe: object,
    column: object,
    *,
    config: AdaptiveCardinalityConfig | None = None,
    batch_size: int = DEFAULT_PYSPARK_BATCH_SIZE,
    backend_metadata: dict[str, object] | None = None,
    pipeline_metadata: dict[str, object] | None = None,
) -> AdaptiveCardinalityResult:
    """Profile one Spark SQL column using distributed two-stage routing.

    Spark cannot mutate one aggregate from exact-set state into Python HLL state inside
    a SQL aggregation. The adapter therefore follows the project's documented two-stage
    fallback: bounded distributed exact discovery first, then our classic HLL register
    computation in Spark SQL only when the exact limits are exceeded.
    """

    pyspark_module, functions, dataframe_value, backend = _validate_pyspark_dataframe(
        dataframe,
        backend_metadata,
    )
    batch_size = _validate_batch_size(batch_size)
    field, position = _resolve_pyspark_field(dataframe_value, column)
    identity = identify_pyspark_dtype(field.dataType)
    if identity.family in {PySparkDtypeFamily.NESTED, PySparkDtypeFamily.OTHER}:
        raise UnsupportedCardinalityDtypeError(
            backend="pyspark",
            column=field.name,
            native_dtype=field.dataType.simpleString(),
            dtype_family=identity.family.value,
        )

    base_config = config or AdaptiveCardinalityConfig()
    kernel = _pyspark_hash_kernel(field.dataType, identity)
    effective_hash = _pyspark_hash_configuration(
        base_config.hash_configuration,
        kernel=kernel,
        pyspark_version=str(getattr(pyspark_module, "__version__", "unknown")),
    )
    effective_config = AdaptiveCardinalityConfig(
        exact_unique_threshold=base_config.exact_unique_threshold,
        exact_memory_budget_bytes=base_config.exact_memory_budget_bytes,
        memory_safety_factor=base_config.memory_safety_factor,
        hll_precision=base_config.hll_precision,
        threshold_check_interval=base_config.threshold_check_interval,
        hash_configuration=effective_hash,
    )

    value_column = _spark_column(functions, field.name)
    included = value_column.isNotNull()
    if identity.family is PySparkDtypeFamily.FLOATING:
        included = included & (~functions.isnan(value_column))
    normalized_value = _spark_normalized_value(functions, value_column, identity)

    metrics = dataframe_value.agg(
        functions.count(functions.lit(1)).alias("__dq_total__"),
        functions.sum(functions.when(included, functions.lit(1)).otherwise(functions.lit(0))).alias(
            "__dq_non_null__"
        ),
    ).collect()[0]
    total_row_count = int(metrics["__dq_total__"])
    non_null_count = int(metrics["__dq_non_null__"] or 0)
    null_count = total_row_count - non_null_count

    candidate_limit = effective_config.exact_unique_threshold + 1
    candidate_rows = (
        dataframe_value.where(included)
        .select(normalized_value.alias("__dq_value__"))
        .distinct()
        .limit(candidate_limit)
        .collect()
    )
    exact_values = [row["__dq_value__"] for row in candidate_rows]
    exact_state_bytes = _estimate_exact_values_bytes(
        exact_values,
        safety_factor=effective_config.memory_safety_factor,
    )
    exact_unique_count = len(exact_values)
    unique_exceeded = exact_unique_count > effective_config.exact_unique_threshold
    memory_exceeded = exact_state_bytes > effective_config.exact_memory_budget_bytes

    lineage = _pyspark_lineage(
        dataframe_value,
        backend,
        field=field,
        position=position,
        identity=identity,
        kernel=kernel,
        effective_hash=effective_hash,
        pipeline_metadata=pipeline_metadata,
        pyspark_version=str(getattr(pyspark_module, "__version__", "unknown")),
    )
    execution_parameters = {
        "batch_size": batch_size,
        "pyspark_batch_size": batch_size,
        "pyspark_execution": "distributed_two_stage_sql",
        "exact_candidate_limit": candidate_limit,
        "hll_register_aggregation": "spark_sql_groupby_register_max_rho",
    }

    if not unique_exceeded and not memory_exceeded:
        return AdaptiveCardinalityResult(
            distinct_count=exact_unique_count,
            is_exact=True,
            final_mode=CardinalityMode.EXACT,
            total_row_count=total_row_count,
            non_null_count=non_null_count,
            null_count=null_count,
            config=effective_config,
            promotion=PromotionMetadata(occurred=False),
            peak_exact_unique_count=exact_unique_count,
            peak_exact_state_bytes=exact_state_bytes,
            retained_exact_unique_count=exact_unique_count,
            retained_exact_state_bytes=exact_state_bytes,
            exact_state_representation=kernel.exact_representation,
            hll_zero_register_count=None,
            hll_register_storage_bytes=None,
            raw_hll_estimate=None,
            execution_parameters=execution_parameters,
            lineage=lineage,
            warnings=(),
        )

    reason = _promotion_reason(unique_exceeded, memory_exceeded)
    sketch = _pyspark_hll_sketch(
        dataframe_value,
        functions,
        normalized_value=normalized_value,
        included=included,
        config=effective_config,
    )
    raw_estimate = sketch.estimate()
    return AdaptiveCardinalityResult(
        distinct_count=round(raw_estimate),
        is_exact=False,
        final_mode=CardinalityMode.HLL,
        total_row_count=total_row_count,
        non_null_count=non_null_count,
        null_count=null_count,
        config=effective_config,
        promotion=PromotionMetadata(
            occurred=True,
            reason=reason,
            row_position=None,
            non_null_position=None,
            unique_count=exact_unique_count,
            exact_state_bytes=exact_state_bytes,
        ),
        peak_exact_unique_count=exact_unique_count,
        peak_exact_state_bytes=exact_state_bytes,
        retained_exact_unique_count=0,
        retained_exact_state_bytes=0,
        exact_state_representation=kernel.exact_representation,
        hll_zero_register_count=sketch.zero_register_count,
        hll_register_storage_bytes=len(sketch.registers),
        raw_hll_estimate=raw_estimate,
        execution_parameters=execution_parameters,
        lineage=lineage,
        warnings=(),
    )


def pyspark_dataframe_cardinality(
    dataframe: object,
    *,
    config: AdaptiveCardinalityConfig | None = None,
    batch_size: int = DEFAULT_PYSPARK_BATCH_SIZE,
    backend_metadata: dict[str, object] | None = None,
    pipeline_metadata: dict[str, object] | None = None,
) -> DataFrameCardinalityResult:
    """Profile every Spark SQL column, isolating unsupported nested/unknown dtypes."""

    _, _, dataframe_value, backend = _validate_pyspark_dataframe(dataframe, backend_metadata)
    outcomes: list[ColumnCardinalityOutcome] = []
    row_count: int | None = None
    for position, field in enumerate(dataframe_value.schema.fields):
        try:
            result = pyspark_adaptive_cardinality(
                dataframe_value,
                field.name,
                config=config,
                batch_size=batch_size,
                backend_metadata=backend,
                pipeline_metadata=pipeline_metadata
                or {
                    "entrypoint": "profile_cardinality",
                    "backend_detection": "data_quality.intake.identify_backend",
                    "adapter": "PySparkCardinalityAdapter",
                    "dtype_source": "pyspark.sql.DataFrame.schema",
                    "input_boundary": "dataframe",
                    "column_selection": "all_columns",
                },
            )
        except UnsupportedCardinalityDtypeError as error:
            outcomes.append(
                ColumnCardinalityOutcome(
                    column=field.name,
                    position=position,
                    result=None,
                    error=error.warning,
                )
            )
        else:
            row_count = result.total_row_count if row_count is None else row_count
            outcomes.append(
                ColumnCardinalityOutcome(
                    column=field.name,
                    position=position,
                    result=result,
                )
            )
    if row_count is None:
        row_count = int(dataframe_value.count())
    return DataFrameCardinalityResult(
        backend=dict(backend),
        total_row_count=row_count,
        column_count=len(dataframe_value.schema.fields),
        columns=tuple(outcomes),
    )


def _pyspark_hll_sketch(
    dataframe: Any,
    functions: Any,
    *,
    normalized_value: Any,
    included: Any,
    config: AdaptiveCardinalityConfig,
) -> HashedHyperLogLog:
    precision = config.hll_precision
    suffix_bits = HASH_WIDTH_BITS - precision
    suffix_mask = (1 << suffix_bits) - 1
    signed_seed = _signed_64(config.hash_configuration.seed)
    hash_column = functions.xxhash64(functions.lit(signed_seed), normalized_value)
    register_index = functions.shiftrightunsigned(hash_column, suffix_bits)
    suffix = hash_column.bitwiseAND(functions.lit(suffix_mask))
    rank = functions.when(
        suffix == functions.lit(0),
        functions.lit(suffix_bits + 1),
    ).otherwise(functions.lit(suffix_bits + 1) - functions.length(functions.bin(suffix)))

    register_rows = (
        dataframe.where(included)
        .select(
            register_index.alias("__dq_register__"),
            rank.alias("__dq_rank__"),
        )
        .groupBy("__dq_register__")
        .agg(functions.max("__dq_rank__").alias("__dq_rank__"))
        .collect()
    )
    numpy_module = _load_numpy()
    indexes = numpy_module.asarray(
        [int(row["__dq_register__"]) for row in register_rows],
        dtype=numpy_module.int64,
    )
    ranks = numpy_module.asarray(
        [int(row["__dq_rank__"]) for row in register_rows],
        dtype=numpy_module.int64,
    )
    sketch = HashedHyperLogLog(
        precision=precision,
        hash_configuration=config.hash_configuration,
    )
    sketch.add_register_array(indexes, ranks)
    return sketch


def _pyspark_hash_configuration(
    base: HashConfiguration,
    *,
    kernel: PySparkHashKernel,
    pyspark_version: str,
) -> HashConfiguration:
    return bind_hash_configuration(
        base,
        runtime_id=PYSPARK_HASH_RUNTIME_ID,
        runtime_version=pyspark_version,
        algorithm_id=PYSPARK_HASH_ALGORITHM_ID,
        engine_id=PYSPARK_HASH_ENGINE_ID,
        canonicalisation_version=PYSPARK_CANONICALISATION_VERSION,
        kernel_id=kernel.kernel_id,
    )


def _pyspark_hash_kernel(
    native_dtype: object,
    identity: PySparkDtypeIdentity,
) -> PySparkHashKernel:
    dtype_text = native_dtype.simpleString()  # type: ignore[attr-defined]
    return PySparkHashKernel(
        kernel_id=f"spark-{identity.family.value}-v1:{dtype_text}",
        exact_representation=f"spark_distinct_native_{identity.family.value}",
    )


def _pyspark_lineage(
    dataframe: Any,
    backend_metadata: dict[str, object],
    *,
    field: Any,
    position: int,
    identity: PySparkDtypeIdentity,
    kernel: PySparkHashKernel,
    effective_hash: HashConfiguration,
    pipeline_metadata: dict[str, object] | None,
    pyspark_version: str,
) -> dict[str, object]:
    session_timezone = None
    try:
        session_timezone = dataframe.sparkSession.conf.get("spark.sql.session.timeZone")
    except Exception:
        session_timezone = None
    native_dtype = field.dataType
    return {
        "backend": backend_metadata,
        "column": {
            "name": field.name,
            "position": position,
            "native_dtype": native_dtype.simpleString(),
            "native_dtype_class": (
                f"{type(native_dtype).__module__}.{type(native_dtype).__qualname__}"
            ),
            "nullable": bool(field.nullable),
            "dtype_source": "pyspark.sql.DataFrame.schema",
            **identity.as_dict(),
        },
        "pipeline": pipeline_metadata
        or {
            "entrypoint": "pyspark_adaptive_cardinality",
            "backend_detection": "data_quality.intake.identify_backend",
            "adapter": "PySparkCardinalityAdapter",
            "dtype_source": "pyspark.sql.DataFrame.schema",
            "input_boundary": "dataframe",
        },
        "pyspark_version": pyspark_version,
        "dataframe_type": f"{type(dataframe).__module__}.{type(dataframe).__qualname__}",
        "dtype_family": identity.family.value,
        "hash_kernel": kernel.as_dict(),
        "scalar_fallback_used": False,
        "hash_execution": "spark_sql_xxhash64_distributed",
        "canonicalisation_execution": identity.canonicalisation_strategy,
        "hash_engine": effective_hash.engine_id,
        "hash_algorithm": effective_hash.algorithm_id,
        "spark_session_timezone": session_timezone,
        "distributed_execution": True,
        "adaptive_strategy": "two_stage_exact_then_hll",
    }


def _spark_normalized_value(
    functions: Any, value_column: Any, identity: PySparkDtypeIdentity
) -> Any:
    if identity.family is PySparkDtypeFamily.FLOATING:
        return functions.when(value_column == functions.lit(0.0), functions.lit(0.0)).otherwise(
            value_column
        )
    return value_column


def _resolve_pyspark_field(dataframe: Any, column: object) -> tuple[Any, int]:
    if not isinstance(column, str):
        raise TypeError("PySpark cardinality column selectors must be column-name strings")
    fields = list(dataframe.schema.fields)
    matching = [(position, field) for position, field in enumerate(fields) if field.name == column]
    if not matching:
        raise KeyError(f"PySpark dataframe has no column {column!r}")
    if len(matching) != 1:
        raise ValueError(f"PySpark column label {column!r} is not unique")
    position, field = matching[0]
    return field, position


def _spark_column(functions: Any, name: str) -> Any:
    escaped = name.replace("`", "``")
    return functions.col(f"`{escaped}`")


def _estimate_exact_values_bytes(values: list[object], *, safety_factor: float) -> int:
    if not values:
        return 0
    raw_bytes = sys.getsizeof(values) + sum(sys.getsizeof(value) for value in values)
    return math.ceil(raw_bytes * safety_factor)


def _promotion_reason(unique_exceeded: bool, memory_exceeded: bool) -> PromotionReason:
    if unique_exceeded and memory_exceeded:
        return PromotionReason.BOTH
    if unique_exceeded:
        return PromotionReason.UNIQUE_THRESHOLD
    return PromotionReason.MEMORY_THRESHOLD


def _validate_pyspark_dataframe(
    dataframe: object,
    backend_metadata: dict[str, object] | None,
) -> tuple[Any, Any, Any, dict[str, object]]:
    try:
        pyspark_module = import_module("pyspark")
        sql_module = import_module("pyspark.sql")
        functions = import_module("pyspark.sql.functions")
    except ImportError as error:
        raise ImportError(
            "PySpark cardinality support requires the optional 'pyspark' dependency"
        ) from error
    dataframe_type = getattr(sql_module, "DataFrame")
    if not isinstance(dataframe, dataframe_type):
        # Spark Connect concrete DataFrames may not share the classic class on older 3.x.
        module_name = type(dataframe).__module__
        if not module_name.startswith("pyspark.sql.connect"):
            raise TypeError("PySpark cardinality functions require a pyspark.sql.DataFrame")
    if backend_metadata is None:
        from data_quality.intake import identify_backend

        backend_metadata = identify_backend(dataframe).as_dict()
    return pyspark_module, functions, dataframe, dict(backend_metadata)


def _load_pyspark_types() -> Any:
    try:
        return import_module("pyspark.sql.types")
    except ImportError as error:
        raise ImportError(
            "PySpark cardinality support requires the optional 'pyspark' dependency"
        ) from error


def _require_type(module: Any, name: str) -> type[Any]:
    candidate = getattr(module, name)
    if not isinstance(candidate, type):
        raise TypeError(f"Expected pyspark.sql.types.{name} to be a type")
    return candidate


def _existing_types(module: Any, *names: str) -> tuple[type[Any], ...]:
    return tuple(
        candidate
        for name in names
        if isinstance((candidate := getattr(module, name, None)), type)
    )


def _is_existing_instance(value: object, module: Any, name: str) -> bool:
    candidate = getattr(module, name, None)
    return isinstance(candidate, type) and isinstance(value, candidate)


def _signed_64(value: int) -> int:
    return value if value < (1 << 63) else value - (1 << 64)


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


def _load_numpy() -> Any:
    try:
        return import_module("numpy")
    except ImportError as error:
        raise ImportError("PySpark cardinality support requires NumPy") from error
