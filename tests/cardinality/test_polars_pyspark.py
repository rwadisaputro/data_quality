from __future__ import annotations

import hashlib
import math
import sys
import types
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from data_quality import AdaptiveCardinalityConfig, profile_cardinality
from data_quality.cardinality import (
    CardinalityWarningCode,
    PolarsCardinalityAdapter,
    PolarsDtypeFamily,
    PySparkCardinalityAdapter,
    PySparkDtypeFamily,
    UnsupportedCardinalityDtypeError,
    identify_polars_dtype,
    identify_pyspark_dtype,
)
from data_quality.cardinality.polars_backend import (
    DEFAULT_POLARS_BATCH_SIZE,
    _polars_hash_seeds,
    _validate_batch_size as validate_polars_batch_size,
    polars_adaptive_cardinality,
    polars_dataframe_cardinality,
)
from data_quality.cardinality.pyspark_backend import (
    _signed_64,
    _validate_batch_size as validate_spark_batch_size,
    pyspark_adaptive_cardinality,
    pyspark_dataframe_cardinality,
)
from data_quality.models import BackendName


class FakePolarsDtype:
    def __init__(
        self,
        name: str,
        *,
        integer: bool = False,
        floating: bool = False,
        decimal: bool = False,
        nested: bool = False,
        object_: bool = False,
    ) -> None:
        self.name = name
        self._integer = integer
        self._floating = floating
        self._decimal = decimal
        self._nested = nested
        self._object = object_

    def __str__(self) -> str:
        return self.name

    def base_type(self) -> type[Any]:
        return type(self.name.split("(", 1)[0], (), {})

    def is_integer(self) -> bool:
        return self._integer

    def is_float(self) -> bool:
        return self._floating

    def is_decimal(self) -> bool:
        return self._decimal

    def is_nested(self) -> bool:
        return self._nested

    def is_object(self) -> bool:
        return self._object


class FakePolarsSchema(dict[str, FakePolarsDtype]):
    def names(self) -> list[str]:
        return list(self.keys())


class FakePolarsSeries:
    __module__ = "polars"

    def __init__(self, *args: object, dtype: FakePolarsDtype | None = None) -> None:
        if len(args) == 1:
            self.name = ""
            values = args[0]
        else:
            self.name = str(args[0])
            values = args[1]
        if isinstance(values, np.ndarray):
            self.values = values.tolist()
        else:
            self.values = list(values)  # type: ignore[arg-type]
        self.dtype = dtype or FakePolarsDtype("Object", object_=True)

    def __len__(self) -> int:
        return len(self.values)

    def is_null(self) -> FakePolarsSeries:
        return FakePolarsSeries([value is None for value in self.values], dtype=PL_BOOLEAN)

    def is_nan(self) -> FakePolarsSeries:
        return FakePolarsSeries(
            [isinstance(value, float) and math.isnan(value) for value in self.values],
            dtype=PL_BOOLEAN,
        )

    def fill_null(self, value: object) -> FakePolarsSeries:
        return FakePolarsSeries(
            [value if item is None else item for item in self.values],
            dtype=self.dtype,
        )

    def to_numpy(self) -> np.ndarray:
        if self.dtype.name.startswith("UInt"):
            return np.asarray(self.values, dtype=np.uint64)
        if self.dtype._integer:
            return np.asarray(self.values, dtype=np.int64)
        if self.dtype._floating:
            return np.asarray(self.values, dtype=np.float64)
        if self.dtype.name == "Boolean":
            return np.asarray(self.values, dtype=bool)
        return np.asarray(self.values, dtype=object)

    def filter(self, mask: FakePolarsSeries) -> FakePolarsSeries:
        return FakePolarsSeries(
            [value for value, keep in zip(self.values, mask.values, strict=True) if keep],
            dtype=self.dtype,
        )

    def cast(self, dtype: FakePolarsDtype) -> FakePolarsSeries:
        if dtype.name in {"Int32", "Int64"}:
            values = [int(value) for value in self.values]
        elif dtype.name == "String":
            values = [str(value) for value in self.values]
        else:
            values = list(self.values)
        return FakePolarsSeries(values, dtype=dtype)

    def hash(
        self,
        *,
        seed: int,
        seed_1: int,
        seed_2: int,
        seed_3: int,
    ) -> FakePolarsSeries:
        seeds = (seed, seed_1, seed_2, seed_3)
        values: list[int] = []
        for value in self.values:
            digest = hashlib.blake2b(
                repr((seeds, value)).encode(),
                digest_size=8,
            ).digest()
            values.append(int.from_bytes(digest, "big"))
        return FakePolarsSeries(values, dtype=PL_UINT64)


class FakePolarsDataFrame:
    __module__ = "polars"

    def __init__(
        self,
        data: dict[str, list[object]],
        schema: dict[str, FakePolarsDtype] | None = None,
    ) -> None:
        self._data = {name: list(values) for name, values in data.items()}
        self.schema = FakePolarsSchema(
            schema
            or {
                name: FakePolarsDtype("Object", object_=True)
                for name in self._data
            }
        )

    def __len__(self) -> int:
        if not self._data:
            return 0
        return len(next(iter(self._data.values())))

    def select(self, names: list[str] | str) -> FakePolarsDataFrame:
        selected = [names] if isinstance(names, str) else names
        return FakePolarsDataFrame(
            {name: self._data[name] for name in selected},
            {name: self.schema[name] for name in selected},
        )

    def iter_slices(self, n_rows: int) -> Any:
        for start in range(0, len(self), n_rows):
            yield FakePolarsDataFrame(
                {
                    name: values[start : start + n_rows]
                    for name, values in self._data.items()
                },
                dict(self.schema),
            )

    def get_column(self, name: str) -> FakePolarsSeries:
        return FakePolarsSeries(name, self._data[name], dtype=self.schema[name])

    def item(self, row: int, column: int) -> object:
        name = list(self._data)[column]
        return self._data[name][row]


class FakePolarsLazyFrame:
    __module__ = "polars"

    def __init__(self, dataframe: FakePolarsDataFrame) -> None:
        self.dataframe = dataframe

    def collect_schema(self) -> FakePolarsSchema:
        return self.dataframe.schema

    def select(self, names: Any) -> FakePolarsLazyFrame:
        if not isinstance(names, (list, str)):
            return FakePolarsLazyFrame(
                FakePolarsDataFrame(
                    {"__dq_rows__": [len(self.dataframe)]},
                    {"__dq_rows__": PL_INT64},
                )
            )
        return FakePolarsLazyFrame(self.dataframe.select(names))

    def collect_batches(self, **kwargs: object) -> Any:
        chunk_size = int(kwargs["chunk_size"])
        yield from self.dataframe.iter_slices(chunk_size)

    def collect(self, **kwargs: object) -> FakePolarsDataFrame:
        return self.dataframe


PL_BOOLEAN = FakePolarsDtype("Boolean")
PL_INT64 = FakePolarsDtype("Int64", integer=True)
PL_INT32 = FakePolarsDtype("Int32", integer=True)
PL_UINT64 = FakePolarsDtype("UInt64", integer=True)
PL_FLOAT64 = FakePolarsDtype("Float64", floating=True)
PL_STRING = FakePolarsDtype("String")
PL_BINARY = FakePolarsDtype("Binary")
PL_DECIMAL = FakePolarsDtype("Decimal(10,2)", decimal=True)
PL_DATE = FakePolarsDtype("Date")
PL_DATETIME = FakePolarsDtype("Datetime(us,UTC)")
PL_DURATION = FakePolarsDtype("Duration(us)")
PL_TIME = FakePolarsDtype("Time")
PL_CATEGORY = FakePolarsDtype("Categorical")
PL_ENUM = FakePolarsDtype("Enum")
PL_OBJECT = FakePolarsDtype("Object", object_=True)
PL_NULL = FakePolarsDtype("Null")
PL_LIST = FakePolarsDtype("List(Int64)", nested=True)
PL_OTHER = FakePolarsDtype("Unknown")


def install_fake_polars(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    module = types.ModuleType("polars")
    module.__version__ = "1.99-test"
    module.DataFrame = FakePolarsDataFrame
    module.LazyFrame = FakePolarsLazyFrame
    module.Series = FakePolarsSeries
    module.Boolean = PL_BOOLEAN
    module.Int64 = PL_INT64
    module.Int32 = PL_INT32
    module.UInt64 = PL_UINT64
    module.Float64 = PL_FLOAT64
    module.String = PL_STRING
    module.Binary = PL_BINARY
    module.Date = PL_DATE
    module.Time = PL_TIME
    module.len = lambda: object()
    monkeypatch.setitem(sys.modules, "polars", module)
    return module


@dataclass
class FakeSparkField:
    name: str
    dataType: object
    nullable: bool = True


class FakeSparkSchema:
    def __init__(self, fields: list[FakeSparkField]) -> None:
        self.fields = fields


class SparkDataType:
    def simpleString(self) -> str:
        return self.__class__.__name__.replace("Type", "").lower()


class BooleanType(SparkDataType): pass
class ByteType(SparkDataType): pass
class ShortType(SparkDataType): pass
class IntegerType(SparkDataType): pass
class LongType(SparkDataType): pass
class FloatType(SparkDataType): pass
class DoubleType(SparkDataType): pass
class DecimalType(SparkDataType): pass
class StringType(SparkDataType): pass
class CharType(SparkDataType): pass
class VarcharType(SparkDataType): pass
class BinaryType(SparkDataType): pass
class DateType(SparkDataType): pass
class TimestampType(SparkDataType): pass
class TimestampNTZType(SparkDataType): pass
class DayTimeIntervalType(SparkDataType): pass
class YearMonthIntervalType(SparkDataType): pass
class NullType(SparkDataType): pass
class ArrayType(SparkDataType): pass
class MapType(SparkDataType): pass
class StructType(SparkDataType): pass
class VariantType(SparkDataType): pass


class FakeSparkExpr:
    def __init__(self, func: Any, name: str | None = None) -> None:
        self.func = func
        self.name = name

    def eval(self, row: dict[str, object]) -> object:
        return self.func(row)

    def alias(self, name: str) -> FakeSparkExpr:
        return FakeSparkExpr(self.func, name)

    def isNotNull(self) -> FakeSparkExpr:
        return FakeSparkExpr(lambda row: self.eval(row) is not None)

    def bitwiseAND(self, other: Any) -> FakeSparkExpr:
        other_expr = as_expr(other)
        return FakeSparkExpr(lambda row: int(self.eval(row)) & int(other_expr.eval(row)))

    def otherwise(self, other: Any) -> FakeSparkExpr:
        raise RuntimeError("otherwise only valid on when expressions")

    def __and__(self, other: Any) -> FakeSparkExpr:
        other_expr = as_expr(other)
        return FakeSparkExpr(lambda row: bool(self.eval(row)) and bool(other_expr.eval(row)))

    def __invert__(self) -> FakeSparkExpr:
        return FakeSparkExpr(lambda row: not bool(self.eval(row)))

    def __eq__(self, other: Any) -> FakeSparkExpr:  # type: ignore[override]
        other_expr = as_expr(other)
        return FakeSparkExpr(lambda row: self.eval(row) == other_expr.eval(row))

    def __sub__(self, other: Any) -> FakeSparkExpr:
        other_expr = as_expr(other)
        return FakeSparkExpr(lambda row: int(self.eval(row)) - int(other_expr.eval(row)))


class FakeWhenExpr(FakeSparkExpr):
    def __init__(self, condition: FakeSparkExpr, yes: FakeSparkExpr) -> None:
        super().__init__(lambda row: None)
        self.condition = condition
        self.yes = yes

    def otherwise(self, other: Any) -> FakeSparkExpr:
        no = as_expr(other)
        return FakeSparkExpr(
            lambda row: self.yes.eval(row) if self.condition.eval(row) else no.eval(row)
        )


class FakeAggExpr(FakeSparkExpr):
    def __init__(self, kind: str, expr: FakeSparkExpr, name: str | None = None) -> None:
        super().__init__(expr.func, name)
        self.kind = kind
        self.expr = expr

    def alias(self, name: str) -> FakeAggExpr:
        return FakeAggExpr(self.kind, self.expr, name)


class FakeSparkRow(dict[str, object]):
    pass


class FakeGroupedData:
    def __init__(self, dataframe: FakeSparkDataFrame, key: str) -> None:
        self.dataframe = dataframe
        self.key = key

    def agg(self, expr: FakeAggExpr) -> FakeSparkDataFrame:
        grouped: dict[object, list[dict[str, object]]] = {}
        for row in self.dataframe.rows:
            grouped.setdefault(row[self.key], []).append(row)
        output = []
        for key, rows in grouped.items():
            values = [expr.expr.eval(row) for row in rows]
            output.append({self.key: key, expr.name or "max": max(values)})
        return FakeSparkDataFrame(
            output,
            [
                FakeSparkField(self.key, LongType()),
                FakeSparkField(expr.name or "max", LongType()),
            ],
        )


class FakeSparkConf:
    def get(self, key: str) -> str:
        assert key == "spark.sql.session.timeZone"
        return "UTC"


class FakeSparkSession:
    def __init__(self) -> None:
        self.conf = FakeSparkConf()


class FakeSparkDataFrame:
    __module__ = "pyspark.sql"

    def __init__(self, rows: list[dict[str, object]], fields: list[FakeSparkField]) -> None:
        self.rows = [dict(row) for row in rows]
        self.schema = FakeSparkSchema(fields)
        self.sparkSession = FakeSparkSession()

    def agg(self, *exprs: FakeAggExpr) -> FakeSparkDataFrame:
        row: dict[str, object] = {}
        for expr in exprs:
            values = [expr.expr.eval(item) for item in self.rows]
            if expr.kind == "count":
                result = sum(value is not None for value in values)
            elif expr.kind == "sum":
                result = sum(int(value) for value in values if value is not None)
            else:
                raise AssertionError(expr.kind)
            row[expr.name or expr.kind] = result
        fields = [FakeSparkField(name, LongType()) for name in row]
        return FakeSparkDataFrame([row], fields)

    def collect(self) -> list[FakeSparkRow]:
        return [FakeSparkRow(row) for row in self.rows]

    def where(self, expr: FakeSparkExpr) -> FakeSparkDataFrame:
        return FakeSparkDataFrame(
            [row for row in self.rows if expr.eval(row)],
            list(self.schema.fields),
        )

    def select(self, *exprs: Any) -> FakeSparkDataFrame:
        if len(exprs) == 1 and isinstance(exprs[0], (list, tuple)):
            exprs = tuple(exprs[0])
        expressions = [as_expr(expr) for expr in exprs]
        rows = [
            {
                expr.name or f"_{index}": expr.eval(row)
                for index, expr in enumerate(expressions)
            }
            for row in self.rows
        ]
        fields = [
            FakeSparkField(expr.name or f"_{index}", LongType())
            for index, expr in enumerate(expressions)
        ]
        return FakeSparkDataFrame(rows, fields)

    def distinct(self) -> FakeSparkDataFrame:
        seen: set[tuple[tuple[str, object], ...]] = set()
        output: list[dict[str, object]] = []
        for row in self.rows:
            key = tuple(row.items())
            if key not in seen:
                seen.add(key)
                output.append(row)
        return FakeSparkDataFrame(output, list(self.schema.fields))

    def limit(self, count: int) -> FakeSparkDataFrame:
        return FakeSparkDataFrame(self.rows[:count], list(self.schema.fields))

    def groupBy(self, key: str) -> FakeGroupedData:
        return FakeGroupedData(self, key)

    def count(self) -> int:
        return len(self.rows)


def as_expr(value: Any) -> FakeSparkExpr:
    if isinstance(value, FakeSparkExpr):
        return value
    return FakeSparkExpr(lambda _row: value)


def install_fake_pyspark(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    pyspark = types.ModuleType("pyspark")
    pyspark.__version__ = "4.0-test"
    sql = types.ModuleType("pyspark.sql")
    sql.DataFrame = FakeSparkDataFrame
    types_module = types.ModuleType("pyspark.sql.types")
    for cls in [
        BooleanType, ByteType, ShortType, IntegerType, LongType, FloatType, DoubleType,
        DecimalType, StringType, CharType, VarcharType, BinaryType, DateType,
        TimestampType, TimestampNTZType, DayTimeIntervalType, YearMonthIntervalType,
        NullType, ArrayType, MapType, StructType, VariantType,
    ]:
        setattr(types_module, cls.__name__, cls)

    functions = types.ModuleType("pyspark.sql.functions")
    functions.lit = lambda value: as_expr(value)
    functions.col = lambda name: FakeSparkExpr(lambda row: row[name.strip("`").replace("``", "`")])
    functions.count = lambda expr: FakeAggExpr("count", as_expr(expr))
    functions.sum = lambda expr: FakeAggExpr("sum", as_expr(expr))
    functions.max = lambda expr: FakeAggExpr(
        "max",
        FakeSparkExpr(lambda row: row[expr]) if isinstance(expr, str) else as_expr(expr),
    )
    functions.when = lambda condition, yes: FakeWhenExpr(as_expr(condition), as_expr(yes))
    functions.isnan = lambda expr: FakeSparkExpr(
        lambda row: isinstance(as_expr(expr).eval(row), float)
        and math.isnan(as_expr(expr).eval(row))
    )

    def xxhash64(*exprs: Any) -> FakeSparkExpr:
        expressions = [as_expr(expr) for expr in exprs]
        def evaluate(row: dict[str, object]) -> int:
            payload = repr(tuple(expr.eval(row) for expr in expressions)).encode()
            unsigned = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")
            return unsigned if unsigned < (1 << 63) else unsigned - (1 << 64)
        return FakeSparkExpr(evaluate)

    functions.xxhash64 = xxhash64
    functions.shiftrightunsigned = lambda expr, bits: FakeSparkExpr(
        lambda row: (int(as_expr(expr).eval(row)) & ((1 << 64) - 1)) >> bits
    )
    functions.bin = lambda expr: FakeSparkExpr(lambda row: bin(int(as_expr(expr).eval(row)))[2:])
    functions.length = lambda expr: FakeSparkExpr(lambda row: len(str(as_expr(expr).eval(row))))

    monkeypatch.setitem(sys.modules, "pyspark", pyspark)
    monkeypatch.setitem(sys.modules, "pyspark.sql", sql)
    monkeypatch.setitem(sys.modules, "pyspark.sql.types", types_module)
    monkeypatch.setitem(sys.modules, "pyspark.sql.functions", functions)
    return pyspark


def test_polars_backend_exact_promote_lazy_and_dataframe_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_polars(monkeypatch)
    dataframe = FakePolarsDataFrame(
        {
            "id": [1, 2, 2, 3, None],
            "label": ["a", "b", "a", "c", "c"],
            "bad": [[1], [2], [1], [3], [4]],
        },
        {"id": PL_INT64, "label": PL_STRING, "bad": PL_LIST},
    )

    exact = profile_cardinality(dataframe, "id", batch_size=2)
    assert exact.distinct_count == 3
    assert exact.is_exact
    assert exact.null_count == 1
    assert exact.lineage["backend"]["name"] == "polars"
    assert exact.lineage["column"]["dtype_family"] == "integer"
    assert exact.lineage["hash_execution"] == "polars_series_hash_bulk"

    config = AdaptiveCardinalityConfig(exact_unique_threshold=2)
    approx = profile_cardinality(dataframe, "id", config=config, batch_size=2)
    assert not approx.is_exact
    assert approx.promotion.occurred
    assert isinstance(approx.distinct_count, int)

    report = profile_cardinality(dataframe, batch_size=2)
    assert report.column_count == 3
    assert report.successful_column_count == 2
    assert report.unsupported_column_count == 1
    assert report.columns[2].error.code is CardinalityWarningCode.UNSUPPORTED_DTYPE

    lazy = FakePolarsLazyFrame(dataframe.select(["id", "label"]))
    lazy_result = profile_cardinality(lazy, "label", batch_size=2)
    assert lazy_result.distinct_count == 3
    assert lazy_result.lineage["pipeline"]["adapter"] == "PolarsCardinalityAdapter"
    assert lazy_result.execution_parameters["polars_execution"] == "lazy_collect_batches_streaming"


def test_polars_dtype_kernels_object_fallback_and_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake_polars(monkeypatch)
    cases = [
        (PL_BOOLEAN, PolarsDtypeFamily.BOOLEAN),
        (PL_INT64, PolarsDtypeFamily.INTEGER),
        (PL_FLOAT64, PolarsDtypeFamily.FLOATING),
        (PL_STRING, PolarsDtypeFamily.STRING),
        (PL_BINARY, PolarsDtypeFamily.BINARY),
        (PL_DECIMAL, PolarsDtypeFamily.DECIMAL),
        (PL_DATE, PolarsDtypeFamily.DATE),
        (PL_DATETIME, PolarsDtypeFamily.DATETIME),
        (PL_DURATION, PolarsDtypeFamily.DURATION),
        (PL_TIME, PolarsDtypeFamily.TIME),
        (PL_CATEGORY, PolarsDtypeFamily.CATEGORICAL),
        (PL_ENUM, PolarsDtypeFamily.ENUM),
        (PL_OBJECT, PolarsDtypeFamily.OBJECT),
        (PL_NULL, PolarsDtypeFamily.NULL),
        (PL_LIST, PolarsDtypeFamily.NESTED),
        (PL_OTHER, PolarsDtypeFamily.OTHER),
    ]
    for dtype, family in cases:
        assert identify_polars_dtype(dtype).family is family

    object_df = FakePolarsDataFrame(
        {"x": [1, "1", None]},
        {"x": PL_OBJECT},
    )
    result = polars_adaptive_cardinality(object_df, "x")
    assert result.distinct_count == 2
    assert result.warnings[0].code is CardinalityWarningCode.SCALAR_FALLBACK

    unsupported_scalar = FakePolarsDataFrame({"x": [object()]}, {"x": PL_OBJECT})
    report = polars_dataframe_cardinality(unsupported_scalar)
    assert report.unsupported_column_count == 1
    assert report.columns[0].error.code is CardinalityWarningCode.UNSUPPORTED_SCALAR

    with pytest.raises(UnsupportedCardinalityDtypeError):
        polars_adaptive_cardinality(
            FakePolarsDataFrame({"x": [[1]]}, {"x": PL_LIST}),
            "x",
        )
    with pytest.raises(KeyError):
        polars_adaptive_cardinality(object_df, "missing")
    with pytest.raises(TypeError):
        polars_adaptive_cardinality(object_df, 0)
    assert PolarsCardinalityAdapter.backend_name is BackendName.POLARS
    assert DEFAULT_POLARS_BATCH_SIZE > 0
    assert len(set(_polars_hash_seeds(123))) == 4
    with pytest.raises(TypeError):
        validate_polars_batch_size(True)
    with pytest.raises(TypeError):
        validate_polars_batch_size(1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        validate_polars_batch_size(0)


def test_pyspark_exact_hll_dtype_dispatch_and_report(monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake_pyspark(monkeypatch)
    fields = [
        FakeSparkField("id", LongType()),
        FakeSparkField("value", DoubleType()),
        FakeSparkField("nested", ArrayType()),
    ]
    dataframe = FakeSparkDataFrame(
        [
            {"id": 1, "value": 0.0, "nested": [1]},
            {"id": 2, "value": -0.0, "nested": [2]},
            {"id": 2, "value": float("nan"), "nested": [1]},
            {"id": 3, "value": 2.0, "nested": [3]},
            {"id": None, "value": None, "nested": None},
        ],
        fields,
    )

    exact = profile_cardinality(dataframe, "id")
    assert exact.is_exact
    assert exact.distinct_count == 3
    assert exact.null_count == 1
    assert exact.lineage["backend"]["name"] == "pyspark"
    assert exact.lineage["adaptive_strategy"] == "two_stage_exact_then_hll"

    float_result = profile_cardinality(dataframe, "value")
    assert float_result.distinct_count == 2
    assert float_result.non_null_count == 3

    approx = profile_cardinality(
        dataframe,
        "id",
        config=AdaptiveCardinalityConfig(exact_unique_threshold=2, hll_precision=10),
    )
    assert not approx.is_exact
    assert approx.promotion.occurred
    assert approx.raw_hll_estimate is not None
    assert isinstance(approx.distinct_count, int)
    assert approx.lineage["hash_execution"] == "spark_sql_xxhash64_distributed"

    report = profile_cardinality(dataframe)
    assert report.column_count == 3
    assert report.successful_column_count == 2
    assert report.unsupported_column_count == 1
    assert report.columns[2].error.code is CardinalityWarningCode.UNSUPPORTED_DTYPE
    assert PySparkCardinalityAdapter.backend_name is BackendName.PYSPARK


def test_pyspark_dtype_families_validation_and_missing_column(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_pyspark(monkeypatch)
    cases = [
        (BooleanType(), PySparkDtypeFamily.BOOLEAN),
        (ByteType(), PySparkDtypeFamily.INTEGER),
        (ShortType(), PySparkDtypeFamily.INTEGER),
        (IntegerType(), PySparkDtypeFamily.INTEGER),
        (LongType(), PySparkDtypeFamily.INTEGER),
        (FloatType(), PySparkDtypeFamily.FLOATING),
        (DoubleType(), PySparkDtypeFamily.FLOATING),
        (DecimalType(), PySparkDtypeFamily.DECIMAL),
        (StringType(), PySparkDtypeFamily.STRING),
        (CharType(), PySparkDtypeFamily.STRING),
        (VarcharType(), PySparkDtypeFamily.STRING),
        (BinaryType(), PySparkDtypeFamily.BINARY),
        (DateType(), PySparkDtypeFamily.DATE),
        (TimestampType(), PySparkDtypeFamily.TIMESTAMP),
        (TimestampNTZType(), PySparkDtypeFamily.TIMESTAMP_NTZ),
        (DayTimeIntervalType(), PySparkDtypeFamily.DAYTIME_INTERVAL),
        (YearMonthIntervalType(), PySparkDtypeFamily.YEARMONTH_INTERVAL),
        (NullType(), PySparkDtypeFamily.NULL),
        (StructType(), PySparkDtypeFamily.NESTED),
        (VariantType(), PySparkDtypeFamily.OTHER),
    ]
    for dtype, family in cases:
        assert identify_pyspark_dtype(dtype).family is family

    dataframe = FakeSparkDataFrame([{"id": 1}], [FakeSparkField("id", LongType())])
    with pytest.raises(KeyError):
        pyspark_adaptive_cardinality(dataframe, "missing")
    with pytest.raises(TypeError):
        pyspark_adaptive_cardinality(dataframe, 0)
    with pytest.raises(UnsupportedCardinalityDtypeError):
        pyspark_adaptive_cardinality(
            FakeSparkDataFrame([{"x": [1]}], [FakeSparkField("x", ArrayType())]),
            "x",
        )
    assert _signed_64(5) == 5
    assert _signed_64((1 << 64) - 1) == -1
    with pytest.raises(TypeError):
        validate_spark_batch_size(True)
    with pytest.raises(TypeError):
        validate_spark_batch_size(1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        validate_spark_batch_size(0)


def test_pyspark_all_unsupported_uses_count(monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake_pyspark(monkeypatch)
    dataframe = FakeSparkDataFrame(
        [{"nested": [1]}, {"nested": [2]}],
        [FakeSparkField("nested", ArrayType())],
    )
    report = pyspark_dataframe_cardinality(dataframe)
    assert report.total_row_count == 2
    assert report.unsupported_column_count == 1


def test_polars_all_kernel_canonicalization_and_hash_edges(monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake_polars(monkeypatch)
    from data_quality.cardinality.hashing import HashConfiguration
    from data_quality.cardinality.polars_backend import (
        PolarsDtypeIdentity,
        _canonicalize_polars_series,
        _hash_polars_tokens,
        _polars_hash_configuration,
        _polars_hash_kernel,
    )

    kernel_cases = [
        (PL_BOOLEAN, PolarsDtypeFamily.BOOLEAN, [True, False]),
        (PL_INT64, PolarsDtypeFamily.INTEGER, [1, 2]),
        (PL_FLOAT64, PolarsDtypeFamily.FLOATING, [0.0, -0.0, 2.5]),
        (PL_STRING, PolarsDtypeFamily.STRING, ["a", "b"]),
        (PL_BINARY, PolarsDtypeFamily.BINARY, [b"a", b"b"]),
        (PL_DECIMAL, PolarsDtypeFamily.DECIMAL, [1, 2]),
        (PL_DATE, PolarsDtypeFamily.DATE, [1, 2]),
        (PL_DATETIME, PolarsDtypeFamily.DATETIME, [1, 2]),
        (PL_DURATION, PolarsDtypeFamily.DURATION, [1, 2]),
        (PL_TIME, PolarsDtypeFamily.TIME, [1, 2]),
        (PL_CATEGORY, PolarsDtypeFamily.CATEGORICAL, ["a", "b"]),
        (PL_ENUM, PolarsDtypeFamily.ENUM, ["a", "b"]),
        (PL_NULL, PolarsDtypeFamily.NULL, [None, None]),
        (PL_OBJECT, PolarsDtypeFamily.OBJECT, [1, "1"]),
    ]
    for dtype, family, values in kernel_cases:
        identity = PolarsDtypeIdentity(
            family=family,
            canonicalisation_strategy=f"test_{family.value}",
            is_nested=False,
        )
        kernel = _polars_hash_kernel(dtype, identity)
        tokens, positions, null_count = _canonicalize_polars_series(
            sys.modules["polars"],
            FakePolarsSeries("x", values, dtype=dtype),
            identity=identity,
            native_dtype=dtype,
            column="x",
            row_offset=10,
        )
        assert positions.ndim == 1
        assert null_count == (2 if family is PolarsDtypeFamily.NULL else 0)
        configuration = _polars_hash_configuration(
            HashConfiguration(seed=7),
            kernel=kernel,
            polars_version="1.99-test",
        )
        hashes = _hash_polars_tokens(
            sys.modules["polars"], tokens, kernel=kernel, configuration=configuration
        )
        assert hashes.dtype == np.uint64
        assert len(hashes) == len(tokens)

    float_identity = PolarsDtypeIdentity(
        PolarsDtypeFamily.FLOATING, "test_float", False
    )
    tokens, positions, null_count = _canonicalize_polars_series(
        sys.modules["polars"],
        FakePolarsSeries("x", [1.0, float("nan"), None], dtype=PL_FLOAT64),
        identity=float_identity,
        native_dtype=PL_FLOAT64,
        column="x",
        row_offset=0,
    )
    assert tokens.tolist() == [1.0]
    assert positions.tolist() == [0]
    assert null_count == 2

    kernel = _polars_hash_kernel(PL_INT64, identify_polars_dtype(PL_INT64))
    config = _polars_hash_configuration(
        HashConfiguration(), kernel=kernel, polars_version="1.99-test"
    )
    with pytest.raises(ValueError):
        _hash_polars_tokens(
            sys.modules["polars"], np.zeros((1, 1)), kernel=kernel, configuration=config
        )
    empty = _hash_polars_tokens(
        sys.modules["polars"], np.asarray([], dtype=np.int64), kernel=kernel, configuration=config
    )
    assert empty.dtype == np.uint64 and empty.size == 0


def test_polars_empty_frames_lazy_fallback_and_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    module = install_fake_polars(monkeypatch)
    import data_quality.cardinality.polars_backend as polars_impl

    empty = FakePolarsDataFrame({}, {})
    report = polars_dataframe_cardinality(empty)
    assert report.total_row_count == 0
    assert report.column_count == 0

    class LenExpr:
        def alias(self, _name: str) -> LenExpr:
            return self

    module.len = lambda: LenExpr()
    lazy_empty = FakePolarsLazyFrame(empty)
    lazy_report = polars_dataframe_cardinality(lazy_empty)
    assert lazy_report.total_row_count == 0
    assert lazy_report.column_count == 0

    unsupported = FakePolarsDataFrame(
        {"nested": [[1], [2], [3]]}, {"nested": PL_LIST}
    )
    report = polars_dataframe_cardinality(unsupported)
    assert report.total_row_count == 3
    assert report.unsupported_column_count == 1

    lazy_unsupported = FakePolarsLazyFrame(unsupported)
    report = polars_dataframe_cardinality(lazy_unsupported)
    assert report.total_row_count == 3

    lazy = FakePolarsLazyFrame(FakePolarsDataFrame({"x": [1, 2, 3]}, {"x": PL_INT64}))
    monkeypatch.setattr(FakePolarsLazyFrame, "collect_batches", None)
    result = polars_adaptive_cardinality(lazy, "x", batch_size=2)
    assert result.distinct_count == 3

    assert polars_impl._schema_names({"a": PL_INT64}) == ["a"]

    class NoBaseType:
        pass

    assert polars_impl._polars_dtype_name(NoBaseType()) == "NoBaseType"
    with pytest.raises(TypeError):
        polars_impl._validate_polars_frame(object(), {})


def test_polars_single_column_unsupported_scalar_and_import_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_polars(monkeypatch)
    import data_quality.cardinality.polars_backend as polars_impl
    from data_quality.cardinality.diagnostics import UnsupportedCardinalityValueError

    with pytest.raises(UnsupportedCardinalityValueError):
        polars_adaptive_cardinality(
            FakePolarsDataFrame({"x": [object()]}, {"x": PL_OBJECT}), "x"
        )

    original_import = polars_impl.import_module

    def fail_polars(name: str) -> Any:
        if name == "polars":
            raise ImportError("missing polars")
        return original_import(name)

    monkeypatch.setattr(polars_impl, "import_module", fail_polars)
    with pytest.raises(ImportError, match="optional 'polars'"):
        polars_impl._load_polars()

    def fail_numpy(name: str) -> Any:
        if name == "numpy":
            raise ImportError("missing numpy")
        return original_import(name)

    monkeypatch.setattr(polars_impl, "import_module", fail_numpy)
    with pytest.raises(ImportError, match="requires NumPy"):
        polars_impl._load_numpy()


def test_pyspark_edge_helpers_and_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake_pyspark(monkeypatch)
    import data_quality.cardinality.pyspark_backend as spark_impl
    from data_quality.cardinality.adaptive import PromotionReason

    duplicate = FakeSparkDataFrame(
        [{"x": 1}], [FakeSparkField("x", LongType()), FakeSparkField("x", LongType())]
    )
    with pytest.raises(ValueError, match="not unique"):
        spark_impl._resolve_pyspark_field(duplicate, "x")

    assert spark_impl._estimate_exact_values_bytes([], safety_factor=1.25) == 0
    assert spark_impl._promotion_reason(True, True) is PromotionReason.BOTH
    assert spark_impl._promotion_reason(True, False) is PromotionReason.UNIQUE_THRESHOLD
    assert spark_impl._promotion_reason(False, True) is PromotionReason.MEMORY_THRESHOLD

    with pytest.raises(TypeError):
        spark_impl._validate_pyspark_dataframe(object(), {})

    ConnectFrame = type("ConnectFrame", (), {"__module__": "pyspark.sql.connect.dataframe"})
    connect = ConnectFrame()
    _, _, validated, backend = spark_impl._validate_pyspark_dataframe(
        connect, {"name": "pyspark"}
    )
    assert validated is connect
    assert backend["name"] == "pyspark"

    with pytest.raises(TypeError, match="Expected pyspark.sql.types"):
        spark_impl._require_type(types.SimpleNamespace(Bad=123), "Bad")

    original_import = spark_impl.import_module

    def fail_pyspark(name: str) -> Any:
        if name == "pyspark":
            raise ImportError("missing pyspark")
        return original_import(name)

    monkeypatch.setattr(spark_impl, "import_module", fail_pyspark)
    with pytest.raises(ImportError, match="optional 'pyspark'"):
        spark_impl._validate_pyspark_dataframe(connect, {})

    def fail_types(name: str) -> Any:
        if name == "pyspark.sql.types":
            raise ImportError("missing types")
        return original_import(name)

    monkeypatch.setattr(spark_impl, "import_module", fail_types)
    with pytest.raises(ImportError, match="optional 'pyspark'"):
        spark_impl._load_pyspark_types()

    def fail_numpy(name: str) -> Any:
        if name == "numpy":
            raise ImportError("missing numpy")
        return original_import(name)

    monkeypatch.setattr(spark_impl, "import_module", fail_numpy)
    with pytest.raises(ImportError, match="requires NumPy"):
        spark_impl._load_numpy()


def test_pyspark_timezone_lookup_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake_pyspark(monkeypatch)
    dataframe = FakeSparkDataFrame([{"id": 1}], [FakeSparkField("id", LongType())])

    class BrokenConf:
        def get(self, _key: str) -> str:
            raise RuntimeError("timezone unavailable")

    dataframe.sparkSession.conf = BrokenConf()
    result = pyspark_adaptive_cardinality(dataframe, "id")
    assert result.lineage["spark_session_timezone"] is None
