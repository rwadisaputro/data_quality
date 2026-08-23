# Dataframe intake, backend identification and physical-schema discovery

This package implements the first three stages of the profiling pipeline:

```text
User supplies dataframe
    ↓
Backend identification
    ↓
Native physical-schema discovery
```

It accepts pandas, Polars, and Spark SQL dataframes, preserves the exact object
supplied by the caller, identifies its native execution engine, and discovers:

- Columns in original ordinal order, including duplicate pandas labels.
- Exact backend-native data type objects and report-friendly type strings.
- Spark top-level field nullability.
- Column count for every supported dataframe.
- Exact row count when already available as frame metadata or when explicitly
  requested through a native count query.

No dataframe is converted to pandas or to another engine. Schema discovery does
not inspect individual values, infer semantic types, or normalize native data
types; those are later pipeline stages.

## Supported inputs and discovery behavior

| Input | Native schema API | Metadata-only shape | Exact row-count behavior |
| --- | --- | --- | --- |
| `pandas.DataFrame` | `DataFrame.dtypes` | `(rows, columns)` | Already available from `DataFrame.shape` |
| `polars.DataFrame` | `DataFrame.schema` | `(rows, columns)` | Already available from `DataFrame.shape` |
| `polars.LazyFrame` | `LazyFrame.collect_schema()` | `(None, columns)` | Executes a scalar `pl.len()` query |
| Bounded `pyspark.sql.DataFrame` | `DataFrame.schema` | `(None, columns)` | Executes `DataFrame.count()` |
| Streaming `pyspark.sql.DataFrame` | `DataFrame.schema` | `(None, columns)` | Reported as unbounded; `count()` is not called |

Spark classic and Spark Connect remain separate connection modes in backend
metadata but use the same public Spark SQL schema adapter.

`LazyFrame.collect_schema()` may itself resolve file or remote metadata. It does
not collect the frame's rows, but Polars documents that schema resolution can be
costly for remote sources.

## Installation during development

Install only the backend extras the environment needs:

```bash
python -m pip install -e '.[pandas,test]'
```

For a multi-backend continuous-integration job:

```bash
python -m pip install -e '.[all-backends,test]'
```

The distribution and import names in `pyproject.toml` remain placeholders until
the library's final name is selected.

## Public usage

The convenience function performs intake, backend identification, and physical
schema discovery:

```python
import pandas as pandas_module

from data_quality import discover_physical_schema


user_dataframe = pandas_module.DataFrame(
    {
        "customer_identifier": pandas_module.Series([1, 2], dtype = "Int64"),
        "email_address": pandas_module.Series(
            ["first@example.com", "second@example.com"],
            dtype = "string",
        ),
    }
)

physical_schema = discover_physical_schema(user_dataframe)

assert physical_schema.shape == (2, 2)
print(physical_schema.as_dict())
```

Report-friendly output:

```python
{
    "backend": {
        "name": "pandas",
        "dataframe_api": "pandas",
        "frame_kind": "dataframe",
        "evaluation_mode": "eager",
        "distribution_mode": "local",
        "connection_mode": "in_process",
        "dataframe_type": "pandas.DataFrame",
        "client_library_version": "...",
    },
    "dimensions": {
        "row_count": 2,
        "column_count": 2,
        "shape": [2, 2],
        "row_count_source": "frame_metadata",
        "has_exact_row_count": True,
    },
    "columns": [
        {
            "position": 0,
            "name": "customer_identifier",
            "native_data_type": "Int64",
            "native_data_type_class": "pandas.core.arrays.integer.Int64Dtype",
            "nullable": None,
        },
        {
            "position": 1,
            "name": "email_address",
            "native_data_type": "string",
            "native_data_type_class": "pandas.core.arrays.string_.StringDtype",
            "nullable": None,
        },
    ],
}
```

The exact module paths in `native_data_type_class` vary by backend version, so
they should be treated as diagnostics rather than stable normalization keys.

## Staged usage without repeated identification

The pipeline can retain the already resolved input context:

```python
from data_quality import (
    discover_physical_schema_from_input,
    resolve_dataframe_input,
)


dataframe_input = resolve_dataframe_input(user_dataframe)
physical_schema = discover_physical_schema_from_input(dataframe_input)

assert dataframe_input.dataframe is user_dataframe
assert physical_schema.backend is dataframe_input.backend
```

Backend identification still performs no row count, collection, conversion,
schema access, query execution, or copy. Only the third-stage function accesses
native schema metadata.

## Exact versus metadata-only row counts

The default is deliberately safe for lazy and distributed frames:

```python
from data_quality import RowCountMode, discover_physical_schema


physical_schema = discover_physical_schema(
    spark_dataframe,
    row_count_mode = RowCountMode.METADATA_ONLY,
)

assert physical_schema.shape == (None, len(spark_dataframe.columns))
assert physical_schema.dimensions.row_count_source.value == "not_computed"
```

Request an exact finite height explicitly:

```python
physical_schema = discover_physical_schema(
    spark_dataframe,
    row_count_mode = RowCountMode.EXACT,
)
```

For a bounded frame with partitions `1, ..., P`, Spark computes the exact row
count conceptually as:

```text
row_count = partition_row_count_1 + ... + partition_row_count_P
```

This can require scanning the whole logical plan. The resulting rows are not
collected into Python, but the distributed work is real. A future structural
profiling stage can reuse or combine this aggregation with other metrics to
avoid an unnecessary extra Spark job.

`RowCountSource` makes every state explicit:

| Value | Meaning |
| --- | --- |
| `frame_metadata` | Exact row count was already stored by the eager frame |
| `executed_query` | An exact native count query was executed |
| `not_computed` | No row query was executed under metadata-only mode |
| `unbounded` | The Spark DataFrame represents a streaming source |

## Native type preservation

Each `NativeColumnSchema` carries both:

- `native_data_type`: the exact pandas dtype, Polars `DataType`, or PySpark
  `DataType` object.
- `native_data_type_string`: a stable, report-friendly representation supplied
  by the backend.

The complete native schema is also retained as `physical_schema.native_schema`:

- pandas: the `Series` returned by `DataFrame.dtypes`.
- Polars: the ordered `Schema` mapping.
- PySpark: the `StructType` returned by `DataFrame.schema`.

This is important for the next normalization stage. For example, a Spark
`array<struct<...>>`, a timezone-aware pandas dtype, and a parameterized Polars
decimal type should not be reduced to broad labels such as `object`, `list`, or
`number` before their parameters are inspected.

Pandas `object` remains a native physical dtype, not a semantic conclusion. It
may contain strings, mixed Python objects, or another logical domain; later
semantic inference must examine values using a bounded profiling strategy.

## Extension contracts

`BackendRegistry` and `PhysicalSchemaRegistry` are immutable. Calling
`with_detector()` or `with_discoverer()` returns a new registry and leaves the
process-wide defaults unchanged. Duplicate detector names and duplicate schema
discoverers for the same backend are rejected during registry construction.

Unexpected backend adapter failures are wrapped in
`PhysicalSchemaDiscoveryError`, with the original exception retained as
`__cause__`. Error messages contain backend/type metadata and never dataframe
values.

## Verification

Run the dependency-light suite with Python's standard library:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
python -m compileall -q src tests
```

The suite covers pandas extension dtypes, duplicate/non-string labels, eager and
lazy Polars behavior, Spark metadata-only and exact modes, streaming Spark
frames, immutable registry failures, JSON-safe reporting, and the original
backend-identification guarantees. Real Polars and PySpark type-routing checks
activate automatically when those optional packages are installed.
