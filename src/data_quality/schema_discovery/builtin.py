"""Native physical-schema discoverers for pandas, Polars and PySpark"""

from __future__ import annotations

from importlib import import_module
from typing import Any, cast

from data_quality.models import BackendIdentity, BackendName, FrameKind
from data_quality.schema_discovery.base import column_schema, non_negative_integer
from data_quality.schema_models import (
    DataFrameDimensions,
    DataFramePhysicalSchema,
    RowCountMode,
    RowCountSource,
)



def _two_dimensional_shape(value: object) -> tuple[int, int]:
    """
    Validate and return a backend's two-dimensional shape tuple
    """

    if not isinstance(value, tuple) or len(value) != 2:
        raise TypeError("A dataframe shape must be a two-item tuple")

    return (
        non_negative_integer(value[0], label = "Row count"),
        non_negative_integer(value[1], label = "Column count"),
    )

class PandasPhysicalSchemaDiscoverer:
    """
    Discover pandas dtypes and metadata-backed dimensions
    """

    backend_name = BackendName.PANDAS

    def discover(self,
                 dataframe: object,
                 backend: BackendIdentity,
                 row_count_mode: RowCountMode) -> DataFramePhysicalSchema:
        # pandas shape is already finite frame metadata
        del row_count_mode
        pandas_dataframe = cast(Any, dataframe)
        native_schema = pandas_dataframe.dtypes
        column_names = tuple(pandas_dataframe.columns)
        native_data_types = tuple(native_schema)
        row_count, column_count = _two_dimensional_shape(pandas_dataframe.shape)

        if len(column_names) != column_count or len(native_data_types) != column_count:
            raise ValueError("Pandas columns, dtypes and shape disagree")

        columns = tuple(
            column_schema(
                position = position,
                name = name,
                native_data_type = native_data_type,
                native_data_type_string = str(native_data_type),
                nullable = None,
            )
            for position, (name, native_data_type) in enumerate(
                zip(column_names, native_data_types, strict = True)
            )
        )

        return DataFramePhysicalSchema(
            backend = backend,
            native_schema = native_schema,
            columns = columns,
            dimensions = DataFrameDimensions(
                row_count = row_count,
                column_count = column_count,
                row_count_source = RowCountSource.FRAME_METADATA,
            ),
        )

class PolarsPhysicalSchemaDiscoverer:
    """Discover eager or lazy Polars schemas without materialising user data"""

    backend_name = BackendName.POLARS
    _ROW_COUNT_COLUMN = "__data_quality_internal_row_count__"

    def discover(self,
                 dataframe: object,
                 backend: BackendIdentity,
                 row_count_mode: RowCountMode) -> DataFramePhysicalSchema:
        polars_frame = cast(Any, dataframe)
        native_schema: Any
        row_count: int | None
        row_count_source: RowCountSource
        shape_column_count: int

        if backend.frame_kind is FrameKind.DATAFRAME:
            native_schema = polars_frame.schema
            row_count, shape_column_count = _two_dimensional_shape(
                polars_frame.shape
            )
            row_count_source = RowCountSource.FRAME_METADATA
        elif backend.frame_kind is FrameKind.LAZY_FRAME:
            native_schema = polars_frame.collect_schema()
            shape_column_count = len(native_schema)

            if row_count_mode is RowCountMode.EXACT:
                polars_module = cast(Any, import_module("polars"))
                row_count_expression = polars_module.len().alias(
                    self._ROW_COUNT_COLUMN
                )
                row_count_result = polars_frame.select(
                    row_count_expression
                ).collect()
                row_count = non_negative_integer(
                    row_count_result.item(),
                    label = "Row count",
                )
                row_count_source = RowCountSource.EXECUTED_QUERY
            else:
                row_count = None
                row_count_source = RowCountSource.NOT_COMPUTED
        else:
            raise ValueError(f"Unsupported Polars frame kind: {backend.frame_kind.value}")

        schema_items = tuple(native_schema.items())
        column_count = non_negative_integer(len(schema_items), label = "Column count")
        if shape_column_count != column_count:
            raise ValueError("Polars schema and shape disagree on column count")

        columns = tuple(
            column_schema(
                position = position,
                name = name,
                native_data_type = native_data_type,
                native_data_type_string = str(native_data_type),
                nullable = None,
            )
            for position, (name, native_data_type) in enumerate(schema_items)
        )

        return DataFramePhysicalSchema(
            backend = backend,
            native_schema = native_schema,
            columns = columns,
            dimensions = DataFrameDimensions(
                row_count = row_count,
                column_count = column_count,
                row_count_source = row_count_source,
            ),
        )

class PySparkPhysicalSchemaDiscoverer:
    """
    Discover Spark SQL fields and optionally execute an exact batch count
    """

    backend_name = BackendName.PYSPARK

    def discover(self,
                 dataframe: object,
                 backend: BackendIdentity,
                 row_count_mode: RowCountMode) -> DataFramePhysicalSchema:
        spark_dataframe = cast(Any, dataframe)
        native_schema = spark_dataframe.schema
        native_fields = tuple(native_schema.fields)
        column_count = non_negative_integer(len(native_fields), label = "Column count")
        row_count: int | None
        row_count_source: RowCountSource

        columns = tuple(
            column_schema(
                position = position,
                name = native_field.name,
                native_data_type = native_field.dataType,
                native_data_type_string = native_field.dataType.simpleString(),
                nullable = bool(native_field.nullable),
            )
            for position, native_field in enumerate(native_fields)
        )

        if bool(spark_dataframe.isStreaming):
            row_count = None
            row_count_source = RowCountSource.UNBOUNDED
        elif row_count_mode is RowCountMode.EXACT:
            row_count = non_negative_integer(
                spark_dataframe.count(),
                label = "Row count",
            )
            row_count_source = RowCountSource.EXECUTED_QUERY
        else:
            row_count = None
            row_count_source = RowCountSource.NOT_COMPUTED

        return DataFramePhysicalSchema(
            backend = backend,
            native_schema = native_schema,
            columns=columns,
            dimensions=DataFrameDimensions(
                row_count = row_count,
                column_count = column_count,
                row_count_source = row_count_source,
            ),
        )

def builtin_physical_schema_discoverers() -> tuple[PandasPhysicalSchemaDiscoverer,
                                                   PolarsPhysicalSchemaDiscoverer,
                                                   PySparkPhysicalSchemaDiscoverer]:
    """
    Build the immutable default schema-discoverer sequence
    """

    return (
        PandasPhysicalSchemaDiscoverer(),
        PolarsPhysicalSchemaDiscoverer(),
        PySparkPhysicalSchemaDiscoverer(),
    )