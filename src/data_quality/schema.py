"""Public entry points for native physical-schema discovery"""

from __future__ import annotations

from typing import TypeVar

from data_quality.backends.registry import BackendRegistry
from data_quality.intake import DEFAULT_BACKEND_REGISTRY, resolve_dataframe_input
from data_quality.models import DataFrameInput
from data_quality.schema_discovery.builtin import builtin_physical_schema_discoverers
from data_quality.schema_discovery.registry import PhysicalSchemaRegistry
from data_quality.schema_models import DataFramePhysicalSchema, RowCountMode



DataFrameType = TypeVar("DataFrameType")
DEFAULT_PHYSICAL_SCHEMA_REGISTRY = PhysicalSchemaRegistry(
    builtin_physical_schema_discoverers()
)

def discover_physical_schema_from_input(dataframe_input: DataFrameInput[object],
                                        *,
                                        row_count_mode: RowCountMode = RowCountMode.METADATA_ONLY,
                                        registry: PhysicalSchemaRegistry = DEFAULT_PHYSICAL_SCHEMA_REGISTRY) -> DataFramePhysicalSchema:
    """
    Discover schema from an already resolved dataframe input

    - The default mode never launches a query solely to count rows
    - Use `RowCountMode.EXACT` when an exact height is required for a Polars `LazyFrame` 
      or a bounded Spark `DataFrame`
    """

    if not isinstance(dataframe_input, DataFrameInput):
        raise TypeError(
            "Expected `DataFrameInput`, call `resolve_dataframe_input()` first or "
            "pass the native dataframe to `discover_physical_schema()`"
        )
    if not isinstance(row_count_mode, RowCountMode):
        raise TypeError("row_count_mode must be a RowCountMode value.")

    return registry.discover(dataframe_input, row_count_mode=row_count_mode)

def discover_physical_schema(dataframe: DataFrameType,
                             *,
                             row_count_mode: RowCountMode = RowCountMode.METADATA_ONLY,
                             backend_registry: BackendRegistry = DEFAULT_BACKEND_REGISTRY,
                             schema_registry: PhysicalSchemaRegistry = DEFAULT_PHYSICAL_SCHEMA_REGISTRY) -> DataFramePhysicalSchema:
    """
    Resolve a native dataframe and discover its physical schema

    - The exact original dataframe is retained by the temporary input context, no conversion or copy is made
    """

    dataframe_input = resolve_dataframe_input(
        dataframe, 
        registry = backend_registry
    )

    return discover_physical_schema_from_input(
        dataframe_input,
        row_count_mode = row_count_mode,
        registry = schema_registry,
    )