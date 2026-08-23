"""Contracts and shared helpers for native physical-schema discoverers"""

from __future__ import annotations

from operator import index
from typing import Protocol

from data_quality.models import BackendIdentity, BackendName
from data_quality.schema_models import (
    DataFramePhysicalSchema,
    NativeColumnSchema,
    RowCountMode,
)



def non_negative_integer(value: object, 
                         *, 
                         label: str) -> int:
    """
    Return an integer supplied by a backend and reject invalid dimensions
    """

    if isinstance(value, bool):
        raise TypeError(f"{label} must be an integer, not boolean")

    try:
        integer_value = index(value)
    except TypeError as error:
        raise TypeError(f"{label} must be an integer") from error

    if integer_value < 0:
        raise ValueError(f"{label} must be non-negative")
    
    return integer_value

def object_type_name(value: object) -> str:
    """
    Return the fully qualified type name of a metadata object
    """

    value_type = type(value)

    return f"{value_type.__module__}.{value_type.__qualname__}"

def column_schema(*,
                  position: int,
                  name: object,
                  native_data_type: object,
                  native_data_type_string: str,
                  nullable: bool | None) -> NativeColumnSchema:
    """
    Build one column definition while retaining exact native objects
    """

    return NativeColumnSchema(
        position = position,
        name = name,
        name_string = str(name),
        native_data_type = native_data_type,
        native_data_type_string = native_data_type_string,
        native_data_type_class = object_type_name(native_data_type),
        nullable = nullable,
    )

class PhysicalSchemaDiscoverer(Protocol):
    """
    Structural interface implemented by backend schema discoverer
    """

    backend_name: BackendName

    def discover(self,
                 dataframe: object,
                 backend: BackendIdentity,
                 row_count_mode: RowCountMode) -> DataFramePhysicalSchema:
        """
        Discover native column definitions and dimensions
        """