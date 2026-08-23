"""Public entry points for the first two profiling-pipeline stages"""

from __future__ import annotations

from typing import TypeVar

from data_quality.backends.builtin import builtin_backend_detectors
from data_quality.backends.registry import BackendRegistry
from data_quality.models import BackendIdentity, DataFrameInput



DataFrameType = TypeVar("DataFrameType")
DEFAULT_BACKEND_REGISTRY = BackendRegistry(builtin_backend_detectors())

def identify_backend(dataframe: object,
                     *,
                     registry: BackendRegistry = DEFAULT_BACKEND_REGISTRY) -> BackendIdentity:
    """
    Identify a dataframe's native backend without evaluating its data

    - This function performs no row count, collection, conversion, schema access,
      query execution or copy
    - It only inspects Python types and package metadata
    """

    return registry.identify(dataframe)

def resolve_dataframe_input(dataframe: DataFrameType,
                            *,
                            registry: BackendRegistry = DEFAULT_BACKEND_REGISTRY) -> DataFrameInput[DataFrameType]:
    """
    Validate and wrap the exact dataframe object supplied by the caller
    """

    backend = identify_backend(
        dataframe, 
        registry = registry
    )

    return DataFrameInput(
        dataframe = dataframe, 
        backend = backend
    )