"""Backend-neutral public entry points for dataframe cardinality profiling."""

from __future__ import annotations

from typing import TypeVar, overload

from data_quality.backends.registry import BackendRegistry
from data_quality.cardinality.adaptive import (
    AdaptiveCardinalityConfig,
    AdaptiveCardinalityResult,
)
from data_quality.cardinality.builtin import builtin_cardinality_adapters
from data_quality.cardinality.registry import CardinalityRegistry
from data_quality.cardinality.report import DataFrameCardinalityResult
from data_quality.intake import DEFAULT_BACKEND_REGISTRY, identify_backend
from data_quality.models import DataFrameInput

DataFrameType = TypeVar("DataFrameType")
DEFAULT_CARDINALITY_REGISTRY = CardinalityRegistry(builtin_cardinality_adapters())
_ALL_COLUMNS = object()


@overload
def profile_cardinality_from_input(
    dataframe_input: DataFrameInput[object],
    *,
    config: AdaptiveCardinalityConfig | None = None,
    batch_size: int | None = None,
    registry: CardinalityRegistry = DEFAULT_CARDINALITY_REGISTRY,
) -> DataFrameCardinalityResult: ...


@overload
def profile_cardinality_from_input(
    dataframe_input: DataFrameInput[object],
    column: object,
    *,
    config: AdaptiveCardinalityConfig | None = None,
    batch_size: int | None = None,
    registry: CardinalityRegistry = DEFAULT_CARDINALITY_REGISTRY,
) -> AdaptiveCardinalityResult: ...


def profile_cardinality_from_input(
    dataframe_input: DataFrameInput[object],
    column: object = _ALL_COLUMNS,
    *,
    config: AdaptiveCardinalityConfig | None = None,
    batch_size: int | None = None,
    registry: CardinalityRegistry = DEFAULT_CARDINALITY_REGISTRY,
) -> AdaptiveCardinalityResult | DataFrameCardinalityResult:
    """Profile one or all columns from an already-resolved dataframe input."""

    if not isinstance(dataframe_input, DataFrameInput):
        raise TypeError(
            "Expected `DataFrameInput`, call `identify_backend()`/`resolve_dataframe_input()` "
            "first or pass the native dataframe to `profile_cardinality()`"
        )

    if column is _ALL_COLUMNS:
        return registry.profile_all(
            dataframe_input,
            config=config,
            batch_size=batch_size,
        )
    return registry.profile(
        dataframe_input,
        column,
        config=config,
        batch_size=batch_size,
    )


@overload
def profile_cardinality(
    dataframe: DataFrameType,
    *,
    config: AdaptiveCardinalityConfig | None = None,
    batch_size: int | None = None,
    backend_registry: BackendRegistry = DEFAULT_BACKEND_REGISTRY,
    cardinality_registry: CardinalityRegistry = DEFAULT_CARDINALITY_REGISTRY,
) -> DataFrameCardinalityResult: ...


@overload
def profile_cardinality(
    dataframe: DataFrameType,
    column: object,
    *,
    config: AdaptiveCardinalityConfig | None = None,
    batch_size: int | None = None,
    backend_registry: BackendRegistry = DEFAULT_BACKEND_REGISTRY,
    cardinality_registry: CardinalityRegistry = DEFAULT_CARDINALITY_REGISTRY,
) -> AdaptiveCardinalityResult: ...


def profile_cardinality(
    dataframe: DataFrameType,
    column: object = _ALL_COLUMNS,
    *,
    config: AdaptiveCardinalityConfig | None = None,
    batch_size: int | None = None,
    backend_registry: BackendRegistry = DEFAULT_BACKEND_REGISTRY,
    cardinality_registry: CardinalityRegistry = DEFAULT_CARDINALITY_REGISTRY,
) -> AdaptiveCardinalityResult | DataFrameCardinalityResult:
    """Detect the native backend, then profile one column or the entire dataframe.

    Omitting ``column`` is the backend-neutral dataframe-first path. Backend detection
    occurs once at the boundary; the registry then delegates dtype-specific execution to
    the native adapter while the exact/HLL state machine remains backend-neutral.
    """

    backend = identify_backend(dataframe, registry=backend_registry)
    dataframe_input: DataFrameInput[object] = DataFrameInput(
        dataframe=dataframe,
        backend=backend,
    )
    if column is _ALL_COLUMNS:
        return profile_cardinality_from_input(
            dataframe_input,
            config=config,
            batch_size=batch_size,
            registry=cardinality_registry,
        )
    return profile_cardinality_from_input(
        dataframe_input,
        column,
        config=config,
        batch_size=batch_size,
        registry=cardinality_registry,
    )


def profile_cardinality_json(
    dataframe: DataFrameType,
    column: object = _ALL_COLUMNS,
    *,
    config: AdaptiveCardinalityConfig | None = None,
    batch_size: int | None = None,
    indent: int | None = 2,
    backend_registry: BackendRegistry = DEFAULT_BACKEND_REGISTRY,
    cardinality_registry: CardinalityRegistry = DEFAULT_CARDINALITY_REGISTRY,
) -> str:
    """Return one-column or full-dataframe cardinality output and lineage as JSON."""

    if column is _ALL_COLUMNS:
        result = profile_cardinality(
            dataframe,
            config=config,
            batch_size=batch_size,
            backend_registry=backend_registry,
            cardinality_registry=cardinality_registry,
        )
    else:
        result = profile_cardinality(
            dataframe,
            column,
            config=config,
            batch_size=batch_size,
            backend_registry=backend_registry,
            cardinality_registry=cardinality_registry,
        )
    return result.to_json(indent=indent)
