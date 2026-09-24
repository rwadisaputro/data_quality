"""Immutable backend registry for cardinality profiling."""

from __future__ import annotations

from collections.abc import Iterable

from data_quality.cardinality.adaptive import (
    AdaptiveCardinalityConfig,
    AdaptiveCardinalityResult,
)
from data_quality.cardinality.base import CardinalityBackendAdapter
from data_quality.cardinality.report import DataFrameCardinalityResult
from data_quality.exceptions import (
    CardinalityProfilingError,
    DuplicateCardinalityAdapterError,
    UnsupportedCardinalityBackendError,
)
from data_quality.models import BackendName, DataFrameInput


class CardinalityRegistry:
    """Dispatch cardinality profiling using an immutable adapter snapshot."""

    def __init__(self, adapters: Iterable[CardinalityBackendAdapter]) -> None:
        adapter_snapshot = tuple(adapters)
        backend_names: set[str] = set()

        for adapter in adapter_snapshot:
            if not isinstance(adapter.backend_name, BackendName):
                raise TypeError("Cardinality adapter backend_name must be a `BackendName` value")
            backend_name = adapter.backend_name.value
            if backend_name in backend_names:
                raise DuplicateCardinalityAdapterError(backend_name)
            backend_names.add(backend_name)

        self._adapters = adapter_snapshot

    @property
    def adapters(self) -> tuple[CardinalityBackendAdapter, ...]:
        """Return the ordered immutable adapter snapshot."""

        return self._adapters

    def with_adapter(self, adapter: CardinalityBackendAdapter) -> CardinalityRegistry:
        """Return a new registry containing ``adapter`` at the end."""

        return CardinalityRegistry((*self._adapters, adapter))

    def profile(
        self,
        dataframe_input: DataFrameInput[object],
        column: object,
        *,
        config: AdaptiveCardinalityConfig | None,
        batch_size: int | None,
    ) -> AdaptiveCardinalityResult:
        """Dispatch one-column profiling to the detected backend adapter."""

        adapter = self._matching_adapter(dataframe_input)
        try:
            return adapter.profile(
                dataframe_input.dataframe,
                dataframe_input.backend,
                column,
                config=config,
                batch_size=batch_size,
            )
        except (KeyError, TypeError, ValueError):
            raise
        except Exception as error:
            raise self._profiling_error(dataframe_input) from error

    def profile_all(
        self,
        dataframe_input: DataFrameInput[object],
        *,
        config: AdaptiveCardinalityConfig | None,
        batch_size: int | None,
    ) -> DataFrameCardinalityResult:
        """Dispatch all-column profiling to the detected backend adapter."""

        adapter = self._matching_adapter(dataframe_input)
        try:
            return adapter.profile_all(
                dataframe_input.dataframe,
                dataframe_input.backend,
                config=config,
                batch_size=batch_size,
            )
        except (KeyError, TypeError, ValueError):
            raise
        except Exception as error:
            raise self._profiling_error(dataframe_input) from error

    def _matching_adapter(
        self,
        dataframe_input: DataFrameInput[object],
    ) -> CardinalityBackendAdapter:
        if not isinstance(dataframe_input, DataFrameInput):
            raise TypeError("`dataframe_input` must be a `DataFrameInput` value")

        matching_adapter = next(
            (
                adapter
                for adapter in self._adapters
                if adapter.backend_name is dataframe_input.backend.name
            ),
            None,
        )
        if matching_adapter is None:
            raise UnsupportedCardinalityBackendError(dataframe_input.backend.name.value)
        return matching_adapter

    @staticmethod
    def _profiling_error(dataframe_input: DataFrameInput[object]) -> CardinalityProfilingError:
        return CardinalityProfilingError(
            backend_name=dataframe_input.backend.name.value,
            dataframe_type=dataframe_input.backend.dataframe_type,
        )
