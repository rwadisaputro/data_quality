"""Immutable registry for native physical-schema discoverers"""

from __future__ import annotations

from collections.abc import Iterable

from data_quality.exceptions import (
    DuplicateSchemaDiscovererError,
    PhysicalSchemaDiscoveryError,
    UnsupportedSchemaDiscoveryBackendError,
)
from data_quality.models import BackendName, DataFrameInput
from data_quality.schema_discovery.base import PhysicalSchemaDiscoverer
from data_quality.schema_models import DataFramePhysicalSchema, RowCountMode



class PhysicalSchemaRegistry:
    """
    Dispatch schema discovery using an immutable adapter snapshot
    """

    def __init__(self, 
                 discoverers: Iterable[PhysicalSchemaDiscoverer]) -> None:
        discoverer_snapshot = tuple(discoverers)
        backend_names: set[str] = set()

        for discoverer in discoverer_snapshot:
            if not isinstance(discoverer.backend_name, BackendName):
                raise TypeError(
                    "Physical-schema discoverer backend_name must be a `BackendName` value"
                )
            backend_name = discoverer.backend_name.value
            if backend_name in backend_names:
                raise DuplicateSchemaDiscovererError(backend_name)
            backend_names.add(backend_name)

        self._discoverers = discoverer_snapshot

    @property
    def discoverers(self) -> tuple[PhysicalSchemaDiscoverer, ...]:
        """
        Return the ordered immutable discoverer snapshot
        """

        return self._discoverers

    def with_discoverer(self,
                        discoverer: PhysicalSchemaDiscoverer) -> PhysicalSchemaRegistry:
        """
        Return a new registry containing `discoverer` at the end
        """

        return PhysicalSchemaRegistry((*self._discoverers, discoverer))

    def discover(self,
                 dataframe_input: DataFrameInput[object],
                 *,
                 row_count_mode: RowCountMode) -> DataFramePhysicalSchema:
        """
        Dispatch to the discoverer registered for the detected backend
        """

        if not isinstance(dataframe_input, DataFrameInput):
            raise TypeError("`dataframe_input` must be a `DataFrameInput` value")
        if not isinstance(row_count_mode, RowCountMode):
            raise TypeError("`row_count_mode` must be a `RowCountMode` value")

        matching_discoverer = next(
            (
                discoverer
                for discoverer in self._discoverers
                if discoverer.backend_name is dataframe_input.backend.name
            ),
            None,
        )
        if matching_discoverer is None:
            raise UnsupportedSchemaDiscoveryBackendError(
                dataframe_input.backend.name.value
            )

        try:
            return matching_discoverer.discover(
                dataframe_input.dataframe,
                dataframe_input.backend,
                row_count_mode,
            )
        except Exception as error:
            raise PhysicalSchemaDiscoveryError(
                backend_name = dataframe_input.backend.name.value,
                dataframe_type = dataframe_input.backend.dataframe_type,
            ) from error