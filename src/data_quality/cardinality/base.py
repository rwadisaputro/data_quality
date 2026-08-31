"""Backend-neutral contracts for dataframe cardinality adapters."""

from __future__ import annotations

from typing import Protocol

from data_quality.cardinality.adaptive import (
    AdaptiveCardinalityConfig,
    AdaptiveCardinalityResult,
)
from data_quality.cardinality.report import DataFrameCardinalityResult
from data_quality.models import BackendIdentity, BackendName


class CardinalityBackendAdapter(Protocol):
    """Structural interface implemented by backend cardinality adapters."""

    backend_name: BackendName

    def profile(
        self,
        dataframe: object,
        backend: BackendIdentity,
        column: object,
        *,
        config: AdaptiveCardinalityConfig | None,
        batch_size: int | None,
    ) -> AdaptiveCardinalityResult:
        """Profile one native dataframe column using backend-specific execution."""

    def profile_all(
        self,
        dataframe: object,
        backend: BackendIdentity,
        *,
        config: AdaptiveCardinalityConfig | None,
        batch_size: int | None,
    ) -> DataFrameCardinalityResult:
        """Profile every native dataframe column using backend-specific execution."""
