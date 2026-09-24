"""Backend-neutral dataframe-level cardinality report models."""

from __future__ import annotations

import json
from dataclasses import dataclass

from data_quality.cardinality.adaptive import AdaptiveCardinalityResult
from data_quality.cardinality.diagnostics import (
    CardinalityWarning,
    json_safe_metadata_value,
)

DATAFRAME_RESULT_SCHEMA_VERSION = "dataframe-cardinality-v1"


@dataclass(frozen=True, slots=True)
class ColumnCardinalityOutcome:
    """Result or structured failure for one dataframe column."""

    column: object
    position: int
    result: AdaptiveCardinalityResult | None
    error: CardinalityWarning | None = None

    @property
    def status(self) -> str:
        return "ok" if self.result is not None else "unsupported"

    @property
    def warnings(self) -> tuple[CardinalityWarning, ...]:
        if self.result is not None:
            return self.result.warnings
        if self.error is not None:
            return (self.error,)
        return ()

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-safe per-column payload."""

        return {
            "column": json_safe_metadata_value(self.column),
            "position": self.position,
            "status": self.status,
            "result": None if self.result is None else self.result.as_dict(),
            "error": None if self.error is None else self.error.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class DataFrameCardinalityResult:
    """Cardinality report for every column in one detected dataframe."""

    backend: dict[str, object]
    total_row_count: int
    column_count: int
    columns: tuple[ColumnCardinalityOutcome, ...]

    @property
    def successful_column_count(self) -> int:
        return sum(outcome.result is not None for outcome in self.columns)

    @property
    def unsupported_column_count(self) -> int:
        return self.column_count - self.successful_column_count

    @property
    def warnings(self) -> tuple[CardinalityWarning, ...]:
        return tuple(warning for outcome in self.columns for warning in outcome.warnings)

    def as_dict(self) -> dict[str, object]:
        """Return complete dataframe-level cardinality output and lineage."""

        return {
            "schema_version": DATAFRAME_RESULT_SCHEMA_VERSION,
            "backend": dict(self.backend),
            "row_count": self.total_row_count,
            "column_count": self.column_count,
            "successful_column_count": self.successful_column_count,
            "unsupported_column_count": self.unsupported_column_count,
            "columns": [outcome.as_dict() for outcome in self.columns],
            "warnings": [warning.as_dict() for warning in self.warnings],
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize the complete dataframe-level report as JSON."""

        return json.dumps(self.as_dict(), indent=indent, sort_keys=True)

