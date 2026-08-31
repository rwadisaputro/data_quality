"""Structured diagnostics emitted by cardinality adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class CardinalityWarningCode(str, Enum):
    """Stable machine-readable cardinality diagnostic codes."""

    SCALAR_FALLBACK = "scalar_canonicalization_fallback"
    UNSUPPORTED_SCALAR = "unsupported_cardinality_scalar"


class CardinalityWarningSeverity(str, Enum):
    """Severity of a structured cardinality diagnostic."""

    PERFORMANCE = "performance"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class CardinalityWarning:
    """JSON-safe warning attached to cardinality lineage or failures."""

    code: CardinalityWarningCode
    message: str
    severity: CardinalityWarningSeverity
    context: dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        """Return the warning as JSON-safe primitives."""

        return {
            "code": self.code.value,
            "message": self.message,
            "severity": self.severity.value,
            "context": dict(self.context),
        }


class UnsupportedCardinalityValueError(TypeError):
    """Raised when a scalar fallback encounters an unsupported logical value."""

    def __init__(
        self,
        *,
        backend: str,
        column: object,
        native_dtype: str,
        row_position: int | None,
        value: object,
    ) -> None:
        scalar_type = f"{type(value).__module__}.{type(value).__qualname__}"
        warning = CardinalityWarning(
            code=CardinalityWarningCode.UNSUPPORTED_SCALAR,
            severity=CardinalityWarningSeverity.ERROR,
            message=(
                f"Unsupported cardinality scalar type {scalar_type} in column {column!r}"
            ),
            context={
                "backend": backend,
                "column": _json_safe_value(column),
                "native_dtype": native_dtype,
                "row_position": row_position,
                "scalar_type": scalar_type,
                "action": "cardinality_not_computed",
            },
        )
        super().__init__(warning.message)
        self.warning = warning

    def as_dict(self) -> dict[str, object]:
        """Return the structured failure payload."""

        return self.warning.as_dict()


def scalar_fallback_warning(
    *,
    backend: str,
    column: object,
    native_dtype: str,
    dtype_family: str,
) -> CardinalityWarning:
    """Build the performance warning for the mixed/object scalar fallback path."""

    return CardinalityWarning(
        code=CardinalityWarningCode.SCALAR_FALLBACK,
        severity=CardinalityWarningSeverity.PERFORMANCE,
        message=(
            f"Column {column!r} uses scalar canonicalization because its pandas dtype "
            f"family is {dtype_family!r}"
        ),
        context={
            "backend": backend,
            "column": _json_safe_value(column),
            "native_dtype": native_dtype,
            "dtype_family": dtype_family,
            "action": "scalar_canonicalization_fallback",
        },
    )


def _json_safe_value(value: object) -> object:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return repr(value)
