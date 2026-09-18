"""Structured diagnostics emitted by cardinality adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class CardinalityWarningCode(str, Enum):
    """Stable machine-readable cardinality diagnostic codes."""

    SCALAR_FALLBACK = "scalar_canonicalization_fallback"
    UNSUPPORTED_SCALAR = "unsupported_cardinality_scalar"
    UNSUPPORTED_DTYPE = "unsupported_cardinality_dtype"
    DRIVER_STREAM_FALLBACK = "distributed_driver_stream_fallback"


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
                f"Unsupported cardinality scalar type {scalar_type} in column "
                f"{_display_metadata_value(column)}"
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



class UnsupportedCardinalityDtypeError(TypeError):
    """Raised when a backend dtype has no defined cardinality semantics."""

    def __init__(
        self,
        *,
        backend: str,
        column: object,
        native_dtype: str,
        dtype_family: str,
    ) -> None:
        warning = CardinalityWarning(
            code=CardinalityWarningCode.UNSUPPORTED_DTYPE,
            severity=CardinalityWarningSeverity.ERROR,
            message=(
                f"Unsupported cardinality dtype {native_dtype} in column "
                f"{_display_metadata_value(column)}"
            ),
            context={
                "backend": backend,
                "column": _json_safe_value(column),
                "native_dtype": native_dtype,
                "dtype_family": dtype_family,
                "action": "cardinality_not_computed",
            },
        )
        super().__init__(warning.message)
        self.warning = warning

    def as_dict(self) -> dict[str, object]:
        """Return the structured failure payload."""

        return self.warning.as_dict()


def driver_stream_fallback_warning(
    *,
    backend: str,
    connection_mode: str,
) -> CardinalityWarning:
    """Describe a distributed backend path that streams rows to the driver."""

    return CardinalityWarning(
        code=CardinalityWarningCode.DRIVER_STREAM_FALLBACK,
        severity=CardinalityWarningSeverity.PERFORMANCE,
        message=(
            f"{backend} cardinality is streaming rows to the driver for "
            f"connection mode {connection_mode!r}"
        ),
        context={
            "backend": backend,
            "connection_mode": connection_mode,
            "action": "driver_stream_fallback",
        },
    )

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
            f"Column {_display_metadata_value(column)} uses scalar canonicalization because "
            f"its {backend} dtype family is {dtype_family}"
        ),
        context={
            "backend": backend,
            "column": _json_safe_value(column),
            "native_dtype": native_dtype,
            "dtype_family": dtype_family,
            "action": "scalar_canonicalization_fallback",
        },
    )


def json_safe_metadata_value(value: object) -> object:
    """Convert metadata labels to JSON-safe data without calling arbitrary ``repr``."""

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, tuple):
        return [json_safe_metadata_value(item) for item in value]
    if isinstance(value, list):
        return [json_safe_metadata_value(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): json_safe_metadata_value(item)
            for key, item in value.items()
            if isinstance(key, (str, int, float, bool)) or key is None
        }
    if isinstance(value, (bytes, bytearray, memoryview)):
        raw = bytes(value)
        return {
            "type": "bytes",
            "length": len(raw),
        }
    return {
        "type": f"{type(value).__module__}.{type(value).__qualname__}",
        "serialization": "omitted",
    }


def _json_safe_value(value: object) -> object:
    return json_safe_metadata_value(value)


def _display_metadata_value(value: object) -> str:
    if value is None or isinstance(value, (bool, int, float, str)):
        return str(value)
    return f"<{type(value).__module__}.{type(value).__qualname__}>"
