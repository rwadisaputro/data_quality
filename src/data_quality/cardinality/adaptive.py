"""Adaptive exact-to-HyperLogLog cardinality state and JSON result models."""

from __future__ import annotations

import json
import math
import operator
import sys
from dataclasses import dataclass, field
from enum import Enum
from importlib import import_module
from typing import Any, Callable

from data_quality.cardinality.diagnostics import CardinalityWarning
from data_quality.cardinality.hashing import (
    DEFAULT_HASH_CONFIGURATION,
    HashConfiguration,
)
from data_quality.cardinality.hll import DEFAULT_PRECISION, HyperLogLog
from data_quality.cardinality.sketch import HashedHyperLogLog

DEFAULT_EXACT_UNIQUE_THRESHOLD = 50_000
DEFAULT_EXACT_MEMORY_BUDGET_BYTES = 32 * 1024 * 1024
DEFAULT_MEMORY_SAFETY_FACTOR = 1.25
DEFAULT_THRESHOLD_CHECK_INTERVAL = 4_096
RESULT_SCHEMA_VERSION = "adaptive-cardinality-v5"
ALGORITHM_VERSION = "exact-to-classic-hll-v1"


class CardinalityMode(str, Enum):
    """Current retained-state representation."""

    EXACT = "exact"
    HLL = "hll"


class PromotionReason(str, Enum):
    """Reason exact state was replaced by HLL state."""

    UNIQUE_THRESHOLD = "exact_unique_threshold"
    MEMORY_THRESHOLD = "exact_memory_budget"
    BOTH = "exact_unique_threshold_and_memory_budget"


@dataclass(frozen=True, slots=True)
class AdaptiveCardinalityConfig:
    """Configuration controlling adaptive exact-to-HLL promotion."""

    exact_unique_threshold: int = DEFAULT_EXACT_UNIQUE_THRESHOLD
    exact_memory_budget_bytes: int = DEFAULT_EXACT_MEMORY_BUDGET_BYTES
    memory_safety_factor: float = DEFAULT_MEMORY_SAFETY_FACTOR
    hll_precision: int = DEFAULT_PRECISION
    threshold_check_interval: int = DEFAULT_THRESHOLD_CHECK_INTERVAL
    hash_configuration: HashConfiguration = field(default=DEFAULT_HASH_CONFIGURATION)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "exact_unique_threshold",
            _positive_integer("exact_unique_threshold", self.exact_unique_threshold),
        )
        object.__setattr__(
            self,
            "exact_memory_budget_bytes",
            _positive_integer("exact_memory_budget_bytes", self.exact_memory_budget_bytes),
        )
        object.__setattr__(
            self,
            "threshold_check_interval",
            _positive_integer("threshold_check_interval", self.threshold_check_interval),
        )
        if isinstance(self.memory_safety_factor, bool) or not isinstance(
            self.memory_safety_factor, (int, float)
        ):
            raise TypeError("memory_safety_factor must be numeric")
        if not math.isfinite(float(self.memory_safety_factor)):
            raise ValueError("memory_safety_factor must be finite")
        if float(self.memory_safety_factor) < 1.0:
            raise ValueError("memory_safety_factor must be at least 1.0")
        object.__setattr__(self, "memory_safety_factor", float(self.memory_safety_factor))

        object.__setattr__(self, "hll_precision", HyperLogLog(self.hll_precision).precision)
        if not isinstance(self.hash_configuration, HashConfiguration):
            raise TypeError("hash_configuration must be a HashConfiguration")

    def as_dict(self) -> dict[str, object]:
        """Return every adaptive parameter in JSON-safe form."""

        register_count = 1 << self.hll_precision
        return {
            "exact_unique_threshold": self.exact_unique_threshold,
            "exact_memory_budget_bytes": self.exact_memory_budget_bytes,
            "memory_safety_factor": self.memory_safety_factor,
            "threshold_check_interval": self.threshold_check_interval,
            "hll_precision": self.hll_precision,
            "hll_register_count": register_count,
            "hll_relative_standard_error": 1.04 / math.sqrt(register_count),
            "hash_configuration": self.hash_configuration.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class PromotionMetadata:
    """Telemetry describing the one-way exact-to-HLL transition."""

    occurred: bool
    reason: PromotionReason | None = None
    row_position: int | None = None
    non_null_position: int | None = None
    unique_count: int | None = None
    exact_state_bytes: int | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "occurred": self.occurred,
            "reason": self.reason.value if self.reason is not None else None,
            "row_position": self.row_position,
            "non_null_position": self.non_null_position,
            "unique_count": self.unique_count,
            "exact_state_bytes": self.exact_state_bytes,
        }


@dataclass(frozen=True, slots=True)
class AdaptiveCardinalityResult:
    """Complete adaptive-cardinality output plus reproducibility lineage."""

    distinct_count: int
    is_exact: bool
    final_mode: CardinalityMode
    total_row_count: int
    non_null_count: int
    null_count: int
    config: AdaptiveCardinalityConfig
    promotion: PromotionMetadata
    peak_exact_unique_count: int
    peak_exact_state_bytes: int
    retained_exact_unique_count: int
    retained_exact_state_bytes: int
    exact_state_representation: str
    hll_zero_register_count: int | None
    hll_register_storage_bytes: int | None
    raw_hll_estimate: float | None
    execution_parameters: dict[str, object]
    lineage: dict[str, object]
    warnings: tuple[CardinalityWarning, ...] = ()

    def as_dict(self) -> dict[str, object]:
        """Return the complete report as JSON-compatible primitives."""

        hll_register_count = 1 << self.config.hll_precision
        return {
            "schema_version": RESULT_SCHEMA_VERSION,
            "algorithm_version": ALGORITHM_VERSION,
            "output": {
                "distinct_count": self.distinct_count,
                "rounded_distinct_count": self.distinct_count,
                "is_exact": self.is_exact,
                "method": "exact" if self.is_exact else "hyperloglog",
                "final_mode": self.final_mode.value,
                "relative_standard_error": (
                    None if self.is_exact else 1.04 / math.sqrt(hll_register_count)
                ),
            },
            "row_metrics": {
                "total_row_count": self.total_row_count,
                "non_null_count": self.non_null_count,
                "null_count": self.null_count,
            },
            "parameters": {
                **self.config.as_dict(),
                "execution": self.execution_parameters,
            },
            "exact_state": {
                "representation": self.exact_state_representation,
                "retained_unique_count": self.retained_exact_unique_count,
                "retained_state_bytes": self.retained_exact_state_bytes,
                "peak_unique_count": self.peak_exact_unique_count,
                "peak_state_bytes": self.peak_exact_state_bytes,
            },
            "hll_state": {
                "algorithm": "classic_hyperloglog",
                "precision": self.config.hll_precision,
                "register_count": hll_register_count,
                "register_storage_bytes": self.hll_register_storage_bytes,
                "zero_register_count": self.hll_zero_register_count,
                "relative_standard_error": 1.04 / math.sqrt(hll_register_count),
                "raw_estimate": self.raw_hll_estimate,
            },
            "promotion": self.promotion.as_dict(),
            "warnings": [warning.as_dict() for warning in self.warnings],
            "lineage": self.lineage,
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize the full result and lineage as JSON."""

        return json.dumps(self.as_dict(), indent=indent, sort_keys=True)


class AdaptiveCardinalityHandler:
    """Retain exact backend-canonical tokens, then promote once to HLL."""

    __slots__ = (
        "_array_hasher",
        "_config",
        "_exact_values",
        "_hll",
        "_mode",
        "_non_null_count",
        "_peak_exact_state_bytes",
        "_peak_exact_unique_count",
        "_promotion",
    )

    def __init__(
        self,
        config: AdaptiveCardinalityConfig | None = None,
        *,
        array_hasher: Callable[[object], object] | None = None,
    ) -> None:
        self._config = config or AdaptiveCardinalityConfig()
        self._array_hasher = array_hasher
        self._load_numpy()
        self._exact_values: Any | None = None
        self._hll: HashedHyperLogLog | None = None
        self._mode = CardinalityMode.EXACT
        self._non_null_count = 0
        self._peak_exact_unique_count = 0
        self._peak_exact_state_bytes = 0
        self._promotion = PromotionMetadata(occurred=False)

    @property
    def mode(self) -> CardinalityMode:
        return self._mode

    @property
    def config(self) -> AdaptiveCardinalityConfig:
        return self._config

    @property
    def non_null_count(self) -> int:
        return self._non_null_count

    @property
    def promotion(self) -> PromotionMetadata:
        return self._promotion

    @property
    def peak_exact_unique_count(self) -> int:
        return self._peak_exact_unique_count

    @property
    def peak_exact_state_bytes(self) -> int:
        return self._peak_exact_state_bytes

    @property
    def retained_exact_unique_count(self) -> int:
        return 0 if self._exact_values is None else len(self._exact_values)

    @property
    def retained_exact_state_bytes(self) -> int:
        return self._measure_exact_state_bytes(self._exact_values)

    @property
    def hll(self) -> HashedHyperLogLog | None:
        return self._hll

    def add_canonical_array(
        self,
        canonical_values: object,
        *,
        row_positions: object | None = None,
    ) -> None:
        """Consume one-dimensional backend-canonical exact tokens in ndarray batches."""

        numpy_module = self._load_numpy()
        values = numpy_module.asarray(canonical_values)
        if values.ndim != 1:
            raise ValueError("canonical_values must be a one-dimensional ndarray")
        if values.size == 0:
            return

        if row_positions is None:
            positions = numpy_module.arange(
                self._non_null_count,
                self._non_null_count + len(values),
                dtype=numpy_module.int64,
            )
        else:
            positions = numpy_module.asarray(row_positions, dtype=numpy_module.int64)
            if positions.ndim != 1 or len(positions) != len(values):
                raise ValueError("row_positions must be one-dimensional and match values")

        non_null_before = self._non_null_count
        self._non_null_count += len(values)

        if self.mode is CardinalityMode.HLL:
            self._add_hll_values(values)
            return

        interval = self.config.threshold_check_interval
        for start in range(0, len(values), interval):
            end = min(start + interval, len(values))
            batch = values[start:end]
            base_exact_values = (
                self._empty_like(batch)
                if self._exact_values is None
                else self._exact_values
            )
            unique_batch = numpy_module.unique(batch)
            merged = numpy_module.union1d(base_exact_values, unique_batch)
            exact_state_bytes = self._measure_exact_state_bytes(merged)
            reason = self._promotion_reason(len(merged), exact_state_bytes)

            if reason is None:
                self._exact_values = merged
                self._update_peaks(len(merged), exact_state_bytes)
                continue

            prefix_length, crossing_state, crossing_bytes, crossing_reason = (
                self._find_first_promotion_prefix(base_exact_values, batch)
            )
            self._exact_values = crossing_state
            self._update_peaks(len(crossing_state), crossing_bytes)
            crossing_index = start + prefix_length
            row_position = int(positions[crossing_index - 1])
            non_null_position = non_null_before + crossing_index
            self._promote(
                reason=crossing_reason,
                row_position=row_position,
                non_null_position=non_null_position,
                exact_state_bytes=crossing_bytes,
            )
            if crossing_index < len(values):
                self._add_hll_values(values[crossing_index:])
            return

    def distinct_count(self) -> int:
        """Return exact cardinality or the rounded HLL estimate."""

        if self.mode is CardinalityMode.EXACT:
            return self.retained_exact_unique_count
        if self._hll is None:  # pragma: no cover - defensive invariant
            raise RuntimeError("HLL mode requires an allocated sketch")
        return round(self._hll.estimate())

    def build_result(
        self,
        *,
        total_row_count: int,
        null_count: int,
        lineage: dict[str, object],
        execution_parameters: dict[str, object] | None = None,
        exact_state_representation: str = "backend_canonical_ndarray",
        warnings: tuple[CardinalityWarning, ...] = (),
    ) -> AdaptiveCardinalityResult:
        """Build a complete JSON-ready result from the current state."""

        if total_row_count != self.non_null_count + null_count:
            raise ValueError("row metrics do not match the processed non-null count")

        hll = self._hll
        return AdaptiveCardinalityResult(
            distinct_count=self.distinct_count(),
            is_exact=self.mode is CardinalityMode.EXACT,
            final_mode=self.mode,
            total_row_count=total_row_count,
            non_null_count=self.non_null_count,
            null_count=null_count,
            config=self.config,
            promotion=self.promotion,
            peak_exact_unique_count=self.peak_exact_unique_count,
            peak_exact_state_bytes=self.peak_exact_state_bytes,
            retained_exact_unique_count=self.retained_exact_unique_count,
            retained_exact_state_bytes=self.retained_exact_state_bytes,
            exact_state_representation=exact_state_representation,
            hll_zero_register_count=(None if hll is None else hll.zero_register_count),
            hll_register_storage_bytes=(None if hll is None else len(hll.registers)),
            raw_hll_estimate=(None if hll is None else hll.estimate()),
            execution_parameters=dict(execution_parameters or {}),
            lineage=dict(lineage),
            warnings=tuple(warnings),
        )

    def _promote(
        self,
        *,
        reason: PromotionReason,
        row_position: int,
        non_null_position: int,
        exact_state_bytes: int,
    ) -> None:
        if self._exact_values is None:  # pragma: no cover - promotion invariant
            raise RuntimeError("Promotion requires retained exact state")
        hll = HashedHyperLogLog(
            precision=self.config.hll_precision,
            hash_configuration=self.config.hash_configuration,
        )
        exact_hashes = self._hash_canonical_array(self._exact_values)
        hll.add_hash_array(exact_hashes)
        unique_count = len(self._exact_values)

        self._exact_values = None
        self._hll = hll
        self._mode = CardinalityMode.HLL
        self._promotion = PromotionMetadata(
            occurred=True,
            reason=reason,
            row_position=row_position,
            non_null_position=non_null_position,
            unique_count=unique_count,
            exact_state_bytes=exact_state_bytes,
        )

    def _add_hll_values(self, canonical_values: object) -> None:
        if self._hll is None:  # pragma: no cover - defensive invariant
            raise RuntimeError("HLL state is not initialised")
        hashes = self._hash_canonical_array(canonical_values)
        self._hll.add_hash_array(hashes)

    def _hash_canonical_array(self, canonical_values: object) -> object:
        if self._array_hasher is None:
            raise RuntimeError(
                "Adaptive promotion requires a backend-provided ndarray hash function"
            )
        return self._array_hasher(canonical_values)

    def _find_first_promotion_prefix(
        self,
        base_exact_values: object,
        batch: object,
    ) -> tuple[int, Any, int, PromotionReason]:
        """Find the first row in a triggering batch that actually crosses a limit."""

        numpy_module = self._load_numpy()
        base = numpy_module.asarray(base_exact_values)
        values = numpy_module.asarray(batch)
        low = 1
        high = len(values)

        while low < high:
            middle = (low + high) // 2
            candidate = numpy_module.union1d(base, numpy_module.unique(values[:middle]))
            candidate_bytes = self._measure_exact_state_bytes(candidate)
            reason = self._promotion_reason(len(candidate), candidate_bytes)
            if reason is None:
                low = middle + 1
            else:
                high = middle

        crossing_state = numpy_module.union1d(base, numpy_module.unique(values[:low]))
        crossing_bytes = self._measure_exact_state_bytes(crossing_state)
        crossing_reason = self._promotion_reason(len(crossing_state), crossing_bytes)
        if crossing_reason is None:  # pragma: no cover - triggering batch invariant
            raise RuntimeError("promotion refinement did not find a crossing prefix")
        return low, crossing_state, crossing_bytes, crossing_reason

    def _promotion_reason(
        self,
        unique_count: int,
        exact_state_bytes: int,
    ) -> PromotionReason | None:
        unique_exceeded = unique_count > self.config.exact_unique_threshold
        memory_exceeded = exact_state_bytes > self.config.exact_memory_budget_bytes
        if unique_exceeded and memory_exceeded:
            return PromotionReason.BOTH
        if unique_exceeded:
            return PromotionReason.UNIQUE_THRESHOLD
        if memory_exceeded:
            return PromotionReason.MEMORY_THRESHOLD
        return None

    def _measure_exact_state_bytes(self, values: object | None) -> int:
        if values is None:
            return 0
        numpy_module = self._load_numpy()
        array = numpy_module.asarray(values)
        if array.size == 0:
            return 0
        raw_bytes = int(array.nbytes)
        if array.dtype.hasobject:
            object_sizes = numpy_module.frompyfunc(sys.getsizeof, 1, 1)(array)
            raw_bytes += int(object_sizes.astype(numpy_module.int64).sum())
        return math.ceil(raw_bytes * self.config.memory_safety_factor)

    def _update_peaks(self, unique_count: int, exact_state_bytes: int) -> None:
        self._peak_exact_unique_count = max(self._peak_exact_unique_count, unique_count)
        self._peak_exact_state_bytes = max(self._peak_exact_state_bytes, exact_state_bytes)

    @staticmethod
    def _empty_like(values: object) -> Any:
        numpy_module = AdaptiveCardinalityHandler._load_numpy()
        array = numpy_module.asarray(values)
        return numpy_module.empty(0, dtype=array.dtype)

    @staticmethod
    def _load_numpy() -> Any:
        try:
            return import_module("numpy")
        except ImportError as exc:
            raise ImportError("Adaptive cardinality requires NumPy") from exc


def _positive_integer(name: str, value: int) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not a boolean")
    try:
        coerced = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be integer-like") from exc
    if coerced <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return coerced
