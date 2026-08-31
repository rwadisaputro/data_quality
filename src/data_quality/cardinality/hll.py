"""Dependency-free classic HyperLogLog sketch for approximate distinct counting.

This module deliberately accepts pre-hashed 64-bit integers rather than raw
Python values. Canonicalisation and backend-specific hashing belong outside the
sketch so every dataframe backend can share the same cardinality mechanics.
"""

from __future__ import annotations

import math
import operator
from collections.abc import Iterable
from importlib import import_module
from typing import Any


DEFAULT_PRECISION = 14
MIN_PRECISION = 4
# This is a library memory-policy ceiling, not a mathematical HLL limit. With
# one byte per register, p=20 caps the dense register array at 1 MiB.
MAX_PRECISION = 20
HASH_WIDTH_BITS = 64
_HASH_MASK = (1 << HASH_WIDTH_BITS) - 1


class HyperLogLog:
    """Classic HyperLogLog sketch over uniform unsigned 64-bit hashes.

    Parameters
    ----------
    precision:
        Number of high-order hash bits used to select a register. The sketch
        contains ``2**precision`` one-byte registers. Supported values are
        4 through 20, inclusive.

    Notes
    -----
    This implementation uses the original HyperLogLog harmonic-mean estimator
    with Linear Counting in the small-cardinality range. It does not implement
    the empirical bias tables from HyperLogLog++.

    The sketch does not hash raw values itself. Callers must provide stable,
    uniformly distributed unsigned 64-bit hashes so equality/canonicalisation
    policy remains independent of this core data structure.
    """

    __slots__ = ("_precision", "_registers", "_suffix_bits", "_suffix_mask")

    def __init__(self, precision: int = DEFAULT_PRECISION) -> None:
        precision_value = self._coerce_precision(precision)
        if not MIN_PRECISION <= precision_value <= MAX_PRECISION:
            raise ValueError(
                f"HyperLogLog precision must be between {MIN_PRECISION} and "
                f"{MAX_PRECISION}, received {precision_value}"
            )

        self._precision = precision_value
        self._registers = bytearray(1 << precision_value)
        self._suffix_bits = HASH_WIDTH_BITS - precision_value
        self._suffix_mask = (1 << self._suffix_bits) - 1

    @property
    def precision(self) -> int:
        """Return the register-index precision ``p``."""

        return self._precision

    @property
    def register_count(self) -> int:
        """Return the number of registers ``m = 2**p``."""

        return len(self._registers)

    @property
    def hash_width_bits(self) -> int:
        """Return the required input hash width."""

        return HASH_WIDTH_BITS

    @property
    def relative_standard_error(self) -> float:
        """Return the conventional classic-HLL relative standard error."""

        return 1.04 / math.sqrt(self.register_count)

    @property
    def zero_register_count(self) -> int:
        """Return the number of registers that have never been updated."""

        return self._registers.count(0)

    @property
    def registers(self) -> bytes:
        """Return an immutable snapshot of the register array."""

        return bytes(self._registers)

    @property
    def is_empty(self) -> bool:
        """Return whether no register has been updated."""

        return self.zero_register_count == self.register_count

    def add_hash(self, hash_value: int) -> None:
        """Update the sketch with one unsigned 64-bit hash value."""

        value = self._coerce_hash(hash_value)
        register_index = value >> self._suffix_bits
        suffix = value & self._suffix_mask
        rank = self._rank(suffix)

        if rank > self._registers[register_index]:
            self._registers[register_index] = rank

    def add_hashes(self, hash_values: Iterable[int]) -> None:
        """Update the sketch with an iterable of unsigned 64-bit hash values."""

        for hash_value in hash_values:
            self.add_hash(hash_value)

    def add_hash_array(self, hash_values: object) -> None:
        """Vectorise register updates from a one-dimensional ``np.uint64`` array.

        NumPy remains optional for the dependency-free HLL core. The dependency is
        loaded only when this bulk method is used by an array-aware backend adapter.
        """

        numpy_module = self._load_numpy()
        values = numpy_module.asarray(hash_values)
        if values.ndim != 1:
            raise ValueError("hash_values must be a one-dimensional ndarray")
        if values.size == 0:
            return

        if not numpy_module.issubdtype(values.dtype, numpy_module.integer):
            raise TypeError("hash_values ndarray must contain integers")
        if numpy_module.issubdtype(values.dtype, numpy_module.signedinteger):
            if bool((values < 0).any()):
                raise ValueError("hash_values ndarray cannot contain negative integers")

        values = values.astype(numpy_module.uint64, copy=False)
        register_indexes = values >> numpy_module.uint64(self._suffix_bits)
        suffixes = values & numpy_module.uint64(self._suffix_mask)

        powers = numpy_module.left_shift(
            numpy_module.uint64(1),
            numpy_module.arange(self._suffix_bits, dtype=numpy_module.uint64),
        )
        bit_lengths = numpy_module.searchsorted(powers, suffixes, side="right")
        ranks = self._suffix_bits - bit_lengths + 1

        registers = numpy_module.frombuffer(self._registers, dtype=numpy_module.uint8)
        numpy_module.maximum.at(
            registers,
            register_indexes.astype(numpy_module.intp, copy=False),
            ranks.astype(numpy_module.uint8, copy=False),
        )

    def estimate(self) -> float:
        """Return the approximate number of distinct input hashes.

        The estimator is classic 64-bit HyperLogLog: use the bias-corrected
        harmonic mean, then Linear Counting while the raw estimate is at most
        ``2.5 * m`` and zero registers remain. No legacy 32-bit large-range
        correction is applied.
        """

        register_count = self.register_count
        harmonic_sum = math.fsum(
            math.ldexp(1.0, -register_value)
            for register_value in self._registers
        )
        raw_estimate = (
            self._alpha(register_count)
            * register_count
            * register_count
            / harmonic_sum
        )

        zero_registers = self.zero_register_count
        if raw_estimate <= 2.5 * register_count and zero_registers > 0:
            return register_count * math.log(register_count / zero_registers)

        return raw_estimate

    def merge(self, other: HyperLogLog) -> None:
        """Merge ``other`` into this sketch using register-wise maxima.

        Both sketches must use the same precision. Hash-algorithm compatibility
        is intentionally handled by the higher-level hashing/configuration layer
        because this core receives hashes rather than raw values.
        """

        if not isinstance(other, HyperLogLog):
            raise TypeError("Can only merge another HyperLogLog sketch")
        if self.precision != other.precision:
            raise ValueError(
                "Cannot merge HyperLogLog sketches with different precisions: "
                f"{self.precision} != {other.precision}"
            )

        for index, other_value in enumerate(other._registers):
            if other_value > self._registers[index]:
                self._registers[index] = other_value

    def copy(self) -> HyperLogLog:
        """Return an independent copy of this sketch."""

        copied = HyperLogLog(self.precision)
        copied._registers[:] = self._registers
        return copied

    def _rank(self, suffix: int) -> int:
        """Return one plus the leading-zero count within the hash suffix."""

        return self._suffix_bits - suffix.bit_length() + 1

    @staticmethod
    def _alpha(register_count: int) -> float:
        """Return the standard HyperLogLog bias-correction constant."""

        if register_count == 16:
            return 0.673
        if register_count == 32:
            return 0.697
        if register_count == 64:
            return 0.709
        return 0.7213 / (1.0 + 1.079 / register_count)

    @staticmethod
    def _load_numpy() -> Any:
        """Load NumPy only for the optional bulk-update path."""

        try:
            return import_module("numpy")
        except ImportError as exc:
            raise ImportError("Bulk HyperLogLog updates require NumPy") from exc

    @staticmethod
    def _coerce_precision(precision: int) -> int:
        """Validate and normalise an integer-like precision value."""

        if isinstance(precision, bool):
            raise TypeError("HyperLogLog precision must be an integer, not a boolean")

        try:
            return operator.index(precision)
        except TypeError as exc:
            raise TypeError("HyperLogLog precision must be integer-like") from exc

    @staticmethod
    def _coerce_hash(hash_value: int) -> int:
        """Validate and normalise an integer-like unsigned 64-bit hash."""

        if isinstance(hash_value, bool):
            raise TypeError("HyperLogLog hash values must be integers, not booleans")

        try:
            value = operator.index(hash_value)
        except TypeError as exc:
            raise TypeError("HyperLogLog hash values must be integer-like") from exc

        if not 0 <= value <= _HASH_MASK:
            raise ValueError(
                "HyperLogLog hash values must be unsigned 64-bit integers "
                f"between 0 and {_HASH_MASK}"
            )
        return value
