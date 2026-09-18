"""Configured HyperLogLog wrapper carrying hash compatibility metadata."""

from __future__ import annotations

from collections.abc import Iterable

from data_quality.cardinality.hashing import (
    DEFAULT_HASH_CONFIGURATION,
    HashConfiguration,
    hash_canonical_bytes,
)
from data_quality.cardinality.hll import DEFAULT_PRECISION, HyperLogLog


class HashedHyperLogLog:
    """HyperLogLog sketch bound to one stable hashing configuration.

    The raw :class:`HyperLogLog` core intentionally knows nothing about hashing.
    This wrapper keeps the metadata required to reject unsafe merges between
    sketches produced with different hash seeds or canonicalisation contracts.
    """

    __slots__ = ("_hash_configuration", "_hll")

    def __init__(
        self,
        precision: int = DEFAULT_PRECISION,
        *,
        hash_configuration: HashConfiguration = DEFAULT_HASH_CONFIGURATION,
    ) -> None:
        if not isinstance(hash_configuration, HashConfiguration):
            raise TypeError("hash_configuration must be a HashConfiguration")

        self._hll = HyperLogLog(precision=precision)
        self._hash_configuration = hash_configuration

    @property
    def precision(self) -> int:
        """Return the HLL register-index precision."""

        return self._hll.precision

    @property
    def register_count(self) -> int:
        """Return the number of HLL registers."""

        return self._hll.register_count

    @property
    def registers(self) -> bytes:
        """Return an immutable register snapshot."""

        return self._hll.registers

    @property
    def relative_standard_error(self) -> float:
        """Return the conventional classic-HLL relative standard error."""

        return self._hll.relative_standard_error

    @property
    def hash_configuration(self) -> HashConfiguration:
        """Return the immutable hashing/canonicalisation compatibility metadata."""

        return self._hash_configuration

    @property
    def is_empty(self) -> bool:
        """Return whether the underlying sketch has never been updated."""

        return self._hll.is_empty

    @property
    def zero_register_count(self) -> int:
        """Return the number of HLL registers that have not been updated."""

        return self._hll.zero_register_count

    def add_hash(self, hash_value: int) -> None:
        """Update with a hash already produced by this sketch's configuration."""

        self._hll.add_hash(hash_value)

    def add_hashes(self, hash_values: Iterable[int]) -> None:
        """Update with hashes already produced by this sketch's configuration."""

        self._hll.add_hashes(hash_values)

    def add_hash_array(self, hash_values: object) -> None:
        """Update from a one-dimensional ndarray of configured 64-bit hashes."""

        self._hll.add_hash_array(hash_values)

    def add_register_array(
        self,
        register_indexes: object,
        ranks: object,
    ) -> None:
        """Merge backend-preaggregated register maxima into the sketch."""

        self._hll.add_register_array(register_indexes, ranks)

    def add_canonical_bytes(self, canonical_value: bytes) -> None:
        """Hash one canonical value with this configuration and update the sketch."""

        self.add_hash(
            hash_canonical_bytes(
                canonical_value,
                configuration=self.hash_configuration,
            )
        )

    def estimate(self) -> float:
        """Return the approximate distinct cardinality."""

        return self._hll.estimate()

    def merge(self, other: HashedHyperLogLog) -> None:
        """Merge a compatible configured sketch into this sketch."""

        if not isinstance(other, HashedHyperLogLog):
            raise TypeError("Can only merge another HashedHyperLogLog sketch")
        if self.hash_configuration != other.hash_configuration:
            raise ValueError(
                "Cannot merge HyperLogLog sketches with different hash configurations"
            )
        self._hll.merge(other._hll)

    def copy(self) -> HashedHyperLogLog:
        """Return an independent configured sketch copy."""

        copied = HashedHyperLogLog(
            precision=self.precision,
            hash_configuration=self.hash_configuration,
        )
        copied._hll = self._hll.copy()
        return copied
