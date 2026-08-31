"""Stable scalar encoding and vectorised 64-bit ndarray hashing.

Pandas homogeneous dtype kernels hash entire NumPy-compatible arrays in bulk.
Mixed/object fallback values still use the stable scalar canonical byte contract,
then those canonical bytes are hashed as one ndarray batch.
"""

from __future__ import annotations

import hashlib
import math
import operator
import struct
from dataclasses import dataclass, replace
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
from importlib import import_module
from typing import Any
from uuid import UUID

HASH_ALGORITHM_ID = "pandas-dtype-kernel-64-v2"
HASH_ENGINE_ID = "pandas.util.hash_array+splitmix64"
HASH_WIDTH_BITS = 64
DEFAULT_HASH_SEED = 0
CANONICALISATION_VERSION = "pandas-dtype-kernels-v2"
NULL_POLICY = "exclude"
CANONICAL_BYTES_KERNEL_ID = "canonical-bytes-v1"
_HASH_MASK = (1 << HASH_WIDTH_BITS) - 1
_SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
_SPLITMIX_MUL_1 = 0xBF58476D1CE4E5B9
_SPLITMIX_MUL_2 = 0x94D049BB133111EB


def _coerce_seed(seed: int) -> int:
    """Validate and normalise an unsigned 64-bit hash seed."""

    if isinstance(seed, bool):
        raise TypeError("Hash seed must be an integer, not a boolean")

    try:
        value = operator.index(seed)
    except TypeError as exc:
        raise TypeError("Hash seed must be integer-like") from exc

    if not 0 <= value <= _HASH_MASK:
        raise ValueError(f"Hash seed must be between 0 and {_HASH_MASK}, received {value}")
    return value


@dataclass(frozen=True, slots=True)
class HashConfiguration:
    """Versioned hashing metadata required for safe sketch compatibility checks."""

    seed: int = DEFAULT_HASH_SEED
    algorithm_id: str = HASH_ALGORITHM_ID
    engine_id: str = HASH_ENGINE_ID
    width_bits: int = HASH_WIDTH_BITS
    canonicalisation_version: str = CANONICALISATION_VERSION
    null_policy: str = NULL_POLICY

    def __post_init__(self) -> None:
        object.__setattr__(self, "seed", _coerce_seed(self.seed))
        for name in ("algorithm_id", "engine_id", "canonicalisation_version", "null_policy"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise TypeError(f"{name} must be a non-empty string")
        if self.width_bits != HASH_WIDTH_BITS:
            raise ValueError(f"Hash width must be {HASH_WIDTH_BITS} bits")
        if self.null_policy != NULL_POLICY:
            raise ValueError(f"Hash null policy must be {NULL_POLICY!r}")

    def as_dict(self) -> dict[str, int | str]:
        """Return JSON-safe hashing and canonicalisation metadata."""

        return {
            "algorithm_id": self.algorithm_id,
            "engine_id": self.engine_id,
            "width_bits": self.width_bits,
            "seed": self.seed,
            "canonicalisation_version": self.canonicalisation_version,
            "null_policy": self.null_policy,
        }


DEFAULT_HASH_CONFIGURATION = HashConfiguration()


def hash_configuration_for_kernel(
    configuration: HashConfiguration,
    *,
    kernel_id: str,
) -> HashConfiguration:
    """Bind generic hash metadata to one dtype kernel for safe sketch merging."""

    if not isinstance(configuration, HashConfiguration):
        raise TypeError("configuration must be a HashConfiguration")
    if (
        configuration.algorithm_id != HASH_ALGORITHM_ID
        or configuration.engine_id != HASH_ENGINE_ID
    ):
        raise ValueError("hash_ndarray only supports the configured pandas ndarray hash engine")
    if not isinstance(kernel_id, str) or not kernel_id:
        raise TypeError("kernel_id must be a non-empty string")
    return replace(
        configuration,
        canonicalisation_version=(
            f"{configuration.canonicalisation_version};kernel={kernel_id}"
        ),
    )


def canonicalize_scalar(value: object) -> bytes:
    """Encode one non-null logical scalar into deterministic type-aware bytes."""

    if value is None:
        raise ValueError("Null values must be excluded before cardinality hashing")

    if isinstance(value, bool):
        return b"b\x01" if value else b"b\x00"

    if isinstance(value, int):
        return b"i" + str(value).encode("ascii")

    if isinstance(value, float):
        if math.isnan(value):
            raise ValueError("NaN values must be excluded before cardinality hashing")
        normalised = 0.0 if value == 0.0 else value
        return b"f" + struct.pack(">d", normalised)

    if isinstance(value, complex):
        if math.isnan(value.real) or math.isnan(value.imag):
            raise ValueError("Complex NaN values must be excluded before cardinality hashing")
        real = 0.0 if value.real == 0.0 else value.real
        imag = 0.0 if value.imag == 0.0 else value.imag
        return b"c" + struct.pack(">dd", real, imag)

    if isinstance(value, str):
        return b"s" + value.encode("utf-8")

    if isinstance(value, (bytes, bytearray, memoryview)):
        return b"y" + bytes(value)

    if isinstance(value, Decimal):
        if value.is_nan():
            raise ValueError("Decimal NaN values must be excluded before cardinality hashing")
        if value.is_zero():
            value = Decimal(0)
        normalised = value.normalize()
        return b"d" + str(normalised).encode("ascii")

    if isinstance(value, datetime):
        if value.tzinfo is not None and value.utcoffset() is not None:
            value = value.astimezone(timezone.utc)
            return b"ta" + value.isoformat().encode("ascii")
        return b"tn" + value.isoformat().encode("ascii")

    if isinstance(value, date):
        return b"D" + value.isoformat().encode("ascii")

    if isinstance(value, time):
        offset = value.utcoffset() if value.tzinfo is not None else None
        if offset is not None:
            local_microseconds = (
                (value.hour * 3_600 + value.minute * 60 + value.second) * 1_000_000
                + value.microsecond
            )
            offset_microseconds = int(offset.total_seconds() * 1_000_000)
            utc_microseconds = (local_microseconds - offset_microseconds) % 86_400_000_000
            return b"Ta" + str(utc_microseconds).encode("ascii")
        return b"Tn" + value.isoformat().encode("ascii")

    if isinstance(value, timedelta):
        total_nanoseconds = (
            (value.days * 86_400 + value.seconds) * 1_000_000_000
            + value.microseconds * 1_000
        )
        return b"u" + str(total_nanoseconds).encode("ascii")

    if isinstance(value, UUID):
        return b"U" + value.bytes

    raise TypeError(
        "Unsupported cardinality scalar type "
        f"{type(value).__module__}.{type(value).__qualname__}"
    )


def hash_ndarray(
    values: object,
    *,
    kernel_id: str,
    configuration: HashConfiguration = DEFAULT_HASH_CONFIGURATION,
) -> Any:
    """Hash a one-dimensional dtype-specialised ndarray into ``np.uint64`` in bulk.

    ``pandas.util.hash_array`` supplies the native vectorised base hash. A vectorised
    SplitMix64 finaliser incorporates both the configured seed and a stable kernel salt,
    ensuring numeric dtypes are seeded even when pandas ignores ``hash_key`` for them.
    """

    if not isinstance(configuration, HashConfiguration):
        raise TypeError("configuration must be a HashConfiguration")
    if (
        configuration.algorithm_id != HASH_ALGORITHM_ID
        or configuration.engine_id != HASH_ENGINE_ID
    ):
        raise ValueError("hash_ndarray only supports the configured pandas ndarray hash engine")
    if not isinstance(kernel_id, str) or not kernel_id:
        raise TypeError("kernel_id must be a non-empty string")

    pandas_module, numpy_module = _load_array_hash_dependencies()
    array = numpy_module.asarray(values)
    if array.ndim != 1:
        raise ValueError("values must be a one-dimensional ndarray")
    if array.size == 0:
        return numpy_module.empty(0, dtype=numpy_module.uint64)

    hash_key = f"{configuration.seed:016x}"
    base_hashes = pandas_module.util.hash_array(
        array,
        encoding="utf8",
        hash_key=hash_key,
        categorize=False,
    )
    hashes = numpy_module.asarray(base_hashes, dtype=numpy_module.uint64).copy()
    hashes ^= numpy_module.uint64(configuration.seed)
    hashes ^= numpy_module.uint64(_kernel_salt(kernel_id))
    return _splitmix64_array(hashes, numpy_module=numpy_module)


def hash_canonical_ndarray(
    canonical_values: object,
    *,
    configuration: HashConfiguration = DEFAULT_HASH_CONFIGURATION,
) -> Any:
    """Hash a one-dimensional ndarray of canonical bytes in one bulk operation."""

    _, numpy_module = _load_array_hash_dependencies()
    values = numpy_module.asarray(canonical_values, dtype=object)
    if values.ndim != 1:
        raise ValueError("canonical_values must be a one-dimensional ndarray")
    if values.size == 0:
        return numpy_module.empty(0, dtype=numpy_module.uint64)

    is_bytes = numpy_module.frompyfunc(lambda value: isinstance(value, bytes), 1, 1)(values)
    if not bool(numpy_module.asarray(is_bytes, dtype=bool).all()):
        raise TypeError("canonical_values must contain only bytes")

    return hash_ndarray(
        values,
        kernel_id=CANONICAL_BYTES_KERNEL_ID,
        configuration=configuration,
    )


def hash_canonical_bytes(
    canonical_value: bytes,
    *,
    configuration: HashConfiguration = DEFAULT_HASH_CONFIGURATION,
) -> int:
    """Compatibility wrapper that routes one canonical value through ndarray hashing."""

    if not isinstance(canonical_value, bytes):
        raise TypeError("canonical_value must be bytes")

    _, numpy_module = _load_array_hash_dependencies()
    values = numpy_module.empty(1, dtype=object)
    values[0] = canonical_value
    return int(hash_canonical_ndarray(values, configuration=configuration)[0])


def hash_scalar(
    value: object,
    *,
    configuration: HashConfiguration = DEFAULT_HASH_CONFIGURATION,
) -> int:
    """Compatibility wrapper for scalar fallback and direct tests."""

    return hash_canonical_bytes(
        canonicalize_scalar(value),
        configuration=configuration,
    )


def _splitmix64_array(values: Any, *, numpy_module: Any) -> Any:
    values = values + numpy_module.uint64(_SPLITMIX_GAMMA)
    values = (values ^ (values >> numpy_module.uint64(30))) * numpy_module.uint64(
        _SPLITMIX_MUL_1
    )
    values = (values ^ (values >> numpy_module.uint64(27))) * numpy_module.uint64(
        _SPLITMIX_MUL_2
    )
    return numpy_module.asarray(
        values ^ (values >> numpy_module.uint64(31)),
        dtype=numpy_module.uint64,
    )


def _kernel_salt(kernel_id: str) -> int:
    digest = hashlib.blake2b(
        kernel_id.encode("utf-8"),
        digest_size=8,
        person=b"dq-hll-k",
    ).digest()
    return int.from_bytes(digest, "big")


def _load_array_hash_dependencies() -> tuple[Any, Any]:
    """Load optional pandas/NumPy dependencies only when ndarray hashing is requested."""

    try:
        pandas_module = import_module("pandas")
        numpy_module = import_module("numpy")
    except ImportError as exc:
        raise ImportError(
            "Array cardinality hashing requires the optional 'pandas' dependency"
        ) from exc
    return pandas_module, numpy_module
