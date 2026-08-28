"""Stable canonical scalar encoding and 64-bit hashing for cardinality sketches.

The exact-count and approximate-count paths must share this canonical representation.
Backend adapters may normalise native scalar objects before calling these helpers, but
must not invent a separate equality or hashing policy.
"""

from __future__ import annotations

import hashlib
import math
import operator
import struct
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
from uuid import UUID

HASH_ALGORITHM_ID = "blake2b-64"
HASH_WIDTH_BITS = 64
DEFAULT_HASH_SEED = 0
CANONICALISATION_VERSION = "scalar-v1"
NULL_POLICY = "exclude"
_HASH_MASK = (1 << HASH_WIDTH_BITS) - 1
_HASH_PERSON = b"dq-card-v1"


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
    algorithm_id: str = field(init=False, default=HASH_ALGORITHM_ID)
    width_bits: int = field(init=False, default=HASH_WIDTH_BITS)
    canonicalisation_version: str = field(init=False, default=CANONICALISATION_VERSION)
    null_policy: str = field(init=False, default=NULL_POLICY)

    def __post_init__(self) -> None:
        object.__setattr__(self, "seed", _coerce_seed(self.seed))


DEFAULT_HASH_CONFIGURATION = HashConfiguration()


def canonicalize_scalar(value: object) -> bytes:
    """Encode one non-null logical scalar into deterministic type-aware bytes.

    Semantics deliberately distinguish booleans, integers, floats, and strings.
    Floating-point signed zero is normalised because ``-0.0 == 0.0``. NaN and
    ``None`` are rejected because the dataframe adapter excludes null values before
    cardinality hashing.
    """

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
        # Unicode code points are preserved exactly in v1; no NFC/NFKC normalisation.
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


def hash_canonical_bytes(
    canonical_value: bytes,
    *,
    configuration: HashConfiguration = DEFAULT_HASH_CONFIGURATION,
) -> int:
    """Hash canonical bytes into one deterministic unsigned 64-bit integer."""

    if not isinstance(canonical_value, bytes):
        raise TypeError("canonical_value must be bytes")

    digest = hashlib.blake2b(
        canonical_value,
        digest_size=HASH_WIDTH_BITS // 8,
        key=configuration.seed.to_bytes(8, "big", signed=False),
        person=_HASH_PERSON,
    ).digest()
    return int.from_bytes(digest, "big", signed=False)


def hash_scalar(
    value: object,
    *,
    configuration: HashConfiguration = DEFAULT_HASH_CONFIGURATION,
) -> int:
    """Canonicalise and hash one supported non-null scalar."""

    return hash_canonical_bytes(
        canonicalize_scalar(value),
        configuration=configuration,
    )

