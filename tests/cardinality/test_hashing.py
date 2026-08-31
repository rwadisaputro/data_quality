from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
from uuid import UUID

import pytest

from data_quality.cardinality.hashing import (
    CANONICALISATION_VERSION,
    DEFAULT_HASH_CONFIGURATION,
    DEFAULT_HASH_SEED,
    HASH_ALGORITHM_ID,
    HASH_ENGINE_ID,
    HASH_WIDTH_BITS,
    NULL_POLICY,
    HashConfiguration,
    canonicalize_scalar,
    hash_canonical_bytes,
    hash_canonical_ndarray,
    hash_scalar,
)


def test_hash_metadata_is_explicit_and_versioned() -> None:
    configuration = DEFAULT_HASH_CONFIGURATION

    assert HASH_ALGORITHM_ID == "pandas-dtype-kernel-64-v2"
    assert HASH_ENGINE_ID == "pandas.util.hash_array+splitmix64"
    assert HASH_WIDTH_BITS == 64
    assert DEFAULT_HASH_SEED == 0
    assert CANONICALISATION_VERSION == "pandas-dtype-kernels-v2"
    assert NULL_POLICY == "exclude"
    assert configuration.algorithm_id == HASH_ALGORITHM_ID
    assert configuration.engine_id == HASH_ENGINE_ID
    assert configuration.width_bits == HASH_WIDTH_BITS
    assert configuration.canonicalisation_version == CANONICALISATION_VERSION
    assert configuration.null_policy == NULL_POLICY


def test_hash_configuration_accepts_integer_like_seed() -> None:
    class IntegerLike:
        def __index__(self) -> int:
            return 7

    assert HashConfiguration(seed=IntegerLike()).seed == 7  # type: ignore[arg-type]


@pytest.mark.parametrize("seed", [True, 1.5, "7"])
def test_hash_configuration_rejects_non_integer_seed(seed: object) -> None:
    with pytest.raises(TypeError):
        HashConfiguration(seed=seed)  # type: ignore[arg-type]


@pytest.mark.parametrize("seed", [-1, 1 << 64])
def test_hash_configuration_rejects_out_of_range_seed(seed: int) -> None:
    with pytest.raises(ValueError):
        HashConfiguration(seed=seed)


def test_canonicalization_is_type_aware() -> None:
    canonical_values = {
        canonicalize_scalar(True),
        canonicalize_scalar(1),
        canonicalize_scalar(1.0),
        canonicalize_scalar("1"),
    }

    assert len(canonical_values) == 4


def test_float_signed_zero_is_normalized() -> None:
    assert canonicalize_scalar(-0.0) == canonicalize_scalar(0.0)
    assert hash_scalar(-0.0) == hash_scalar(0.0)


def test_unicode_is_preserved_without_normalization() -> None:
    assert canonicalize_scalar("é") != canonicalize_scalar("e\u0301")


def test_decimal_equal_values_share_canonical_form() -> None:
    assert canonicalize_scalar(Decimal("1.0")) == canonicalize_scalar(Decimal("1.00"))
    assert canonicalize_scalar(Decimal("-0")) == canonicalize_scalar(Decimal("0"))


def test_aware_datetimes_are_normalized_to_utc() -> None:
    utc_value = datetime(2024, 1, 1, 12, tzinfo=timezone.utc)
    offset_value = datetime(2024, 1, 1, 7, tzinfo=timezone(timedelta(hours=-5)))

    assert canonicalize_scalar(utc_value) == canonicalize_scalar(offset_value)


def test_naive_and_aware_datetimes_remain_distinct() -> None:
    naive = datetime(2024, 1, 1, 12)
    aware = datetime(2024, 1, 1, 12, tzinfo=timezone.utc)

    assert canonicalize_scalar(naive) != canonicalize_scalar(aware)



def test_aware_times_are_normalized_to_utc_clock_time() -> None:
    utc_value = time(12, tzinfo=timezone.utc)
    offset_value = time(7, tzinfo=timezone(timedelta(hours=-5)))

    assert utc_value == offset_value
    assert canonicalize_scalar(utc_value) == canonicalize_scalar(offset_value)


def test_naive_and_aware_times_remain_distinct() -> None:
    naive = time(12)
    aware = time(12, tzinfo=timezone.utc)

    assert canonicalize_scalar(naive) != canonicalize_scalar(aware)

def test_supported_scalar_encodings_are_deterministic() -> None:
    values = [
        False,
        42,
        -3.5,
        "hello",
        b"bytes",
        Decimal("12.50"),
        datetime(2024, 1, 2, 3, 4, 5, 6),
        date(2024, 1, 2),
        time(3, 4, 5, 6),
        timedelta(days=2, microseconds=3),
        UUID("12345678-1234-5678-1234-567812345678"),
    ]

    for value in values:
        assert canonicalize_scalar(value) == canonicalize_scalar(value)
        assert hash_scalar(value) == hash_scalar(value)


@pytest.mark.parametrize("value", [None, float("nan"), Decimal("NaN")])
def test_null_like_scalars_are_rejected_by_canonicalizer(value: object) -> None:
    with pytest.raises(ValueError):
        canonicalize_scalar(value)


def test_unsupported_scalar_type_is_rejected() -> None:
    with pytest.raises(TypeError, match="Unsupported cardinality scalar type"):
        canonicalize_scalar([1, 2, 3])


def test_hash_canonical_bytes_requires_bytes() -> None:
    with pytest.raises(TypeError, match="canonical_value must be bytes"):
        hash_canonical_bytes(bytearray(b"abc"))  # type: ignore[arg-type]


def test_hash_seed_changes_hash_stream() -> None:
    first = hash_scalar("same", configuration=HashConfiguration(seed=1))
    second = hash_scalar("same", configuration=HashConfiguration(seed=2))

    assert first != second


def test_hash_values_fit_unsigned_64_bit_range() -> None:
    hashed = hash_scalar("bounded")

    assert 0 <= hashed < 1 << 64


def test_stable_hash_vectors() -> None:
    assert hash_scalar(False) == 13523928325204652336
    assert hash_scalar(True) == 3503252267930221076
    assert hash_scalar(1) == 3600679870756998109
    assert hash_scalar(1.5) == 12480509705983363768
    assert hash_scalar("abc") == 13727073273513852047
    assert hash_scalar(b"abc") == 683284965577791299


def test_complex_values_are_type_aware_and_normalize_signed_zero() -> None:
    assert canonicalize_scalar(complex(-0.0, 2.0)) == canonicalize_scalar(complex(0.0, 2.0))
    assert canonicalize_scalar(complex(1.0, 0.0)) != canonicalize_scalar(1.0)


def test_complex_nan_is_rejected() -> None:
    with pytest.raises(ValueError, match="Complex NaN"):
        canonicalize_scalar(complex(float("nan"), 1.0))


def test_hash_canonical_ndarray_returns_uint64_array() -> None:
    import numpy as np

    canonical = np.array([canonicalize_scalar(1), canonicalize_scalar("1")], dtype=object)
    hashes = hash_canonical_ndarray(canonical)

    assert isinstance(hashes, np.ndarray)
    assert hashes.dtype == np.uint64
    assert hashes.shape == (2,)
    assert int(hashes[0]) == hash_scalar(1)
    assert int(hashes[1]) == hash_scalar("1")


def test_hash_canonical_ndarray_rejects_non_bytes() -> None:
    import numpy as np

    with pytest.raises(TypeError, match="only bytes"):
        hash_canonical_ndarray(np.array([b"ok", "not-bytes"], dtype=object))


def test_hash_configuration_serializes_all_hash_parameters() -> None:
    metadata = HashConfiguration(seed=99).as_dict()

    assert metadata == {
        "algorithm_id": HASH_ALGORITHM_ID,
        "engine_id": HASH_ENGINE_ID,
        "width_bits": HASH_WIDTH_BITS,
        "seed": 99,
        "canonicalisation_version": CANONICALISATION_VERSION,
        "null_policy": NULL_POLICY,
    }


def test_hash_configuration_can_describe_future_backend_hash_engines() -> None:
    configuration = HashConfiguration(
        algorithm_id="future-backend-hash-v1",
        engine_id="future.ndarray.hash",
    )

    assert configuration.algorithm_id == "future-backend-hash-v1"
    assert configuration.engine_id == "future.ndarray.hash"


def test_pandas_ndarray_hasher_rejects_mismatched_engine_metadata() -> None:
    import numpy as np

    configuration = HashConfiguration(
        algorithm_id="future-backend-hash-v1",
        engine_id="future.ndarray.hash",
    )
    canonical = np.array([canonicalize_scalar(1)], dtype=object)

    with pytest.raises(ValueError, match="configured pandas ndarray hash engine"):
        hash_canonical_ndarray(canonical, configuration=configuration)


def test_hash_configuration_rejects_non_64_bit_width_and_incompatible_null_policy() -> None:
    with pytest.raises(ValueError, match="64 bits"):
        HashConfiguration(width_bits=32)
    with pytest.raises(ValueError, match="null policy"):
        HashConfiguration(null_policy="include")


def test_hash_configuration_rejects_empty_metadata_identifiers() -> None:
    with pytest.raises(TypeError, match="algorithm_id"):
        HashConfiguration(algorithm_id="")
