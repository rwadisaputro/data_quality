from __future__ import annotations

import pytest

from data_quality.cardinality import HashConfiguration, HashedHyperLogLog
from data_quality.cardinality.hashing import canonicalize_scalar, hash_scalar
from data_quality.cardinality.hll import HyperLogLog


def test_configured_sketch_exposes_hll_metadata() -> None:
    sketch = HashedHyperLogLog(precision=10, hash_configuration=HashConfiguration(seed=7))

    assert sketch.precision == 10
    assert sketch.register_count == 1 << 10
    assert sketch.relative_standard_error == pytest.approx(1.04 / (1 << 10) ** 0.5)
    assert sketch.hash_configuration.seed == 7
    assert sketch.is_empty


def test_configured_sketch_rejects_wrong_configuration_type() -> None:
    with pytest.raises(TypeError, match="HashConfiguration"):
        HashedHyperLogLog(hash_configuration=object())  # type: ignore[arg-type]


def test_add_hash_delegates_to_hll_core() -> None:
    configured = HashedHyperLogLog(precision=10)
    raw = HyperLogLog(precision=10)
    hashed = hash_scalar("value")

    configured.add_hash(hashed)
    raw.add_hash(hashed)

    assert configured.registers == raw.registers


def test_add_hashes_delegates_to_hll_core() -> None:
    values = [hash_scalar(value) for value in ["a", "b", "c", "a"]]
    configured = HashedHyperLogLog(precision=10)
    raw = HyperLogLog(precision=10)

    configured.add_hashes(values)
    raw.add_hashes(values)

    assert configured.registers == raw.registers


def test_add_canonical_bytes_uses_bound_hash_configuration() -> None:
    configuration = HashConfiguration(seed=17)
    sketch = HashedHyperLogLog(precision=10, hash_configuration=configuration)
    expected = HyperLogLog(precision=10)
    canonical = canonicalize_scalar("value")

    sketch.add_canonical_bytes(canonical)
    expected.add_hash(hash_scalar("value", configuration=configuration))

    assert sketch.registers == expected.registers


def test_compatible_configured_sketches_merge() -> None:
    first = HashedHyperLogLog(precision=10, hash_configuration=HashConfiguration(seed=3))
    second = HashedHyperLogLog(precision=10, hash_configuration=HashConfiguration(seed=3))
    union = HashedHyperLogLog(precision=10, hash_configuration=HashConfiguration(seed=3))

    first.add_hashes(
        hash_scalar(value, configuration=first.hash_configuration) for value in range(100)
    )
    second.add_hashes(
        hash_scalar(value, configuration=second.hash_configuration) for value in range(50, 150)
    )
    union.add_hashes(
        hash_scalar(value, configuration=union.hash_configuration) for value in range(150)
    )

    first.merge(second)

    assert first.registers == union.registers


def test_merge_rejects_different_hash_configuration() -> None:
    first = HashedHyperLogLog(hash_configuration=HashConfiguration(seed=1))
    second = HashedHyperLogLog(hash_configuration=HashConfiguration(seed=2))

    with pytest.raises(ValueError, match="different hash configurations"):
        first.merge(second)


def test_merge_rejects_wrong_type() -> None:
    sketch = HashedHyperLogLog()

    with pytest.raises(TypeError, match="HashedHyperLogLog"):
        sketch.merge(HyperLogLog())  # type: ignore[arg-type]


def test_merge_rejects_different_precision_through_core_validation() -> None:
    first = HashedHyperLogLog(precision=10)
    second = HashedHyperLogLog(precision=11)

    with pytest.raises(ValueError, match="different precisions"):
        first.merge(second)


def test_copy_is_independent_and_retains_configuration() -> None:
    original = HashedHyperLogLog(precision=10, hash_configuration=HashConfiguration(seed=9))
    original.add_hash(hash_scalar("a", configuration=original.hash_configuration))

    copied = original.copy()
    copied.add_hash(hash_scalar("b", configuration=copied.hash_configuration))

    assert copied.hash_configuration == original.hash_configuration
    assert copied.registers != original.registers


def test_add_hash_array_delegates_to_hll_core() -> None:
    import numpy as np

    values = np.array([hash_scalar(value) for value in ["a", "b", "c", "a"]], dtype=np.uint64)
    configured = HashedHyperLogLog(precision=10)
    raw = HyperLogLog(precision=10)

    configured.add_hash_array(values)
    raw.add_hash_array(values)

    assert configured.registers == raw.registers
