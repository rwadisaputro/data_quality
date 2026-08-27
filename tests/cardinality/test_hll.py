"""Tests for the dependency-free classic HyperLogLog core."""

from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Iterable

import pytest

from data_quality.cardinality import (
    DEFAULT_PRECISION,
    HASH_WIDTH_BITS,
    MAX_PRECISION,
    MIN_PRECISION,
    HyperLogLog,
)

_HASH_MASK = (1 << HASH_WIDTH_BITS) - 1


class _IndexablePrecision:
    def __init__(self, value: int) -> None:
        self._value = value

    def __index__(self) -> int:
        return self._value


def _stable_hash(value: int) -> int:
    payload = value.to_bytes(8, byteorder="big", signed=False)
    digest = hashlib.blake2b(payload, digest_size=8, person=b"dq-hll-v1").digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


def _splitmix64(value: int, seed: int = 0) -> int:
    z = (value + seed + 0x9E3779B97F4A7C15) & _HASH_MASK
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _HASH_MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _HASH_MASK
    return z ^ (z >> 31)


def _hash_for_rank(register_index: int, precision: int, rank: int) -> int:
    suffix_bits = HASH_WIDTH_BITS - precision
    if rank == suffix_bits + 1:
        suffix = 0
    else:
        suffix = 1 << (suffix_bits - rank)
    return (register_index << suffix_bits) | suffix


def _reference_rank(suffix: int, suffix_bits: int) -> int:
    if suffix == 0:
        return suffix_bits + 1
    return f"{suffix:0{suffix_bits}b}".index("1") + 1


def _reference_registers(hash_values: Iterable[int], precision: int) -> bytes:
    register_count = 1 << precision
    suffix_bits = HASH_WIDTH_BITS - precision
    suffix_mask = (1 << suffix_bits) - 1
    registers = [0] * register_count

    for value in hash_values:
        register_index = value >> suffix_bits
        suffix = value & suffix_mask
        rank = _reference_rank(suffix, suffix_bits)
        registers[register_index] = max(registers[register_index], rank)

    return bytes(registers)


def _reference_alpha(register_count: int) -> float:
    if register_count == 16:
        return 0.673
    if register_count == 32:
        return 0.697
    if register_count == 64:
        return 0.709
    return 0.7213 / (1.0 + 1.079 / register_count)


def _reference_estimate(registers: bytes) -> float:
    register_count = len(registers)
    harmonic_sum = sum(2.0 ** (-register_value) for register_value in registers)
    raw_estimate = _reference_alpha(register_count) * register_count**2 / harmonic_sum
    zero_registers = registers.count(0)

    if raw_estimate <= 2.5 * register_count and zero_registers > 0:
        return register_count * math.log(register_count / zero_registers)
    return raw_estimate


def test_default_configuration() -> None:
    sketch = HyperLogLog()

    assert sketch.precision == DEFAULT_PRECISION
    assert sketch.register_count == 1 << DEFAULT_PRECISION
    assert sketch.hash_width_bits == HASH_WIDTH_BITS
    assert sketch.zero_register_count == sketch.register_count
    assert sketch.is_empty
    assert sketch.estimate() == 0.0


@pytest.mark.parametrize("precision", [MIN_PRECISION, 10, 14, MAX_PRECISION])
def test_supported_precision_allocates_expected_registers(precision: int) -> None:
    sketch = HyperLogLog(precision)

    assert sketch.register_count == 1 << precision
    assert len(sketch.registers) == 1 << precision
    assert sketch.relative_standard_error == pytest.approx(1.04 / math.sqrt(1 << precision))


def test_integer_like_precision_is_accepted() -> None:
    sketch = HyperLogLog(_IndexablePrecision(12))  # type: ignore[arg-type]

    assert sketch.precision == 12
    assert sketch.register_count == 1 << 12


@pytest.mark.parametrize("precision", [MIN_PRECISION - 1, MAX_PRECISION + 1])
def test_out_of_range_precision_is_rejected(precision: int) -> None:
    with pytest.raises(ValueError, match="precision must be between"):
        HyperLogLog(precision)


@pytest.mark.parametrize("precision", [True, 14.0, "14"])
def test_non_integer_precision_is_rejected(precision: object) -> None:
    with pytest.raises(TypeError, match="precision must be"):
        HyperLogLog(precision)  # type: ignore[arg-type]


def test_hash_high_bits_select_register_and_suffix_sets_rank() -> None:
    sketch = HyperLogLog(precision=4)
    suffix_bits = HASH_WIDTH_BITS - sketch.precision
    register_index = 10

    rank_one_hash = (register_index << suffix_bits) | (1 << (suffix_bits - 1))
    rank_three_hash = (register_index << suffix_bits) | (1 << (suffix_bits - 3))

    sketch.add_hash(rank_one_hash)
    assert sketch.registers[register_index] == 1

    sketch.add_hash(rank_three_hash)
    assert sketch.registers[register_index] == 3

    sketch.add_hash(rank_one_hash)
    assert sketch.registers[register_index] == 3


def test_all_zero_suffix_produces_maximum_rank() -> None:
    sketch = HyperLogLog(precision=4)
    suffix_bits = HASH_WIDTH_BITS - sketch.precision
    register_index = 7

    sketch.add_hash(register_index << suffix_bits)

    assert sketch.registers[register_index] == suffix_bits + 1


@pytest.mark.parametrize("precision", [4, 10, 14, 20])
def test_rank_matches_independent_binary_reference(precision: int) -> None:
    sketch = HyperLogLog(precision)
    suffix_bits = HASH_WIDTH_BITS - precision
    suffix_mask = (1 << suffix_bits) - 1
    rng = random.Random(20260827 + precision)
    suffixes = [0, 1, 1 << (suffix_bits - 1), suffix_mask]
    suffixes.extend(rng.getrandbits(suffix_bits) for _ in range(500))

    for suffix in suffixes:
        assert sketch._rank(suffix) == _reference_rank(suffix, suffix_bits)


@pytest.mark.parametrize("precision", [4, 10, 14, 20])
def test_register_updates_match_independent_reference(precision: int) -> None:
    rng = random.Random(0xC0FFEE + precision)
    hashes = [rng.getrandbits(HASH_WIDTH_BITS) for _ in range(5_000)]
    sketch = HyperLogLog(precision)

    sketch.add_hashes(hashes)

    assert sketch.registers == _reference_registers(hashes, precision)


@pytest.mark.parametrize("precision", [4, 10, 14, 20])
def test_register_values_never_exceed_maximum_rank(precision: int) -> None:
    sketch = HyperLogLog(precision)
    suffix_bits = HASH_WIDTH_BITS - precision
    rng = random.Random(0xBAD5EED + precision)
    sketch.add_hashes(rng.getrandbits(HASH_WIDTH_BITS) for _ in range(10_000))
    sketch.add_hash(_hash_for_rank(0, precision, suffix_bits + 1))

    assert max(sketch.registers) <= suffix_bits + 1
    assert sketch.registers[0] == suffix_bits + 1


@pytest.mark.parametrize("hash_value", [-1, 1 << HASH_WIDTH_BITS])
def test_hash_outside_unsigned_64_bit_range_is_rejected(hash_value: int) -> None:
    sketch = HyperLogLog()

    with pytest.raises(ValueError, match="unsigned 64-bit"):
        sketch.add_hash(hash_value)


@pytest.mark.parametrize("hash_value", [True, 1.5, "123"])
def test_non_integer_hash_is_rejected(hash_value: object) -> None:
    sketch = HyperLogLog()

    with pytest.raises(TypeError, match="hash values must be"):
        sketch.add_hash(hash_value)  # type: ignore[arg-type]


def test_add_hashes_matches_repeated_single_updates() -> None:
    hashes = [_stable_hash(value) for value in range(10_000)]
    bulk = HyperLogLog(precision=12)
    incremental = HyperLogLog(precision=12)

    bulk.add_hashes(hashes)
    for hash_value in hashes:
        incremental.add_hash(hash_value)

    assert bulk.registers == incremental.registers
    assert bulk.estimate() == incremental.estimate()


def test_duplicate_updates_are_idempotent() -> None:
    sketch = HyperLogLog(precision=10)
    value_hash = _stable_hash(1234)

    sketch.add_hash(value_hash)
    registers_after_first_update = sketch.registers
    estimate_after_first_update = sketch.estimate()

    sketch.add_hashes([value_hash] * 1_000)

    assert sketch.registers == registers_after_first_update
    assert sketch.estimate() == estimate_after_first_update


@pytest.mark.parametrize("cardinality", [1, 10, 100, 1_000, 10_000, 50_000, 100_000])
def test_estimate_tracks_known_cardinality_at_p14(cardinality: int) -> None:
    sketch = HyperLogLog(precision=14)
    sketch.add_hashes(_stable_hash(value) for value in range(cardinality))

    relative_error = abs(sketch.estimate() - cardinality) / cardinality

    assert relative_error < 2.0 * sketch.relative_standard_error


@pytest.mark.parametrize("precision", [8, 10, 12, 14])
def test_multiseed_rmse_is_consistent_with_classic_hll_rse(precision: int) -> None:
    register_count = 1 << precision
    cardinality = 8 * register_count
    relative_errors: list[float] = []

    for trial in range(8):
        seed = (trial * 0xD1B54A32D192ED03) & _HASH_MASK
        sketch = HyperLogLog(precision)
        sketch.add_hashes(_splitmix64(value, seed) for value in range(cardinality))
        relative_errors.append((sketch.estimate() - cardinality) / cardinality)

    rmse = math.sqrt(
        sum(relative_error**2 for relative_error in relative_errors) / len(relative_errors)
    )
    mean_bias = sum(relative_errors) / len(relative_errors)
    expected_rse = 1.04 / math.sqrt(register_count)

    assert rmse < 1.5 * expected_rse
    assert abs(mean_bias) < expected_rse


def test_estimate_matches_independent_reference_estimator() -> None:
    hashes = [_splitmix64(value, seed=12345) for value in range(50_000)]
    sketch = HyperLogLog(precision=12)
    sketch.add_hashes(hashes)

    assert sketch.estimate() == pytest.approx(_reference_estimate(sketch.registers))


def test_linear_counting_is_used_below_small_range_threshold() -> None:
    sketch = HyperLogLog(precision=4)
    for register_index in range(12):
        sketch.add_hash(_hash_for_rank(register_index, precision=4, rank=4))

    register_count = sketch.register_count
    raw_estimate = _reference_alpha(register_count) * register_count**2 / sum(
        2.0 ** (-register_value) for register_value in sketch.registers
    )
    expected_linear_count = register_count * math.log(
        register_count / sketch.zero_register_count
    )

    assert raw_estimate <= 2.5 * register_count
    assert sketch.zero_register_count > 0
    assert sketch.estimate() == pytest.approx(expected_linear_count)


def test_raw_estimator_is_used_above_small_range_threshold_with_zero_registers() -> None:
    sketch = HyperLogLog(precision=4)
    for register_index in range(13):
        sketch.add_hash(_hash_for_rank(register_index, precision=4, rank=4))

    register_count = sketch.register_count
    expected_raw_estimate = _reference_alpha(register_count) * register_count**2 / sum(
        2.0 ** (-register_value) for register_value in sketch.registers
    )

    assert expected_raw_estimate > 2.5 * register_count
    assert sketch.zero_register_count > 0
    assert sketch.estimate() == pytest.approx(expected_raw_estimate)


def test_merge_matches_single_sketch_over_union() -> None:
    left = HyperLogLog(precision=12)
    right = HyperLogLog(precision=12)
    full = HyperLogLog(precision=12)

    left.add_hashes(_stable_hash(value) for value in range(0, 20_000))
    right.add_hashes(_stable_hash(value) for value in range(10_000, 30_000))
    full.add_hashes(_stable_hash(value) for value in range(0, 30_000))

    left.merge(right)

    assert left.registers == full.registers
    assert left.estimate() == full.estimate()


def test_merge_is_commutative() -> None:
    left = HyperLogLog(precision=10)
    right = HyperLogLog(precision=10)
    left.add_hashes(_stable_hash(value) for value in range(0, 5_000))
    right.add_hashes(_stable_hash(value) for value in range(2_500, 8_000))

    left_then_right = left.copy()
    right_then_left = right.copy()
    left_then_right.merge(right)
    right_then_left.merge(left)

    assert left_then_right.registers == right_then_left.registers


def test_merge_is_associative() -> None:
    first = HyperLogLog(precision=10)
    second = HyperLogLog(precision=10)
    third = HyperLogLog(precision=10)
    first.add_hashes(_stable_hash(value) for value in range(0, 4_000))
    second.add_hashes(_stable_hash(value) for value in range(2_000, 6_000))
    third.add_hashes(_stable_hash(value) for value in range(5_000, 9_000))

    left_grouped = first.copy()
    left_grouped.merge(second)
    left_grouped.merge(third)

    second_and_third = second.copy()
    second_and_third.merge(third)
    right_grouped = first.copy()
    right_grouped.merge(second_and_third)

    assert left_grouped.registers == right_grouped.registers


def test_merge_is_idempotent() -> None:
    sketch = HyperLogLog(precision=10)
    sketch.add_hashes(_stable_hash(value) for value in range(5_000))
    before = sketch.registers

    sketch.merge(sketch)

    assert sketch.registers == before


def test_merge_rejects_different_precisions() -> None:
    left = HyperLogLog(precision=12)
    right = HyperLogLog(precision=14)

    with pytest.raises(ValueError, match="different precisions"):
        left.merge(right)


def test_merge_rejects_non_hll_object() -> None:
    sketch = HyperLogLog()

    with pytest.raises(TypeError, match="another HyperLogLog"):
        sketch.merge(object())  # type: ignore[arg-type]


def test_copy_is_independent() -> None:
    original = HyperLogLog(precision=10)
    original.add_hashes(_stable_hash(value) for value in range(1_000))

    copied = original.copy()
    zero_index = original.registers.index(0)
    suffix_bits = HASH_WIDTH_BITS - original.precision
    copied.add_hash((zero_index << suffix_bits) | (1 << (suffix_bits - 1)))

    assert copied is not original
    assert copied.registers != original.registers
    assert original.registers[zero_index] == 0
    assert copied.registers[zero_index] == 1


@pytest.mark.parametrize(
    ("precision", "expected_alpha"),
    [
        (4, 0.673),
        (5, 0.697),
        (6, 0.709),
        (7, 0.7213 / (1.0 + 1.079 / 128)),
    ],
)
def test_estimator_uses_standard_alpha_constants(
    precision: int,
    expected_alpha: float,
) -> None:
    sketch = HyperLogLog(precision=precision)
    suffix_bits = HASH_WIDTH_BITS - precision

    for register_index in range(sketch.register_count):
        sketch.add_hash((register_index << suffix_bits) | (1 << (suffix_bits - 1)))

    expected_raw_estimate = 2.0 * expected_alpha * sketch.register_count
    assert sketch.estimate() == pytest.approx(expected_raw_estimate)


def test_classic_64_bit_estimator_does_not_apply_legacy_large_range_correction() -> None:
    sketch = HyperLogLog(precision=4)
    target_rank = 56

    for register_index in range(sketch.register_count):
        sketch.add_hash(_hash_for_rank(register_index, precision=4, rank=target_rank))

    expected_raw_estimate = 0.673 * sketch.register_count * (2.0**target_rank)
    hash_space = float(1 << HASH_WIDTH_BITS)

    assert expected_raw_estimate > hash_space / 30.0
    assert expected_raw_estimate < hash_space
    assert sketch.estimate() == pytest.approx(expected_raw_estimate)


def test_classic_estimator_remains_finite_at_maximum_register_rank() -> None:
    sketch = HyperLogLog(precision=4)
    maximum_rank = HASH_WIDTH_BITS - sketch.precision + 1

    for register_index in range(sketch.register_count):
        sketch.add_hash(_hash_for_rank(register_index, precision=4, rank=maximum_rank))

    assert math.isfinite(sketch.estimate())
    assert sketch.estimate() > float(1 << HASH_WIDTH_BITS)
