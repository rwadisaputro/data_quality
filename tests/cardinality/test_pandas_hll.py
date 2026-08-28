from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from data_quality.cardinality import HashConfiguration
from data_quality.cardinality.pandas import (
    DEFAULT_PANDAS_CHUNK_SIZE,
    iter_pandas_hashes,
    pandas_hll_nunique,
    pandas_hll_sketch,
)


def test_default_chunk_size_is_bounded() -> None:
    assert DEFAULT_PANDAS_CHUNK_SIZE == 65_536


def test_iter_pandas_hashes_rejects_non_series() -> None:
    with pytest.raises(TypeError, match="pandas.Series"):
        list(iter_pandas_hashes(pd.DataFrame({"value": [1]})))


@pytest.mark.parametrize("chunk_size", [True, 1.5, "10"])
def test_invalid_chunk_size_type_is_rejected(chunk_size: object) -> None:
    with pytest.raises(TypeError):
        list(iter_pandas_hashes(pd.Series([1]), chunk_size=chunk_size))  # type: ignore[arg-type]


@pytest.mark.parametrize("chunk_size", [0, -1])
def test_non_positive_chunk_size_is_rejected(chunk_size: int) -> None:
    with pytest.raises(ValueError):
        list(iter_pandas_hashes(pd.Series([1]), chunk_size=chunk_size))


def test_numpy_integer_chunk_size_is_accepted() -> None:
    hashes = list(iter_pandas_hashes(pd.Series([1, 2]), chunk_size=np.int64(1)))

    assert len(hashes) == 2


def test_empty_series_estimates_zero() -> None:
    assert pandas_hll_nunique(pd.Series([], dtype="object")) == 0.0


def test_all_null_values_are_excluded() -> None:
    series = pd.Series([None, np.nan, pd.NA, pd.NaT], dtype="object")

    assert list(iter_pandas_hashes(series)) == []
    assert pandas_hll_nunique(series) == 0.0


def test_duplicate_values_hash_identically() -> None:
    hashes = list(iter_pandas_hashes(pd.Series(["same", "same", "different"])))

    assert hashes[0] == hashes[1]
    assert hashes[0] != hashes[2]


def test_project_type_aware_semantics_are_preserved_for_object_series() -> None:
    series = pd.Series([1, True, 1.0, "1"], dtype="object")
    hashes = list(iter_pandas_hashes(series))

    assert len(set(hashes)) == 4
    assert pandas_hll_nunique(series) == pytest.approx(4.0, abs=0.01)


@pytest.mark.parametrize(
    ("series", "expected"),
    [
        (pd.Series([1, 2, 2, None], dtype="Int64"), 2),
        (pd.Series([True, False, True, None], dtype="boolean"), 2),
        (pd.Series(["a", "b", "a", None], dtype="string"), 2),
        (pd.Series(pd.Categorical(["a", "b", "a", None])), 2),
        (pd.Series(pd.to_datetime(["2020-01-01", "2020-01-02", None])), 2),
        (pd.Series(pd.to_timedelta(["1ns", "2ns", None])), 2),
    ],
)
def test_common_pandas_dtypes_match_exact_non_null_count(
    series: pd.Series,
    expected: int,
) -> None:
    assert series.nunique(dropna=True) == expected
    assert len(set(iter_pandas_hashes(series))) == expected
    assert pandas_hll_nunique(series) == pytest.approx(expected, abs=0.01)


def test_period_dtype_is_supported() -> None:
    series = pd.Series(
        [
            pd.Period("2020-01", freq="M"),
            pd.Period("2020-02", freq="M"),
            pd.Period("2020-01", freq="M"),
            pd.NaT,
        ]
    )

    assert len(set(iter_pandas_hashes(series))) == 2


def test_timezone_aware_timestamps_use_instant_equality() -> None:
    utc_value = pd.Timestamp("2020-01-01T00:00:00Z")
    offset_value = pd.Timestamp("2019-12-31T19:00:00-05:00")
    series = pd.Series([utc_value, offset_value], dtype="object")

    hashes = list(iter_pandas_hashes(series))
    assert hashes[0] == hashes[1]
    assert pandas_hll_nunique(series) == pytest.approx(1.0, abs=0.01)


def test_python_and_pandas_timedelta_share_logical_encoding() -> None:
    series = pd.Series([pd.Timedelta(microseconds=1), timedelta(microseconds=1)], dtype="object")

    hashes = list(iter_pandas_hashes(series))
    assert hashes[0] == hashes[1]


def test_chunk_size_does_not_change_hashes() -> None:
    series = pd.Series([1, None, 2, 3, None, 4, 5], dtype="Int64")

    one_at_a_time = list(iter_pandas_hashes(series, chunk_size=1))
    larger_chunks = list(iter_pandas_hashes(series, chunk_size=4))

    assert one_at_a_time == larger_chunks


def test_input_order_does_not_change_sketch() -> None:
    series = pd.Series(range(1_000), dtype="int64")
    reversed_series = series.iloc[::-1].reset_index(drop=True)

    first = pandas_hll_sketch(series)
    second = pandas_hll_sketch(reversed_series)

    assert first.registers == second.registers


def test_index_does_not_affect_hashing() -> None:
    first = pd.Series(["a", "b", "c"], index=[1, 2, 3])
    second = pd.Series(["a", "b", "c"], index=[100, 200, 300])

    assert list(iter_pandas_hashes(first)) == list(iter_pandas_hashes(second))


def test_seed_changes_pandas_hash_stream() -> None:
    series = pd.Series(["a", "b", "c"])

    first = list(iter_pandas_hashes(series, configuration=HashConfiguration(seed=1)))
    second = list(iter_pandas_hashes(series, configuration=HashConfiguration(seed=2)))

    assert first != second


def test_large_unique_integer_series_has_expected_hll_accuracy() -> None:
    cardinality = 100_000
    series = pd.Series(range(cardinality), dtype="int64")

    estimate = pandas_hll_nunique(series, precision=14)
    relative_error = abs(estimate - cardinality) / cardinality

    assert relative_error < 0.02


def test_repeated_large_series_tracks_distinct_values_not_rows() -> None:
    series = pd.Series(np.tile(np.arange(10_000), 10), dtype="int64")

    estimate = pandas_hll_nunique(series, precision=14)
    relative_error = abs(estimate - 10_000) / 10_000

    assert relative_error < 0.02


def test_unsupported_object_scalar_fails_explicitly() -> None:
    series = pd.Series([[1, 2], [1, 2]], dtype="object")

    with pytest.raises(TypeError, match="Unsupported cardinality scalar type"):
        list(iter_pandas_hashes(series))


def test_complex_dtype_is_supported() -> None:
    series = pd.Series([1 + 2j, 3 + 4j, 1 + 2j, np.nan + 0j], dtype="complex128")

    assert len(set(iter_pandas_hashes(series))) == 2
    assert pandas_hll_nunique(series) == pytest.approx(2.0, abs=0.01)


def test_interval_dtype_is_supported() -> None:
    first = pd.Interval(0, 1, closed="right")
    second = pd.Interval(1, 2, closed="right")
    series = pd.Series([first, second, first])

    assert len(set(iter_pandas_hashes(series))) == 2


def test_numpy_object_scalars_are_normalized() -> None:
    series = pd.Series(
        [
            np.float32(1.5),
            np.complex64(1 + 2j),
            np.str_("text"),
            np.bytes_(b"bytes"),
        ],
        dtype="object",
    )

    assert len(set(iter_pandas_hashes(series))) == 4


def test_numpy_datetime_and_timedelta_scalars_are_supported_in_object_series() -> None:
    datetime_series = pd.Series([np.datetime64("2024-01-01")], dtype="object")
    timedelta_series = pd.Series([np.timedelta64(5, "ns")], dtype="object")

    assert len(list(iter_pandas_hashes(datetime_series))) == 1
    assert len(list(iter_pandas_hashes(timedelta_series))) == 1


def test_missing_optional_pandas_dependency_has_actionable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import data_quality.cardinality.pandas as pandas_hll_module

    original_import_module = pandas_hll_module.import_module

    def fail_pandas_import(name: str) -> object:
        if name == "pandas":
            raise ImportError("missing pandas")
        return original_import_module(name)

    monkeypatch.setattr(pandas_hll_module, "import_module", fail_pandas_import)

    with pytest.raises(ImportError, match="optional 'pandas' dependency"):
        list(pandas_hll_module.iter_pandas_hashes(pd.Series([1])))
