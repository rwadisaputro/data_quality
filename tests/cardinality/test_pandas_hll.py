from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from data_quality.cardinality import HashConfiguration, hash_scalar
from data_quality.cardinality.pandas_backend import (
    DEFAULT_PANDAS_CHUNK_SIZE,
    iter_pandas_hash_arrays,
    iter_pandas_hashes,
    pandas_hll_nunique,
    pandas_hll_sketch,
)


def _frame(series: pd.Series, name: str = "value") -> pd.DataFrame:
    return pd.DataFrame({name: series})


def test_default_chunk_size_is_bounded() -> None:
    assert DEFAULT_PANDAS_CHUNK_SIZE == 65_536


def test_pandas_hashing_requires_dataframe_boundary() -> None:
    with pytest.raises(TypeError, match="pandas.DataFrame"):
        list(iter_pandas_hashes(pd.Series([1]), "value"))


def test_pandas_hashing_requires_existing_unique_column() -> None:
    dataframe = pd.DataFrame([[1, 2]], columns=["id", "id"])

    with pytest.raises(ValueError, match="not unique"):
        list(iter_pandas_hashes(dataframe, "id"))
    with pytest.raises(KeyError, match="no column"):
        list(iter_pandas_hashes(pd.DataFrame({"id": [1]}), "missing"))


@pytest.mark.parametrize("chunk_size", [True, 1.5, "10"])
def test_invalid_chunk_size_type_is_rejected(chunk_size: object) -> None:
    dataframe = pd.DataFrame({"value": [1]})
    with pytest.raises(TypeError):
        list(
            iter_pandas_hashes(
                dataframe,
                "value",
                chunk_size=chunk_size,  # type: ignore[arg-type]
            )
        )


@pytest.mark.parametrize("chunk_size", [0, -1])
def test_non_positive_chunk_size_is_rejected(chunk_size: int) -> None:
    dataframe = pd.DataFrame({"value": [1]})
    with pytest.raises(ValueError):
        list(iter_pandas_hashes(dataframe, "value", chunk_size=chunk_size))


def test_numpy_integer_chunk_size_is_accepted() -> None:
    dataframe = pd.DataFrame({"value": [1, 2]})
    hashes = list(iter_pandas_hashes(dataframe, "value", chunk_size=np.int64(1)))
    assert len(hashes) == 2


def test_empty_dataframe_column_estimates_zero() -> None:
    dataframe = _frame(pd.Series([], dtype="object"))
    assert pandas_hll_nunique(dataframe, "value") == 0


def test_all_null_values_are_excluded() -> None:
    dataframe = _frame(pd.Series([None, np.nan, pd.NA, pd.NaT], dtype="object"))
    assert list(iter_pandas_hashes(dataframe, "value")) == []
    assert pandas_hll_nunique(dataframe, "value") == 0


def test_duplicate_values_hash_identically() -> None:
    dataframe = _frame(pd.Series(["same", "same", "different"]))
    hashes = list(iter_pandas_hashes(dataframe, "value"))
    assert hashes[0] == hashes[1]
    assert hashes[0] != hashes[2]


def test_project_type_aware_semantics_are_preserved_for_object_dtype() -> None:
    dataframe = _frame(pd.Series([1, True, 1.0, "1"], dtype="object"))
    hashes = list(iter_pandas_hashes(dataframe, "value"))
    assert len(set(hashes)) == 4
    assert pandas_hll_nunique(dataframe, "value") == 4


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
    dataframe = _frame(series)
    assert dataframe["value"].nunique(dropna=True) == expected
    assert len(set(iter_pandas_hashes(dataframe, "value"))) == expected
    assert pandas_hll_nunique(dataframe, "value") == expected


def test_categorical_hashes_are_independent_of_category_order_and_ordered_flag() -> None:
    left = _frame(
        pd.Series(
            pd.Categorical(
                ["a", "b", "a"],
                categories=["a", "b"],
                ordered=False,
            )
        )
    )
    right = _frame(
        pd.Series(
            pd.Categorical(
                ["a", "b", "a"],
                categories=["b", "a"],
                ordered=True,
            )
        )
    )

    left_hashes = np.concatenate(list(iter_pandas_hash_arrays(left, "value", chunk_size=2)))
    right_hashes = np.concatenate(
        list(iter_pandas_hash_arrays(right, "value", chunk_size=2))
    )

    assert np.array_equal(left_hashes, right_hashes)


def test_categorical_sketches_with_different_dictionaries_can_merge() -> None:
    left = _frame(
        pd.Series(pd.Categorical(["a", "b"], categories=["a", "b"]))
    )
    right = _frame(
        pd.Series(pd.Categorical(["b", "c"], categories=["b", "c"]))
    )

    left_sketch = pandas_hll_sketch(left, "value")
    right_sketch = pandas_hll_sketch(right, "value")

    left_sketch.merge(right_sketch)

    assert round(left_sketch.estimate()) == 3


def test_period_dtype_is_supported() -> None:
    series = pd.Series(
        [
            pd.Period("2020-01", freq="M"),
            pd.Period("2020-02", freq="M"),
            pd.Period("2020-01", freq="M"),
            pd.NaT,
        ]
    )
    dataframe = _frame(series)
    assert len(set(iter_pandas_hashes(dataframe, "value"))) == 2


def test_timezone_aware_timestamps_use_instant_equality() -> None:
    series = pd.Series(
        [pd.Timestamp("2020-01-01T00:00:00Z"), pd.Timestamp("2019-12-31T19:00:00-05:00")],
        dtype="object",
    )
    dataframe = _frame(series)
    hashes = list(iter_pandas_hashes(dataframe, "value"))
    assert hashes[0] == hashes[1]
    assert pandas_hll_nunique(dataframe, "value") == 1


def test_python_and_pandas_timedelta_share_logical_encoding() -> None:
    series = pd.Series([pd.Timedelta(microseconds=1), timedelta(microseconds=1)], dtype="object")
    dataframe = _frame(series)
    hashes = list(iter_pandas_hashes(dataframe, "value"))
    assert hashes[0] == hashes[1]


def test_chunk_size_does_not_change_hashes() -> None:
    dataframe = _frame(pd.Series([1, None, 2, 3, None, 4, 5], dtype="Int64"))
    one_at_a_time = list(iter_pandas_hashes(dataframe, "value", chunk_size=1))
    larger_chunks = list(iter_pandas_hashes(dataframe, "value", chunk_size=4))
    assert one_at_a_time == larger_chunks


def test_input_order_does_not_change_sketch() -> None:
    series = pd.Series(range(1_000), dtype="int64")
    first = pandas_hll_sketch(_frame(series), "value")
    second = pandas_hll_sketch(_frame(series.iloc[::-1].reset_index(drop=True)), "value")
    assert first.registers == second.registers


def test_dataframe_index_does_not_affect_hashing() -> None:
    first = pd.DataFrame({"value": ["a", "b", "c"]}, index=[1, 2, 3])
    second = pd.DataFrame({"value": ["a", "b", "c"]}, index=[100, 200, 300])
    assert list(iter_pandas_hashes(first, "value")) == list(iter_pandas_hashes(second, "value"))


def test_seed_changes_pandas_hash_stream() -> None:
    dataframe = pd.DataFrame({"value": ["a", "b", "c"]})
    first = list(
        iter_pandas_hashes(dataframe, "value", configuration=HashConfiguration(seed=1))
    )
    second = list(
        iter_pandas_hashes(dataframe, "value", configuration=HashConfiguration(seed=2))
    )
    assert first != second


def test_large_unique_integer_column_has_expected_hll_accuracy() -> None:
    cardinality = 100_000
    dataframe = pd.DataFrame({"value": np.arange(cardinality, dtype=np.int64)})
    estimate = pandas_hll_nunique(dataframe, "value", precision=14)
    assert isinstance(estimate, int)
    assert abs(estimate - cardinality) / cardinality < 0.02


def test_repeated_large_column_tracks_distinct_values_not_rows() -> None:
    dataframe = pd.DataFrame({"value": np.tile(np.arange(10_000), 10)})
    estimate = pandas_hll_nunique(dataframe, "value", precision=14)
    assert abs(estimate - 10_000) / 10_000 < 0.02


def test_unsupported_object_scalar_fails_explicitly() -> None:
    dataframe = _frame(pd.Series([[1, 2], [1, 2]], dtype="object"))
    with pytest.raises(TypeError, match="Unsupported cardinality scalar type"):
        list(iter_pandas_hashes(dataframe, "value"))


def test_complex_interval_and_numpy_object_scalars_are_supported() -> None:
    complex_frame = _frame(pd.Series([1 + 2j, 3 + 4j, 1 + 2j, np.nan + 0j]))
    assert len(set(iter_pandas_hashes(complex_frame, "value"))) == 2

    first = pd.Interval(0, 1, closed="right")
    second = pd.Interval(1, 2, closed="right")
    interval_frame = _frame(pd.Series([first, second, first]))
    assert len(set(iter_pandas_hashes(interval_frame, "value"))) == 2

    object_frame = _frame(
        pd.Series(
            [np.float32(1.5), np.complex64(1 + 2j), np.str_("text"), np.bytes_(b"bytes")],
            dtype="object",
        )
    )
    assert len(set(iter_pandas_hashes(object_frame, "value"))) == 4


def test_numpy_datetime_and_timedelta_scalars_are_supported_in_object_dtype() -> None:
    datetime_frame = _frame(pd.Series([np.datetime64("2024-01-01")], dtype="object"))
    timedelta_frame = _frame(pd.Series([np.timedelta64(5, "ns")], dtype="object"))
    assert len(list(iter_pandas_hashes(datetime_frame, "value"))) == 1
    assert len(list(iter_pandas_hashes(timedelta_frame, "value"))) == 1


def test_missing_optional_pandas_dependency_has_actionable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import data_quality.cardinality.pandas_backend as pandas_hll_module

    original_import_module = pandas_hll_module.import_module

    def fail_pandas_import(name: str) -> object:
        if name == "pandas":
            raise ImportError("missing pandas")
        return original_import_module(name)

    monkeypatch.setattr(pandas_hll_module, "import_module", fail_pandas_import)
    with pytest.raises(ImportError, match="optional 'pandas' dependency"):
        list(pandas_hll_module.iter_pandas_hashes(pd.DataFrame({"value": [1]}), "value"))


@pytest.mark.parametrize(
    "series",
    [
        pd.Series([1, -2, 3], dtype="Int64"),
        pd.Series([True, False, True], dtype="boolean"),
        pd.Series(["alpha", "β", "gamma"], dtype="string"),
    ],
)
def test_dtype_specialized_hashing_is_deterministic_and_distinct(
    series: pd.Series,
) -> None:
    dataframe = _frame(series)
    first = list(iter_pandas_hashes(dataframe, "value"))
    second = list(iter_pandas_hashes(dataframe, "value"))
    assert first == second
    assert len(set(first)) == series.nunique(dropna=True)

