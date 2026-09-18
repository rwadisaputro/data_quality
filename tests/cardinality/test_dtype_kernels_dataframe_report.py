from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

import data_quality.cardinality.pandas_backend as pandas_impl
from data_quality import (
    AdaptiveCardinalityConfig,
    CardinalityProfilingError,
    CardinalityRegistry,
    DataFrameCardinalityResult,
    DataFrameInput,
    UnsupportedCardinalityValueError,
    identify_backend,
    profile_cardinality,
    profile_cardinality_from_input,
    profile_cardinality_json,
)
from data_quality.cardinality import (
    CardinalityWarning,
    CardinalityWarningCode,
    CardinalityWarningSeverity,
    ColumnCardinalityOutcome,
    HashConfiguration,
    PandasCardinalityAdapter,
    PandasDtypeFamily,
    pandas_adaptive_cardinality,
    pandas_hll_sketch,
)
from data_quality.cardinality.hashing import (
    hash_configuration_for_kernel,
    hash_ndarray,
)
from data_quality.models import BackendName


def _frame(series: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({"value": series})


@pytest.mark.parametrize(
    ("series", "expected", "family"),
    [
        (pd.Series([True, False, True], dtype="boolean"), 2, PandasDtypeFamily.BOOLEAN),
        (pd.Series([1, 2, 1], dtype="Int64"), 2, PandasDtypeFamily.INTEGER),
        (pd.Series([0.0, -0.0, 1.5]), 2, PandasDtypeFamily.FLOATING),
        (pd.Series([0 + 0j, -0.0 + 0j, 1 + 2j]), 2, PandasDtypeFamily.COMPLEX),
        (pd.Series(["a", "b", "a"], dtype="string"), 2, PandasDtypeFamily.STRING),
        (pd.Series(pd.Categorical(["a", "b", "a"])), 2, PandasDtypeFamily.CATEGORICAL),
        (
            pd.Series(pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-01"])),
            2,
            PandasDtypeFamily.DATETIME,
        ),
        (
            pd.Series(
                pd.to_datetime(
                    ["2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z", "2024-01-01T00:00:00Z"]
                )
            ),
            2,
            PandasDtypeFamily.DATETIME_TZ,
        ),
        (
            pd.Series(pd.to_timedelta([1, 2, 1], unit="ns")),
            2,
            PandasDtypeFamily.TIMEDELTA,
        ),
        (
            pd.Series(
                [
                    pd.Period("2024-01", freq="M"),
                    pd.Period("2024-02", freq="M"),
                    pd.Period("2024-01", freq="M"),
                ]
            ),
            2,
            PandasDtypeFamily.PERIOD,
        ),
        (
            pd.Series(pd.arrays.IntervalArray.from_tuples([(0, 1), (1, 2), (0, 1)])),
            2,
            PandasDtypeFamily.INTERVAL,
        ),
    ],
)
def test_homogeneous_dtype_kernels_never_use_scalar_fallback(
    monkeypatch: pytest.MonkeyPatch,
    series: pd.Series,
    expected: int,
    family: PandasDtypeFamily,
) -> None:
    def fail_scalar(*args: object, **kwargs: object) -> bytes:
        raise AssertionError("homogeneous dtype kernel must not use scalar canonicalization")

    monkeypatch.setattr(pandas_impl, "_canonicalize_pandas_scalar", fail_scalar)
    result = pandas_adaptive_cardinality(_frame(series), "value")

    assert result.distinct_count == expected
    assert result.lineage["dtype_family"] == family.value
    assert result.lineage["hash_kernel"]["scalar_fallback"] is False
    assert result.lineage["scalar_fallback_used"] is False
    assert result.warnings == ()


def test_signed_and_unsigned_integer_kernels_are_distinct_without_int64_coercion() -> None:
    signed = _frame(pd.Series([1, 2, 3], dtype="int64"))
    unsigned = _frame(
        pd.Series([2**63 + 1, 2**63 + 2, 2**63 + 1], dtype="UInt64")
    )

    signed_result = pandas_adaptive_cardinality(signed, "value")
    unsigned_result = pandas_adaptive_cardinality(unsigned, "value")

    assert signed_result.lineage["column"]["integer_signedness"] == "signed"
    assert unsigned_result.lineage["column"]["integer_signedness"] == "unsigned"
    assert signed_result.lineage["hash_kernel"]["kernel_id"] == "pandas-signed-integer-v2"
    assert unsigned_result.lineage["hash_kernel"]["kernel_id"] == "pandas-unsigned-integer-v2"
    assert unsigned_result.distinct_count == 2

    signed_sketch = pandas_hll_sketch(signed, "value")
    unsigned_small = _frame(pd.Series([1, 2, 3], dtype="uint64"))
    unsigned_sketch = pandas_hll_sketch(unsigned_small, "value")
    assert signed_sketch.hash_configuration != unsigned_sketch.hash_configuration
    with pytest.raises(ValueError, match="different hash configurations"):
        signed_sketch.merge(unsigned_sketch)


def test_temporal_kernels_normalize_supported_units_to_nanoseconds() -> None:
    expected_datetime = np.array(
        [1577836800000000000, 1577923200000000000],
        dtype=np.int64,
    )
    for unit in ("s", "ms", "us", "ns"):
        series = pd.Series(
            np.array(["2020-01-01", "2020-01-02"], dtype=f"datetime64[{unit}]")
        )
        identity = pandas_impl.identify_pandas_dtype(series.dtype)
        arrays = list(
            pandas_impl._iter_pandas_canonical_arrays(
                series,
                chunk_size=8,
                dtype_identity=identity,
            )
        )
        assert len(arrays) == 1
        tokens, _, null_count = arrays[0]
        assert null_count == 0
        assert np.array_equal(tokens, expected_datetime)

    for unit, multiplier in (("s", 1_000_000_000), ("ms", 1_000_000), ("us", 1_000), ("ns", 1)):
        series = pd.Series(np.array([1, 2], dtype=f"timedelta64[{unit}]"))
        identity = pandas_impl.identify_pandas_dtype(series.dtype)
        tokens, _, _ = next(
            pandas_impl._iter_pandas_canonical_arrays(
                series,
                chunk_size=8,
                dtype_identity=identity,
            )
        )
        assert tokens.tolist() == [multiplier, 2 * multiplier]


def test_object_numpy_unsigned_scalar_remains_distinct_from_signed_integer() -> None:
    dataframe = _frame(pd.Series([np.int64(1), np.uint64(1)], dtype=object))
    result = pandas_adaptive_cardinality(dataframe, "value")
    assert result.distinct_count == 2


def test_pandas_null_count_reuses_chunk_masks_without_second_full_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataframe = _frame(pd.Series([1, None, 2, None, 3], dtype="Int64"))
    original_isna = pd.Series.isna
    lengths: list[int] = []

    def tracking_isna(series: pd.Series) -> pd.Series:
        lengths.append(len(series))
        return original_isna(series)

    monkeypatch.setattr(pd.Series, "isna", tracking_isna)
    result = pandas_adaptive_cardinality(
        dataframe,
        "value",
        chunk_size=2,
    )

    assert result.null_count == 2
    assert lengths == [2, 2, 1]


def test_object_dtype_uses_structured_scalar_fallback_warning() -> None:
    dataframe = _frame(pd.Series([1, True, 1.0, "1"], dtype="object"))
    result = pandas_adaptive_cardinality(dataframe, "value")

    assert result.distinct_count == 4
    assert result.lineage["scalar_fallback_used"] is True
    assert len(result.warnings) == 1
    warning = result.warnings[0]
    assert warning.code is CardinalityWarningCode.SCALAR_FALLBACK
    assert warning.severity is CardinalityWarningSeverity.PERFORMANCE
    assert warning.context["column"] == "value"
    assert result.as_dict()["warnings"][0]["code"] == "scalar_canonicalization_fallback"


def test_categorical_private_tokenization_builds_lookup_without_repr() -> None:
    categorical = pd.Categorical(
        ["a", "b", "a"],
        categories=["b", "a"],
        ordered=True,
    )
    dtype_identity = pandas_impl.identify_pandas_dtype(categorical.dtype)

    tokens = pandas_impl._canonicalize_pandas_ndarray(
        categorical,
        dtype_identity=dtype_identity,
        pandas_module=pd,
        numpy_module=np,
        row_positions=np.array([0, 1, 2], dtype=np.int64),
        column="value",
        native_dtype=categorical.dtype,
    )

    assert tokens.tolist() == [b"sa", b"sb", b"sa"]


def test_unsupported_categorical_value_reports_first_used_row() -> None:
    class UnsupportedCategory:
        pass

    value = UnsupportedCategory()
    dataframe = _frame(
        pd.Series(
            pd.Categorical(
                [value, value],
                categories=[value],
            )
        )
    )

    with pytest.raises(UnsupportedCardinalityValueError) as exc_info:
        pandas_adaptive_cardinality(dataframe, "value")

    payload = exc_info.value.as_dict()
    assert payload["context"]["row_position"] == 0
    assert payload["context"]["scalar_type"].endswith("UnsupportedCategory")


def test_unsupported_object_scalar_has_structured_failure_and_row_position() -> None:
    dataframe = pd.DataFrame({"value": [1, [2, 3], 4]}, dtype="object")

    with pytest.raises(UnsupportedCardinalityValueError) as exc_info:
        pandas_adaptive_cardinality(dataframe, "value")

    payload = exc_info.value.as_dict()
    assert payload["code"] == "unsupported_cardinality_scalar"
    assert payload["severity"] == "error"
    assert payload["context"]["row_position"] == 1
    assert payload["context"]["scalar_type"] == "builtins.list"


def test_dataframe_only_entrypoint_profiles_every_column_and_isolates_unsupported() -> None:
    dataframe = pd.DataFrame(
        {
            "id": pd.Series([1, 2, 2], dtype="Int64"),
            "ratio": [0.0, -0.0, 1.0],
            "mixed": pd.Series([1, "1", True], dtype="object"),
            "unsupported": pd.Series([[1], [2], [1]], dtype="object"),
        }
    )

    result = profile_cardinality(dataframe)

    assert isinstance(result, DataFrameCardinalityResult)
    assert result.total_row_count == 3
    assert result.column_count == 4
    assert result.successful_column_count == 3
    assert result.unsupported_column_count == 1
    assert [outcome.status for outcome in result.columns] == ["ok", "ok", "ok", "unsupported"]
    assert result.columns[0].result is not None
    assert result.columns[0].result.distinct_count == 2
    assert result.columns[2].result is not None
    assert result.columns[2].result.distinct_count == 3
    assert result.columns[3].error is not None
    assert len(result.warnings) == 2


def test_dataframe_only_json_contains_per_column_results_and_structured_error() -> None:
    dataframe = pd.DataFrame(
        {
            "id": pd.Series([1, 2, 2], dtype="Int64"),
            "bad": pd.Series([[1], [2], [1]], dtype="object"),
        }
    )
    payload = json.loads(profile_cardinality_json(dataframe, indent=None))

    assert payload["schema_version"] == "dataframe-cardinality-v1"
    assert payload["backend"]["name"] == "pandas"
    assert payload["successful_column_count"] == 1
    assert payload["unsupported_column_count"] == 1
    assert payload["columns"][0]["result"]["output"]["distinct_count"] == 2
    assert payload["columns"][1]["status"] == "unsupported"
    assert payload["columns"][1]["error"]["code"] == "unsupported_cardinality_scalar"


def test_preidentified_dataframe_input_can_profile_all_columns() -> None:
    dataframe = pd.DataFrame({"a": [1, 1], "b": [2, 3]})
    dataframe_input = DataFrameInput(dataframe=dataframe, backend=identify_backend(dataframe))

    result = profile_cardinality_from_input(dataframe_input)

    assert isinstance(result, DataFrameCardinalityResult)
    assert [outcome.result.distinct_count for outcome in result.columns if outcome.result] == [1, 2]


def test_dataframe_all_column_path_supports_duplicate_labels_by_position() -> None:
    dataframe = pd.DataFrame([[1, 10], [1, 20]], columns=["id", "id"])
    result = profile_cardinality(dataframe)

    assert isinstance(result, DataFrameCardinalityResult)
    assert [outcome.position for outcome in result.columns] == [0, 1]
    assert [outcome.result.distinct_count for outcome in result.columns if outcome.result] == [1, 2]


def test_none_is_still_a_valid_explicit_column_label() -> None:
    dataframe = pd.DataFrame({None: [1, 1, 2], "x": [3, 4, 5]})
    result = profile_cardinality(dataframe, None)
    assert result.distinct_count == 2


def test_dtype_kernel_is_bound_into_sketch_merge_compatibility() -> None:
    integer = pandas_hll_sketch(pd.DataFrame({"x": [1, 2, 3]}), "x")
    floating = pandas_hll_sketch(pd.DataFrame({"x": [1.0, 2.0, 3.0]}), "x")

    assert integer.hash_configuration != floating.hash_configuration
    with pytest.raises(ValueError, match="hash configurations"):
        integer.merge(floating)


def test_numeric_hash_seed_changes_vectorized_hashes() -> None:
    values = np.arange(10, dtype=np.int64)
    first = hash_ndarray(values, kernel_id="integer-test", configuration=HashConfiguration(seed=1))
    second = hash_ndarray(values, kernel_id="integer-test", configuration=HashConfiguration(seed=2))
    assert first.dtype == np.uint64
    assert not np.array_equal(first, second)


def test_hash_ndarray_validation_and_empty_array() -> None:
    with pytest.raises(TypeError, match="HashConfiguration"):
        hash_ndarray(np.array([1]), kernel_id="x", configuration=object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="kernel_id"):
        hash_ndarray(np.array([1]), kernel_id="")
    with pytest.raises(ValueError, match="one-dimensional"):
        hash_ndarray(np.array([[1]]), kernel_id="x")

    empty = hash_ndarray(np.array([], dtype=np.int64), kernel_id="x")
    assert empty.dtype == np.uint64
    assert empty.size == 0


def test_hash_configuration_for_kernel_validation_and_metadata() -> None:
    configured = hash_configuration_for_kernel(
        HashConfiguration(seed=9),
        kernel_id="float-v1",
        runtime_version="2.2-test",
    )
    assert configured.seed == 9
    assert configured.canonicalisation_version == "pandas-dtype-kernels-v3"
    assert configured.runtime_id == "pandas"
    assert configured.runtime_version == "2.2-test"
    assert configured.kernel_id == "float-v1"
    assert hash_configuration_for_kernel(
        configured,
        kernel_id="float-v1",
        runtime_version="2.2-test",
    ) is configured

    with pytest.raises(TypeError, match="HashConfiguration"):
        hash_configuration_for_kernel(object(), kernel_id="x")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="kernel_id"):
        hash_configuration_for_kernel(HashConfiguration(), kernel_id="")
    with pytest.raises(ValueError, match="supported base hash contract"):
        hash_configuration_for_kernel(
            HashConfiguration(algorithm_id="other", engine_id="other"),
            kernel_id="x",
        )


def test_report_models_expose_attributes_and_json_safe_non_string_column() -> None:
    warning = CardinalityWarning(
        code=CardinalityWarningCode.UNSUPPORTED_SCALAR,
        message="unsupported",
        severity=CardinalityWarningSeverity.ERROR,
    )
    outcome = ColumnCardinalityOutcome(
        column=("multi", "label"),
        position=0,
        result=None,
        error=warning,
    )
    empty_outcome = ColumnCardinalityOutcome(column="empty", position=1, result=None)
    report = DataFrameCardinalityResult(
        backend={"name": "pandas"},
        total_row_count=0,
        column_count=2,
        columns=(outcome, empty_outcome),
    )

    assert outcome.status == "unsupported"
    assert outcome.warnings == (warning,)
    assert empty_outcome.warnings == ()
    assert report.successful_column_count == 0
    assert report.unsupported_column_count == 2
    assert report.warnings == (warning,)
    payload = json.loads(report.to_json(indent=None))
    assert payload["columns"][0]["column"] == ["multi", "label"]


@dataclass
class _ExplodingAllAdapter:
    backend_name: BackendName = BackendName.PANDAS

    def profile(self, *args: object, **kwargs: object) -> object:
        raise RuntimeError("single")

    def profile_all(self, *args: object, **kwargs: object) -> object:
        raise RuntimeError("all")


def test_registry_wraps_unexpected_all_column_adapter_failure() -> None:
    dataframe = pd.DataFrame({"id": [1, 2]})
    dataframe_input = DataFrameInput(dataframe=dataframe, backend=identify_backend(dataframe))
    registry = CardinalityRegistry((_ExplodingAllAdapter(),))

    with pytest.raises(CardinalityProfilingError) as exc_info:
        registry.profile_all(dataframe_input, config=None, batch_size=None)

    assert exc_info.value.__cause__ is not None


def test_pandas_adapter_profile_all_directly() -> None:
    dataframe = pd.DataFrame({"a": [1, 1], "b": [1, 2]})
    backend = identify_backend(dataframe)
    result = PandasCardinalityAdapter().profile_all(
        dataframe,
        backend,
        config=AdaptiveCardinalityConfig(exact_unique_threshold=10),
        batch_size=1,
    )
    assert result.successful_column_count == 2


def test_direct_pandas_dataframe_profile_discovers_backend_metadata() -> None:
    result = pandas_impl.pandas_dataframe_cardinality(pd.DataFrame({"id": [1, 1, 2]}))
    assert result.backend["name"] == "pandas"


def test_temporal_private_kernels_cover_plain_ndarray_fallback_shapes() -> None:
    period_identity = pandas_impl.PandasDtypeIdentity(
        family=PandasDtypeFamily.PERIOD,
        numpy_kind="O",
        is_extension_dtype=True,
        canonicalisation_strategy="test",
    )
    period_values = np.array(
        [pd.Period("2024-01", freq="M"), pd.Period("2024-02", freq="M")],
        dtype=object,
    )
    period_tokens = pandas_impl._canonicalize_pandas_ndarray(
        period_values,
        dtype_identity=period_identity,
        pandas_module=pd,
        numpy_module=np,
    )
    assert period_tokens.tolist() == [
        pd.Timestamp("2024-01-01").value,
        pd.Timestamp("2024-02-01").value,
    ]

    datetime_identity = pandas_impl.PandasDtypeIdentity(
        family=PandasDtypeFamily.DATETIME,
        numpy_kind="O",
        is_extension_dtype=False,
        canonicalisation_strategy="test",
    )
    datetime_tokens = pandas_impl._canonicalize_pandas_ndarray(
        np.array([pd.Timestamp("2024-01-01")], dtype=object),
        dtype_identity=datetime_identity,
        pandas_module=pd,
        numpy_module=np,
    )
    assert datetime_tokens.dtype == np.int64


def test_object_scalar_fallback_covers_period_interval_and_numpy_integer() -> None:
    period = pandas_impl._canonicalize_pandas_scalar(
        pd.Period("2024-01", freq="M"),
        pandas_module=pd,
        numpy_module=np,
    )
    interval = pandas_impl._canonicalize_pandas_scalar(
        pd.Interval(0, 1, closed="right"),
        pandas_module=pd,
        numpy_module=np,
    )
    integer = pandas_impl._canonicalize_pandas_scalar(
        np.int64(7),
        pandas_module=pd,
        numpy_module=np,
    )
    assert period.startswith(b"p")
    assert interval.startswith(b"v")
    assert integer == b"i7"


def test_pandas_column_position_validation_rejects_out_of_range() -> None:
    with pytest.raises(IndexError, match="out of range"):
        pandas_impl._resolve_pandas_dataframe_column_position(pd.DataFrame({"x": [1]}), 2)


def test_structured_unsupported_error_json_safes_nonprimitive_column_label() -> None:
    error = UnsupportedCardinalityValueError(
        backend="pandas",
        column=("group", "value"),
        native_dtype="object",
        row_position=None,
        value=[1],
    )
    assert error.as_dict()["context"]["column"] == ["group", "value"]


@dataclass
class _ValueErrorAllAdapter:
    backend_name: BackendName = BackendName.PANDAS

    def profile(self, *args: object, **kwargs: object) -> object:
        raise ValueError("bad")

    def profile_all(self, *args: object, **kwargs: object) -> object:
        raise ValueError("bad")


def test_registry_all_column_validation_errors_are_not_wrapped() -> None:
    dataframe = pd.DataFrame({"id": [1]})
    dataframe_input = DataFrameInput(dataframe=dataframe, backend=identify_backend(dataframe))
    registry = CardinalityRegistry((_ValueErrorAllAdapter(),))
    with pytest.raises(ValueError, match="bad"):
        registry.profile_all(dataframe_input, config=None, batch_size=None)


def test_adaptive_exact_memory_measure_handles_empty_native_array() -> None:
    from data_quality.cardinality import AdaptiveCardinalityHandler

    handler = AdaptiveCardinalityHandler()
    assert handler._measure_exact_state_bytes(None) == 0
    assert handler._measure_exact_state_bytes(np.array([], dtype=np.int64)) == 0
    assert np.array_equal(
        handler._merge_sorted_unique(
            np.array([1, 2], dtype=np.int64),
            np.array([], dtype=np.int64),
        ),
        np.array([1, 2], dtype=np.int64),
    )


def test_json_safe_metadata_never_calls_custom_repr() -> None:
    from data_quality.cardinality.diagnostics import json_safe_metadata_value

    class ExplosiveRepr:
        def __repr__(self) -> str:
            raise AssertionError("repr must not be called")

    value = ExplosiveRepr()
    assert json_safe_metadata_value([value]) == [
        {
            "type": (
                "test_dtype_kernels_dataframe_report."
                "test_json_safe_metadata_never_calls_custom_repr.<locals>.ExplosiveRepr"
            ),
            "serialization": "omitted",
        }
    ]
    assert json_safe_metadata_value({"x": b"abc", object(): "ignored"}) == {
        "x": {"type": "bytes", "length": 3}
    }


def test_temporal_private_kernel_timedelta_plain_array_fallback_is_nanoseconds() -> None:
    identity = pandas_impl.PandasDtypeIdentity(
        family=PandasDtypeFamily.TIMEDELTA,
        numpy_kind="O",
        is_extension_dtype=False,
        canonicalisation_strategy="test",
    )
    tokens = pandas_impl._canonicalize_pandas_ndarray(
        np.array([pd.Timedelta(seconds=1), pd.Timedelta(seconds=2)], dtype=object),
        dtype_identity=identity,
        pandas_module=pd,
        numpy_module=np,
    )
    assert tokens.tolist() == [1_000_000_000, 2_000_000_000]
