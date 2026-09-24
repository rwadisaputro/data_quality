from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from data_quality.cardinality import (
    AdaptiveCardinalityConfig,
    CardinalityMode,
    HashConfiguration,
    PromotionReason,
    iter_pandas_hash_arrays,
    pandas_adaptive_cardinality,
    pandas_adaptive_cardinality_json,
    pandas_hll_sketch,
)


def _frame(series: pd.Series, name: str = "value") -> pd.DataFrame:
    return pd.DataFrame({name: series})


def test_low_cardinality_stays_exact() -> None:
    dataframe = _frame(pd.Series(np.tile(np.arange(50), 20), dtype="int64"))
    config = AdaptiveCardinalityConfig(
        exact_unique_threshold=100,
        exact_memory_budget_bytes=10_000_000,
        threshold_check_interval=16,
    )
    result = pandas_adaptive_cardinality(dataframe, "value", config=config, chunk_size=128)
    assert result.final_mode is CardinalityMode.EXACT
    assert result.is_exact
    assert result.distinct_count == 50
    assert not result.promotion.occurred
    assert result.retained_exact_unique_count == 50


def test_unique_threshold_promotes_once_to_hll() -> None:
    dataframe = _frame(pd.Series(np.arange(150), dtype="int64"))
    config = AdaptiveCardinalityConfig(
        exact_unique_threshold=100,
        exact_memory_budget_bytes=10_000_000,
        threshold_check_interval=10,
    )
    result = pandas_adaptive_cardinality(dataframe, "value", config=config, chunk_size=150)
    assert result.final_mode is CardinalityMode.HLL
    assert not result.is_exact
    assert result.promotion.occurred
    assert result.promotion.reason is PromotionReason.UNIQUE_THRESHOLD
    assert result.promotion.unique_count == 101
    assert result.promotion.row_position == 100
    assert result.promotion.non_null_position == 101
    assert result.retained_exact_unique_count == 0
    assert isinstance(result.distinct_count, int)
    assert abs(result.distinct_count - 150) / 150 < 0.05


def test_duplicate_heavy_input_does_not_promote_on_row_count() -> None:
    dataframe = _frame(pd.Series(np.tile(np.arange(10), 10_000), dtype="int64"))
    config = AdaptiveCardinalityConfig(
        exact_unique_threshold=20,
        exact_memory_budget_bytes=10_000_000,
        threshold_check_interval=128,
    )
    result = pandas_adaptive_cardinality(dataframe, "value", config=config, chunk_size=1_024)
    assert result.is_exact
    assert result.distinct_count == 10
    assert not result.promotion.occurred


def test_duplicate_only_batch_skips_exact_state_merge_and_memory_remeasure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from data_quality.cardinality import AdaptiveCardinalityHandler

    handler = AdaptiveCardinalityHandler(
        AdaptiveCardinalityConfig(
            exact_unique_threshold=100,
            exact_memory_budget_bytes=10_000_000,
            threshold_check_interval=32,
        ),
        array_hasher=lambda values: np.asarray(values, dtype=np.uint64),
    )
    handler.add_canonical_array(np.array([1, 2, 3], dtype=np.int64))

    def fail_merge(*args: object, **kwargs: object) -> object:
        raise AssertionError("duplicate-only batches must not rebuild exact state")

    def fail_measure(*args: object, **kwargs: object) -> int:
        raise AssertionError("duplicate-only batches must not remeasure exact memory")

    def fail_unique(*args: object, **kwargs: object) -> object:
        raise AssertionError("duplicate-only batches must not sort/unique the chunk")

    monkeypatch.setattr(
        AdaptiveCardinalityHandler,
        "_merge_sorted_unique",
        staticmethod(fail_merge),
    )
    monkeypatch.setattr(
        AdaptiveCardinalityHandler,
        "_measure_exact_state_raw_bytes",
        fail_measure,
    )
    monkeypatch.setattr(np, "unique", fail_unique)

    handler.add_canonical_array(
        np.array([3, 2, 1, 1, 2, 3] * 10, dtype=np.int64)
    )

    assert handler.mode is CardinalityMode.EXACT
    assert handler.distinct_count() == 3


def test_triggering_batch_uniques_once_and_does_not_rescan_prefixes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from data_quality.cardinality import AdaptiveCardinalityHandler

    handler = AdaptiveCardinalityHandler(
        AdaptiveCardinalityConfig(
            exact_unique_threshold=5,
            exact_memory_budget_bytes=10_000_000,
            threshold_check_interval=64,
        ),
        array_hasher=lambda values: np.asarray(values, dtype=np.uint64),
    )

    real_unique = np.unique
    unique_calls = 0

    def counting_unique(*args: object, **kwargs: object) -> object:
        nonlocal unique_calls
        unique_calls += 1
        return real_unique(*args, **kwargs)

    monkeypatch.setattr(np, "unique", counting_unique)
    handler.add_canonical_array(np.array([0, 1, 2, 3, 4, 5, 6], dtype=np.int64))

    assert handler.mode is CardinalityMode.HLL
    assert handler.promotion.row_position == 5
    assert handler.promotion.unique_count == 6
    assert unique_calls == 1


def test_memory_threshold_can_promote_before_unique_threshold() -> None:
    dataframe = _frame(pd.Series(["x" * 100 + str(i) for i in range(20)], dtype="string"))
    config = AdaptiveCardinalityConfig(
        exact_unique_threshold=10_000,
        exact_memory_budget_bytes=1_000,
        memory_safety_factor=1.0,
        threshold_check_interval=1,
    )
    result = pandas_adaptive_cardinality(dataframe, "value", config=config, chunk_size=20)
    assert result.final_mode is CardinalityMode.HLL
    assert result.promotion.reason is PromotionReason.MEMORY_THRESHOLD
    assert result.promotion.exact_state_bytes is not None
    assert result.promotion.exact_state_bytes > config.exact_memory_budget_bytes


def test_null_metrics_and_original_row_position_are_preserved() -> None:
    dataframe = _frame(pd.Series([None, 1, None, 2, 3, 4, 5], dtype="Int64"))
    config = AdaptiveCardinalityConfig(
        exact_unique_threshold=2,
        exact_memory_budget_bytes=10_000_000,
        threshold_check_interval=1,
    )
    result = pandas_adaptive_cardinality(dataframe, "value", config=config, chunk_size=7)
    assert result.total_row_count == 7
    assert result.non_null_count == 5
    assert result.null_count == 2
    assert result.promotion.row_position == 4
    assert result.promotion.non_null_position == 3


def test_json_contains_dataframe_output_parameters_and_lineage() -> None:
    dataframe = pd.DataFrame(
        {
            "customer_id": pd.Series([1, 2, 2, None], dtype="Int64"),
            "payload": ["a", "b", "c", "d"],
        }
    )
    config = AdaptiveCardinalityConfig(
        exact_unique_threshold=50,
        exact_memory_budget_bytes=12_345,
        memory_safety_factor=1.5,
        hll_precision=12,
        threshold_check_interval=7,
        hash_configuration=HashConfiguration(seed=99),
    )
    payload = json.loads(
        pandas_adaptive_cardinality_json(
            dataframe,
            "customer_id",
            config=config,
            chunk_size=3,
        )
    )
    assert payload["schema_version"] == "adaptive-cardinality-v6"
    assert payload["output"]["distinct_count"] == 2
    assert payload["output"]["is_exact"] is True
    assert payload["row_metrics"] == {
        "total_row_count": 4,
        "non_null_count": 3,
        "null_count": 1,
    }
    assert payload["parameters"]["exact_unique_threshold"] == 50
    assert payload["parameters"]["exact_memory_budget_bytes"] == 12_345
    assert payload["parameters"]["memory_safety_factor"] == 1.5
    assert payload["parameters"]["threshold_check_interval"] == 7
    assert payload["parameters"]["hll_precision"] == 12
    assert payload["parameters"]["hash_configuration"]["seed"] == 99
    assert payload["promotion"]["occurred"] is False
    assert payload["lineage"]["backend"]["name"] == "pandas"
    assert payload["lineage"]["column"]["name"] == "customer_id"
    assert payload["lineage"]["column"]["native_dtype"] == "Int64"
    assert payload["lineage"]["dataframe_column_count"] == 2
    assert payload["parameters"]["execution"]["pandas_chunk_size"] == 3
    assert payload["lineage"]["hash_execution"] == "numpy_ndarray_bulk"


def test_hll_json_reports_sketch_metadata_after_promotion() -> None:
    dataframe = pd.DataFrame({"id": np.arange(1_000, dtype=np.int64)})
    config = AdaptiveCardinalityConfig(
        exact_unique_threshold=100,
        exact_memory_budget_bytes=10_000_000,
        hll_precision=10,
        threshold_check_interval=50,
    )
    payload = pandas_adaptive_cardinality(dataframe, "id", config=config).as_dict()
    assert payload["output"]["method"] == "hyperloglog"
    assert payload["hll_state"]["precision"] == 10
    assert payload["hll_state"]["register_count"] == 1 << 10
    assert payload["hll_state"]["register_storage_bytes"] == 1 << 10
    assert isinstance(payload["hll_state"]["zero_register_count"], int)
    assert isinstance(payload["output"]["distinct_count"], int)
    assert isinstance(payload["hll_state"]["raw_estimate"], float)


def test_iter_pandas_hash_arrays_is_uint64_and_chunk_invariant() -> None:
    dataframe = _frame(pd.Series([1, None, 2, 3, 4, 5], dtype="Int64"))
    small = list(iter_pandas_hash_arrays(dataframe, "value", chunk_size=1))
    large = list(iter_pandas_hash_arrays(dataframe, "value", chunk_size=4))
    small_flat = np.concatenate(small)
    large_flat = np.concatenate(large)
    assert all(array.dtype == np.uint64 for array in small + large)
    assert np.array_equal(small_flat, large_flat)


def test_pandas_paths_do_not_call_scalar_hash_function(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import data_quality.cardinality.hashing as hashing_module

    def fail_scalar_hash(*args: object, **kwargs: object) -> int:
        raise AssertionError("scalar hashing must not be used")

    monkeypatch.setattr(hashing_module, "hash_scalar", fail_scalar_hash)
    dataframe = pd.DataFrame({"id": np.arange(1_000, dtype=np.int64)})
    sketch = pandas_hll_sketch(dataframe, "id", chunk_size=128)
    result = pandas_adaptive_cardinality(
        dataframe,
        "id",
        config=AdaptiveCardinalityConfig(
            exact_unique_threshold=100,
            threshold_check_interval=32,
        ),
        chunk_size=128,
    )
    assert sketch.estimate() > 0
    assert result.final_mode is CardinalityMode.HLL


def test_pandas_adaptive_requires_dataframe_not_series() -> None:
    with pytest.raises(TypeError, match="pandas.DataFrame"):
        pandas_adaptive_cardinality(pd.Series([1, 2, 3]), "value")


def test_config_validation_rejects_invalid_thresholds() -> None:
    with pytest.raises(ValueError, match="exact_unique_threshold"):
        AdaptiveCardinalityConfig(exact_unique_threshold=0)
    with pytest.raises(ValueError, match="exact_memory_budget_bytes"):
        AdaptiveCardinalityConfig(exact_memory_budget_bytes=0)
    with pytest.raises(ValueError, match="threshold_check_interval"):
        AdaptiveCardinalityConfig(threshold_check_interval=0)
    with pytest.raises(ValueError, match="at least 1.0"):
        AdaptiveCardinalityConfig(memory_safety_factor=0.5)


def test_config_json_includes_derived_hll_parameters() -> None:
    config = AdaptiveCardinalityConfig(hll_precision=14)
    metadata = config.as_dict()
    assert metadata["hll_register_count"] == 16_384
    assert metadata["hll_relative_standard_error"] == pytest.approx(0.008125)


def test_both_thresholds_can_trigger_same_promotion() -> None:
    dataframe = _frame(pd.Series(["x" * 100 + str(i) for i in range(20)], dtype="string"))
    config = AdaptiveCardinalityConfig(
        exact_unique_threshold=2,
        exact_memory_budget_bytes=300,
        memory_safety_factor=1.0,
        threshold_check_interval=5,
    )
    result = pandas_adaptive_cardinality(dataframe, "value", config=config, chunk_size=20)
    assert result.promotion.reason is PromotionReason.BOTH


def test_handler_direct_array_validation_and_hll_property() -> None:
    from data_quality.cardinality import (
        AdaptiveCardinalityHandler,
        canonicalize_scalar,
        hash_canonical_ndarray,
    )

    handler = AdaptiveCardinalityHandler(
        AdaptiveCardinalityConfig(
            exact_unique_threshold=1,
            exact_memory_budget_bytes=10_000_000,
            threshold_check_interval=1,
        ),
        array_hasher=hash_canonical_ndarray,
    )
    with pytest.raises(ValueError, match="one-dimensional"):
        handler.add_canonical_array(np.array([[canonicalize_scalar(1)]], dtype=object))
    handler.add_canonical_array(np.array([], dtype=object))
    handler.add_canonical_array(
        np.array([canonicalize_scalar(1), canonicalize_scalar(2)], dtype=object)
    )
    assert handler.hll is not None
    assert handler.mode is CardinalityMode.HLL


def test_handler_validates_row_positions_and_result_metrics() -> None:
    from data_quality.cardinality import AdaptiveCardinalityHandler, canonicalize_scalar

    handler = AdaptiveCardinalityHandler()
    values = np.array([canonicalize_scalar(1)], dtype=object)
    with pytest.raises(ValueError, match="row_positions"):
        handler.add_canonical_array(values, row_positions=np.array([0, 1]))
    handler.add_canonical_array(values)
    with pytest.raises(ValueError, match="row metrics"):
        handler.build_result(total_row_count=2, null_count=0, lineage={})


def test_adaptive_config_rejects_non_numeric_or_nonfinite_safety_factor() -> None:
    with pytest.raises(TypeError, match="memory_safety_factor"):
        AdaptiveCardinalityConfig(memory_safety_factor="1.25")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="finite"):
        AdaptiveCardinalityConfig(memory_safety_factor=float("inf"))


def test_adaptive_config_rejects_non_integer_like_thresholds() -> None:
    with pytest.raises(TypeError, match="exact_unique_threshold"):
        AdaptiveCardinalityConfig(exact_unique_threshold=True)
    with pytest.raises(TypeError, match="exact_memory_budget_bytes"):
        AdaptiveCardinalityConfig(exact_memory_budget_bytes=1.5)  # type: ignore[arg-type]


def test_empty_dataframe_column_has_complete_exact_json_output() -> None:
    dataframe = _frame(pd.Series([], dtype="int64"))
    result = pandas_adaptive_cardinality(dataframe, "value")
    payload = result.as_dict()
    assert result.distinct_count == 0
    assert result.is_exact
    assert result.retained_exact_state_bytes == 0
    assert payload["row_metrics"] == {
        "total_row_count": 0,
        "non_null_count": 0,
        "null_count": 0,
    }


def test_exact_memory_helpers_cover_object_and_empty_incremental_arrays() -> None:
    from data_quality.cardinality import AdaptiveCardinalityHandler

    handler = AdaptiveCardinalityHandler()
    object_values = np.array([b"a", b"bb"], dtype=object)

    assert handler._measure_exact_state_raw_bytes(object_values) > object_values.nbytes
    assert handler._measure_exact_state_raw_bytes(np.array([1, 2], dtype=np.int64)) == 16
    incremental = handler._measure_exact_value_raw_bytes(object_values)
    assert incremental.shape == (2,)
    assert int(incremental.sum()) == handler._measure_exact_state_raw_bytes(object_values)
    assert handler._measure_exact_value_raw_bytes(np.array([], dtype=np.int64)).size == 0
    assert handler._values_not_in_sorted_state_mask(
        np.array([1], dtype=np.int64),
        np.array([], dtype=np.int64),
    ).size == 0
