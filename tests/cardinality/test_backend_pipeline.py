from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from data_quality import (
    AdaptiveCardinalityConfig,
    BackendName,
    CardinalityRegistry,
    DataFrameInput,
    DuplicateCardinalityAdapterError,
    UnsupportedCardinalityBackendError,
    identify_backend,
    profile_cardinality,
    profile_cardinality_from_input,
    profile_cardinality_json,
)
from data_quality.cardinality import (
    PandasCardinalityAdapter,
    PandasDtypeFamily,
    identify_pandas_dtype,
)


def test_dataframe_api_detects_backend_before_pandas_dispatch() -> None:
    dataframe = pd.DataFrame({"customer_id": pd.Series([1, 2, 2, None], dtype="Int64")})
    expected_backend = identify_backend(dataframe)

    result = profile_cardinality(dataframe, "customer_id")
    lineage = result.as_dict()["lineage"]

    assert lineage["backend"] == expected_backend.as_dict()
    assert lineage["column"]["name"] == "customer_id"
    assert lineage["column"]["position"] == 0
    assert lineage["column"]["native_dtype"] == "Int64"
    assert lineage["column"]["dtype_family"] == "integer"
    assert lineage["column"]["dtype_source"] == "pandas.DataFrame.dtypes"
    assert lineage["dtype_family"] == "integer"
    assert lineage["pipeline"] == {
        "entrypoint": "profile_cardinality",
        "backend_detection": "data_quality.intake.identify_backend",
        "adapter": "PandasCardinalityAdapter",
        "dtype_source": "pandas.DataFrame.dtypes",
        "input_boundary": "dataframe",
        "column_selection": "single_column",
    }
    assert result.distinct_count == 2


def test_dataframe_json_contains_full_backend_and_dtype_lineage() -> None:
    dataframe = pd.DataFrame(
        {
            "flag": pd.Series([True, False, True, None], dtype="boolean"),
            "value": np.arange(4),
        }
    )

    payload = json.loads(profile_cardinality_json(dataframe, "flag", batch_size=2))

    assert payload["schema_version"] == "adaptive-cardinality-v6"
    assert payload["lineage"]["backend"]["name"] == "pandas"
    assert payload["lineage"]["backend"]["dataframe_api"] == "pandas"
    assert payload["lineage"]["backend"]["frame_kind"] == "dataframe"
    assert payload["lineage"]["column"]["native_dtype"] == "boolean"
    assert payload["lineage"]["column"]["dtype_family"] == "boolean"
    assert payload["lineage"]["column"]["is_extension_dtype"] is True
    assert payload["parameters"]["execution"]["batch_size"] == 2
    assert payload["parameters"]["execution"]["pandas_chunk_size"] == 2


def test_dataframe_api_promotes_using_same_backend_neutral_handler() -> None:
    dataframe = pd.DataFrame({"id": np.arange(500, dtype=np.int64)})
    config = AdaptiveCardinalityConfig(
        exact_unique_threshold=50,
        exact_memory_budget_bytes=10_000_000,
        threshold_check_interval=16,
    )

    result = profile_cardinality(dataframe, "id", config=config, batch_size=64)

    assert not result.is_exact
    assert result.promotion.occurred
    assert result.promotion.unique_count == 51
    assert result.as_dict()["lineage"]["column"]["dtype_family"] == "integer"


def test_pandas_adapter_uses_dataframe_dtype_metadata_not_value_sampling() -> None:
    dataframe = pd.DataFrame(
        {
            "nullable_id": pd.Series([1, 2, None], dtype="Int64"),
            "object_id": pd.Series([1, 2, None], dtype="object"),
        }
    )

    nullable = profile_cardinality(dataframe, "nullable_id").as_dict()["lineage"]["column"]
    object_column = profile_cardinality(dataframe, "object_id").as_dict()["lineage"]["column"]

    assert nullable["native_dtype"] == "Int64"
    assert nullable["dtype_family"] == "integer"
    assert object_column["native_dtype"] == "object"
    assert object_column["dtype_family"] == "object"


def test_missing_and_duplicate_pandas_column_labels_are_rejected() -> None:
    dataframe = pd.DataFrame([[1, 2]], columns=["id", "id"])

    with pytest.raises(ValueError, match="not unique"):
        profile_cardinality(dataframe, "id")

    with pytest.raises(KeyError, match="no column"):
        profile_cardinality(pd.DataFrame({"id": [1]}), "missing")


def test_profile_from_preidentified_dataframe_input_reuses_backend_metadata() -> None:
    dataframe = pd.DataFrame({"id": [1, 2, 2]})
    backend = identify_backend(dataframe)
    dataframe_input = DataFrameInput(dataframe=dataframe, backend=backend)

    result = profile_cardinality_from_input(dataframe_input, "id")

    assert result.as_dict()["lineage"]["backend"] == backend.as_dict()


def test_registry_rejects_duplicate_backend_adapters() -> None:
    with pytest.raises(DuplicateCardinalityAdapterError):
        CardinalityRegistry((PandasCardinalityAdapter(), PandasCardinalityAdapter()))


def test_registry_reports_detected_backend_without_registered_adapter() -> None:
    dataframe = pd.DataFrame({"id": [1, 2]})
    dataframe_input = DataFrameInput(dataframe=dataframe, backend=identify_backend(dataframe))

    with pytest.raises(UnsupportedCardinalityBackendError):
        profile_cardinality_from_input(
            dataframe_input,
            "id",
            registry=CardinalityRegistry(()),
        )


def test_dtype_classifier_covers_core_pandas_dtype_families() -> None:
    cases = [
        (pd.Series([True], dtype="boolean").dtype, PandasDtypeFamily.BOOLEAN),
        (pd.Series([1], dtype="Int64").dtype, PandasDtypeFamily.INTEGER),
        (pd.Series([1.0], dtype="float64").dtype, PandasDtypeFamily.FLOATING),
        (pd.Series([1 + 2j], dtype="complex128").dtype, PandasDtypeFamily.COMPLEX),
        (pd.Series(["a"], dtype="string").dtype, PandasDtypeFamily.STRING),
        (pd.Series(["a"], dtype="category").dtype, PandasDtypeFamily.CATEGORICAL),
        (pd.Series(pd.to_datetime(["2024-01-01"])).dtype, PandasDtypeFamily.DATETIME),
        (
            pd.Series(pd.to_datetime(["2024-01-01"], utc=True)).dtype,
            PandasDtypeFamily.DATETIME_TZ,
        ),
        (pd.Series(pd.to_timedelta([1], unit="s")).dtype, PandasDtypeFamily.TIMEDELTA),
        (
            pd.Series(pd.period_range("2024-01", periods=1, freq="M")).dtype,
            PandasDtypeFamily.PERIOD,
        ),
        (
            pd.Series(pd.arrays.IntervalArray.from_tuples([(0, 1)])).dtype,
            PandasDtypeFamily.INTERVAL,
        ),
        (pd.Series([object()], dtype="object").dtype, PandasDtypeFamily.OBJECT),
    ]

    for dtype, expected_family in cases:
        assert identify_pandas_dtype(dtype).family is expected_family


def test_registry_adapter_backend_name_is_pandas_enum() -> None:
    assert PandasCardinalityAdapter.backend_name is BackendName.PANDAS
