from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

import data_quality.cardinality.adaptive as adaptive_module
import data_quality.cardinality.hashing as hashing_module
import data_quality.cardinality.hll as hll_module
import data_quality.cardinality.pandas as pandas_module_impl
from data_quality import (
    AdaptiveCardinalityConfig,
    CardinalityProfilingError,
    CardinalityRegistry,
    DataFrameInput,
    identify_backend,
    profile_cardinality_from_input,
)
from data_quality.cardinality import (
    AdaptiveCardinalityHandler,
    HashConfiguration,
    HyperLogLog,
    PandasCardinalityAdapter,
    PandasDtypeFamily,
    hash_canonical_ndarray,
    identify_pandas_dtype,
)
from data_quality.models import BackendName


def _raise_import_error(name: str) -> object:
    raise ImportError(name)


def test_adaptive_config_rejects_non_hash_configuration() -> None:
    with pytest.raises(TypeError, match="HashConfiguration"):
        AdaptiveCardinalityConfig(hash_configuration=object())  # type: ignore[arg-type]


def test_adaptive_handler_requires_backend_array_hasher_after_promotion() -> None:
    handler = AdaptiveCardinalityHandler(
        AdaptiveCardinalityConfig(
            exact_unique_threshold=1,
            exact_memory_budget_bytes=10_000_000,
            threshold_check_interval=1,
        )
    )
    with pytest.raises(RuntimeError, match="backend-provided ndarray hash function"):
        handler.add_canonical_array(np.array([b"i1", b"i2"], dtype=object))


def test_adaptive_numpy_dependency_error_is_clear(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(adaptive_module, "import_module", _raise_import_error)
    with pytest.raises(ImportError, match="Adaptive cardinality requires NumPy"):
        AdaptiveCardinalityHandler()


def test_profile_from_input_rejects_native_dataframe() -> None:
    with pytest.raises(TypeError, match="Expected `DataFrameInput`"):
        profile_cardinality_from_input(pd.DataFrame({"id": [1]}), "id")  # type: ignore[arg-type]


def test_hash_ndarray_rejects_multidimensional_and_supports_empty() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        hash_canonical_ndarray(np.array([[b"i1"]], dtype=object))

    result = hash_canonical_ndarray(np.empty(0, dtype=object))
    assert result.dtype == np.uint64
    assert result.size == 0


def test_hash_ndarray_dependency_error_is_clear(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hashing_module, "import_module", _raise_import_error)
    with pytest.raises(ImportError, match="optional 'pandas' dependency"):
        hashing_module._load_array_hash_dependencies()


def test_hll_bulk_update_accepts_nonnegative_signed_integer_array() -> None:
    sketch = HyperLogLog(precision=10)
    sketch.add_hash_array(np.array([0, 1, 2, 3], dtype=np.int64))
    assert sum(sketch.registers) > 0


def test_hll_bulk_update_dependency_error_is_clear(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hll_module, "import_module", _raise_import_error)
    with pytest.raises(ImportError, match="Bulk HyperLogLog updates require NumPy"):
        HyperLogLog._load_numpy()


def test_pandas_dtype_classifier_covers_numpy_string_and_other_fallbacks() -> None:
    assert identify_pandas_dtype(np.dtype("U3")).family is PandasDtypeFamily.STRING
    assert identify_pandas_dtype(np.dtype("V4")).family is PandasDtypeFamily.OTHER


def test_pandas_dtype_classifier_dependency_error_is_clear(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pandas_module_impl, "import_module", _raise_import_error)
    with pytest.raises(ImportError, match="optional 'pandas' dependency"):
        identify_pandas_dtype(np.dtype("int64"))


def test_pandas_private_canonical_array_validates_shape_and_empty() -> None:
    dtype_identity = identify_pandas_dtype(np.dtype("int64"))

    with pytest.raises(ValueError, match="one-dimensional"):
        pandas_module_impl._canonicalize_pandas_ndarray(
            np.array([[1]], dtype=np.int64),
            dtype_identity=dtype_identity,
            pandas_module=pd,
            numpy_module=np,
        )

    result = pandas_module_impl._canonicalize_pandas_ndarray(
        np.empty(0, dtype=np.int64),
        dtype_identity=dtype_identity,
        pandas_module=pd,
        numpy_module=np,
    )
    assert result.dtype == np.dtype("int64")
    assert result.size == 0


def test_pandas_scalar_canonicalizer_handles_numpy_boolean() -> None:
    assert (
        pandas_module_impl._canonicalize_pandas_scalar(
            np.bool_(True),
            pandas_module=pd,
            numpy_module=np,
        )
        == b"b\x01"
    )


def test_pandas_series_validator_rejects_non_series() -> None:
    with pytest.raises(TypeError, match="pandas.Series"):
        pandas_module_impl._validate_pandas_series(pd.DataFrame({"id": [1]}))


def test_pandas_series_validator_dependency_error_is_clear(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pandas_module_impl, "import_module", _raise_import_error)
    with pytest.raises(ImportError, match="optional 'pandas' dependency"):
        pandas_module_impl._validate_pandas_series(pd.Series([1]))


class _InvalidBackendNameAdapter:
    backend_name = "pandas"


@dataclass
class _ExplodingAdapter:
    backend_name: BackendName = BackendName.PANDAS

    def profile(self, *args: object, **kwargs: object) -> object:
        raise RuntimeError("boom")


def test_registry_rejects_non_enum_backend_name() -> None:
    with pytest.raises(TypeError, match="BackendName"):
        CardinalityRegistry((_InvalidBackendNameAdapter(),))  # type: ignore[arg-type]


def test_registry_exposes_immutable_snapshot_and_with_adapter() -> None:
    registry = CardinalityRegistry(())
    assert registry.adapters == ()

    extended = registry.with_adapter(PandasCardinalityAdapter())
    assert registry.adapters == ()
    assert len(extended.adapters) == 1
    assert extended.adapters[0].backend_name is BackendName.PANDAS


def test_registry_rejects_non_dataframe_input() -> None:
    with pytest.raises(TypeError, match="DataFrameInput"):
        CardinalityRegistry((PandasCardinalityAdapter(),)).profile(
            object(),  # type: ignore[arg-type]
            "id",
            config=None,
            batch_size=None,
        )


def test_registry_wraps_unexpected_adapter_failure() -> None:
    dataframe = pd.DataFrame({"id": [1, 2]})
    dataframe_input = DataFrameInput(dataframe=dataframe, backend=identify_backend(dataframe))
    registry = CardinalityRegistry((_ExplodingAdapter(),))

    with pytest.raises(CardinalityProfilingError) as exc_info:
        registry.profile(dataframe_input, "id", config=None, batch_size=None)

    assert exc_info.value.__cause__ is not None
    assert "pandas" in str(exc_info.value)
