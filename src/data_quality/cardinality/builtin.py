"""Built-in backend adapters for cardinality profiling."""

from __future__ import annotations

from data_quality.cardinality.adaptive import (
    AdaptiveCardinalityConfig,
    AdaptiveCardinalityResult,
)
from data_quality.cardinality.pandas import (
    DEFAULT_PANDAS_CHUNK_SIZE,
    pandas_adaptive_cardinality,
    pandas_dataframe_cardinality,
)
from data_quality.cardinality.report import DataFrameCardinalityResult
from data_quality.models import BackendIdentity, BackendName


class PandasCardinalityAdapter:
    """Profile pandas DataFrames using native dtype metadata and ndarray kernels."""

    backend_name = BackendName.PANDAS

    def profile(
        self,
        dataframe: object,
        backend: BackendIdentity,
        column: object,
        *,
        config: AdaptiveCardinalityConfig | None,
        batch_size: int | None,
    ) -> AdaptiveCardinalityResult:
        effective_batch_size = DEFAULT_PANDAS_CHUNK_SIZE if batch_size is None else batch_size
        return pandas_adaptive_cardinality(
            dataframe,
            column,
            config=config,
            chunk_size=effective_batch_size,
            backend_metadata=backend.as_dict(),
            pipeline_metadata={
                "entrypoint": "profile_cardinality",
                "backend_detection": "data_quality.intake.identify_backend",
                "adapter": "PandasCardinalityAdapter",
                "dtype_source": "pandas.DataFrame.dtypes",
                "input_boundary": "dataframe",
                "column_selection": "single_column",
            },
        )

    def profile_all(
        self,
        dataframe: object,
        backend: BackendIdentity,
        *,
        config: AdaptiveCardinalityConfig | None,
        batch_size: int | None,
    ) -> DataFrameCardinalityResult:
        effective_batch_size = DEFAULT_PANDAS_CHUNK_SIZE if batch_size is None else batch_size
        return pandas_dataframe_cardinality(
            dataframe,
            config=config,
            chunk_size=effective_batch_size,
            backend_metadata=backend.as_dict(),
            pipeline_metadata={
                "entrypoint": "profile_cardinality",
                "backend_detection": "data_quality.intake.identify_backend",
                "adapter": "PandasCardinalityAdapter",
                "dtype_source": "pandas.DataFrame.dtypes",
                "input_boundary": "dataframe",
                "column_selection": "all_columns",
            },
        )


def builtin_cardinality_adapters() -> tuple[PandasCardinalityAdapter]:
    """Build the immutable default cardinality-adapter sequence."""

    return (PandasCardinalityAdapter(),)
