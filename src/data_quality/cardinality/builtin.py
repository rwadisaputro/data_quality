"""Built-in backend adapters for cardinality profiling."""

from __future__ import annotations

from data_quality.cardinality.adaptive import (
    AdaptiveCardinalityConfig,
    AdaptiveCardinalityResult,
)
from data_quality.cardinality.pandas_backend import (
    DEFAULT_PANDAS_CHUNK_SIZE,
    pandas_adaptive_cardinality,
    pandas_dataframe_cardinality,
)
from data_quality.cardinality.polars_backend import (
    DEFAULT_POLARS_BATCH_SIZE,
    polars_adaptive_cardinality,
    polars_dataframe_cardinality,
)
from data_quality.cardinality.pyspark_backend import (
    DEFAULT_PYSPARK_BATCH_SIZE,
    pyspark_adaptive_cardinality,
    pyspark_dataframe_cardinality,
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
            pipeline_metadata=_pipeline_metadata(
                adapter="PandasCardinalityAdapter",
                dtype_source="pandas.DataFrame.dtypes",
                column_selection="single_column",
            ),
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
            pipeline_metadata=_pipeline_metadata(
                adapter="PandasCardinalityAdapter",
                dtype_source="pandas.DataFrame.dtypes",
                column_selection="all_columns",
            ),
        )


class PolarsCardinalityAdapter:
    """Profile eager/lazy Polars frames through streaming native dtype kernels."""

    backend_name = BackendName.POLARS

    def profile(
        self,
        dataframe: object,
        backend: BackendIdentity,
        column: object,
        *,
        config: AdaptiveCardinalityConfig | None,
        batch_size: int | None,
    ) -> AdaptiveCardinalityResult:
        effective_batch_size = DEFAULT_POLARS_BATCH_SIZE if batch_size is None else batch_size
        dtype_source = (
            "polars.LazyFrame.collect_schema"
            if backend.frame_kind.value == "lazy_frame"
            else "polars.DataFrame.schema"
        )
        return polars_adaptive_cardinality(
            dataframe,
            column,
            config=config,
            batch_size=effective_batch_size,
            backend_metadata=backend.as_dict(),
            pipeline_metadata=_pipeline_metadata(
                adapter="PolarsCardinalityAdapter",
                dtype_source=dtype_source,
                column_selection="single_column",
            ),
        )

    def profile_all(
        self,
        dataframe: object,
        backend: BackendIdentity,
        *,
        config: AdaptiveCardinalityConfig | None,
        batch_size: int | None,
    ) -> DataFrameCardinalityResult:
        effective_batch_size = DEFAULT_POLARS_BATCH_SIZE if batch_size is None else batch_size
        dtype_source = (
            "polars.LazyFrame.collect_schema"
            if backend.frame_kind.value == "lazy_frame"
            else "polars.DataFrame.schema"
        )
        return polars_dataframe_cardinality(
            dataframe,
            config=config,
            batch_size=effective_batch_size,
            backend_metadata=backend.as_dict(),
            pipeline_metadata=_pipeline_metadata(
                adapter="PolarsCardinalityAdapter",
                dtype_source=dtype_source,
                column_selection="all_columns",
            ),
        )


class PySparkCardinalityAdapter:
    """Profile Spark SQL DataFrames with distributed two-stage exact/HLL routing."""

    backend_name = BackendName.PYSPARK

    def profile(
        self,
        dataframe: object,
        backend: BackendIdentity,
        column: object,
        *,
        config: AdaptiveCardinalityConfig | None,
        batch_size: int | None,
    ) -> AdaptiveCardinalityResult:
        effective_batch_size = DEFAULT_PYSPARK_BATCH_SIZE if batch_size is None else batch_size
        return pyspark_adaptive_cardinality(
            dataframe,
            column,
            config=config,
            batch_size=effective_batch_size,
            backend_metadata=backend.as_dict(),
            pipeline_metadata=_pipeline_metadata(
                adapter="PySparkCardinalityAdapter",
                dtype_source="pyspark.sql.DataFrame.schema",
                column_selection="single_column",
            ),
        )

    def profile_all(
        self,
        dataframe: object,
        backend: BackendIdentity,
        *,
        config: AdaptiveCardinalityConfig | None,
        batch_size: int | None,
    ) -> DataFrameCardinalityResult:
        effective_batch_size = DEFAULT_PYSPARK_BATCH_SIZE if batch_size is None else batch_size
        return pyspark_dataframe_cardinality(
            dataframe,
            config=config,
            batch_size=effective_batch_size,
            backend_metadata=backend.as_dict(),
            pipeline_metadata=_pipeline_metadata(
                adapter="PySparkCardinalityAdapter",
                dtype_source="pyspark.sql.DataFrame.schema",
                column_selection="all_columns",
            ),
        )


def _pipeline_metadata(
    *,
    adapter: str,
    dtype_source: str,
    column_selection: str,
) -> dict[str, object]:
    return {
        "entrypoint": "profile_cardinality",
        "backend_detection": "data_quality.intake.identify_backend",
        "adapter": adapter,
        "dtype_source": dtype_source,
        "input_boundary": "dataframe",
        "column_selection": column_selection,
    }


def builtin_cardinality_adapters() -> tuple[
    PandasCardinalityAdapter,
    PolarsCardinalityAdapter,
    PySparkCardinalityAdapter,
]:
    """Build the immutable default cardinality-adapter sequence."""

    return (
        PandasCardinalityAdapter(),
        PolarsCardinalityAdapter(),
        PySparkCardinalityAdapter(),
    )
