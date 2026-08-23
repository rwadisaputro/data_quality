"""Built-in detectors for pandas, Polars and PySpark dataframes"""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Any

from data_quality.backends.base import (
    method_resolution_order_uses_prefix,
    module_roots_in_method_resolution_order,
    qualified_type_name,
)
from data_quality.models import (
    BackendIdentity,
    BackendName,
    ConnectionMode,
    DataFrameApi,
    DistributionMode,
    EvaluationMode,
    FrameKind,
)



def _client_library_version(module: ModuleType) -> str | None:
    version = getattr(module, "__version__", None)

    return str(version) if version is not None else None

def _public_type(module: ModuleType, 
                 attribute_name: str) -> type[Any]:
    candidate = getattr(module, attribute_name)
    if not isinstance(candidate, type):
        raise TypeError(
            f"Expected '{module.__name__}.{attribute_name}' to be a type, "
            f"but received '{type(candidate).__name__}'."
        )
    
    return candidate

class PandasBackendDetector:
    """
    Detect `pandas.DataFrame` instances and subclasses
    """

    name = "pandas"
    supported_inputs = ("pandas.DataFrame",)

    def detect(self, 
               dataframe: object) -> BackendIdentity | None:
        if "pandas" not in module_roots_in_method_resolution_order(dataframe):

            return None

        pandas_module = import_module("pandas")
        pandas_dataframe_type = _public_type(pandas_module, "DataFrame")
        if not isinstance(dataframe, pandas_dataframe_type):

            return None

        return BackendIdentity(
            name = BackendName.PANDAS,
            dataframe_api = DataFrameApi.PANDAS,
            frame_kind = FrameKind.DATAFRAME,
            evaluation_mode = EvaluationMode.EAGER,
            distribution_mode = DistributionMode.LOCAL,
            connection_mode = ConnectionMode.IN_PROCESS,
            dataframe_type = qualified_type_name(dataframe),
            client_library_version = _client_library_version(pandas_module),
        )

class PolarsBackendDetector:
    """
    Detect both eager and lazy Polars frames
    """

    name = "polars"
    supported_inputs = (
        "polars.DataFrame", 
        "polars.LazyFrame"
    )

    def detect(self, 
               dataframe: object) -> BackendIdentity | None:
        if "polars" not in module_roots_in_method_resolution_order(dataframe):

            return None

        polars_module = import_module("polars")
        polars_dataframe_type = _public_type(polars_module, "DataFrame")
        polars_lazy_frame_type = _public_type(polars_module, "LazyFrame")

        if isinstance(dataframe, polars_dataframe_type):
            frame_kind = FrameKind.DATAFRAME
            evaluation_mode = EvaluationMode.EAGER
        elif isinstance(dataframe, polars_lazy_frame_type):
            frame_kind = FrameKind.LAZY_FRAME
            evaluation_mode = EvaluationMode.LAZY
        else:

            return None

        return BackendIdentity(
            name = BackendName.POLARS,
            dataframe_api = DataFrameApi.POLARS,
            frame_kind = frame_kind,
            evaluation_mode = evaluation_mode,
            distribution_mode = DistributionMode.LOCAL,
            connection_mode = ConnectionMode.IN_PROCESS,
            dataframe_type = qualified_type_name(dataframe),
            client_library_version = _client_library_version(polars_module),
        )

class PySparkBackendDetector:
    """
    Detect Spark SQL DataFrames in classic and Spark Connect modes
    """

    name = "pyspark"
    supported_inputs = (
        "pyspark.sql.DataFrame (Spark classic)",
        "pyspark.sql.DataFrame (Spark Connect)",
    )

    def detect(self, 
               dataframe: object) -> BackendIdentity | None:
        if "pyspark" not in module_roots_in_method_resolution_order(dataframe):

            return None

        pyspark_module = import_module("pyspark")
        pyspark_sql_module = import_module("pyspark.sql")
        spark_dataframe_types: list[type[Any]] = [
            _public_type(pyspark_sql_module, "DataFrame")
        ]

        # Spark 3.4+ exposes a dedicated Connect implementation
            # It subclasses the public `DataFrame` in current Spark releases:
            # - Explicitly adding its already-loaded concrete type keeps detection compatible with
            #   older 3.x layouts without importing Connect for classic users.
        if method_resolution_order_uses_prefix(dataframe, "pyspark.sql.connect"):
            connect_dataframe_module = import_module("pyspark.sql.connect.dataframe")
            connect_dataframe_type = _public_type(connect_dataframe_module, "DataFrame")
            if connect_dataframe_type not in spark_dataframe_types:
                spark_dataframe_types.append(connect_dataframe_type)

        if not isinstance(dataframe, tuple(spark_dataframe_types)):

            return None

        is_spark_connect = method_resolution_order_uses_prefix(
            dataframe,
            "pyspark.sql.connect",
        )

        return BackendIdentity(
            name = BackendName.PYSPARK,
            dataframe_api = DataFrameApi.SPARK_SQL,
            frame_kind = FrameKind.DATAFRAME,
            evaluation_mode = EvaluationMode.LAZY,
            distribution_mode = DistributionMode.DISTRIBUTED,
            connection_mode = (
                ConnectionMode.SPARK_CONNECT
                if is_spark_connect
                else ConnectionMode.SPARK_CLASSIC
            ),
            dataframe_type = qualified_type_name(dataframe),
            client_library_version = _client_library_version(pyspark_module),
        )

def builtin_backend_detectors() -> tuple[PandasBackendDetector,
                                         PolarsBackendDetector,
                                         PySparkBackendDetector]:
    """
    Build the immutable default detector sequence
    """

    return (
        PandasBackendDetector(),
        PolarsBackendDetector(),
        PySparkBackendDetector(),
    )