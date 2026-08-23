"""Data-quality library public APIs"""

from data_quality.backends import BackendDetector, BackendRegistry
from data_quality.exceptions import (
    AmbiguousBackendError,
    BackendDetectorError,
    DataFrameInputError,
    DataQualityError,
    DuplicateDetectorNameError,
    DuplicateSchemaDiscovererError,
    MissingDataFrameError,
    PhysicalSchemaDiscoveryError,
    PhysicalSchemaError,
    UnsupportedDataFrameError,
    UnsupportedSchemaDiscoveryBackendError,
)
from data_quality.intake import (
    DEFAULT_BACKEND_REGISTRY,
    identify_backend,
    resolve_dataframe_input,
)
from data_quality.models import (
    BackendIdentity,
    BackendName,
    ConnectionMode,
    DataFrameApi,
    DataFrameInput,
    DistributionMode,
    EvaluationMode,
    FrameKind,
)
from data_quality.schema import (
    DEFAULT_PHYSICAL_SCHEMA_REGISTRY,
    discover_physical_schema,
    discover_physical_schema_from_input,
)
from data_quality.schema_discovery import (
    PhysicalSchemaDiscoverer,
    PhysicalSchemaRegistry,
)
from data_quality.schema_models import (
    DataFrameDimensions,
    DataFramePhysicalSchema,
    NativeColumnSchema,
    RowCountMode,
    RowCountSource,
)

__all__ = [
    "AmbiguousBackendError",
    "BackendDetector",
    "BackendDetectorError",
    "BackendIdentity",
    "BackendName",
    "BackendRegistry",
    "ConnectionMode",
    "DEFAULT_BACKEND_REGISTRY",
    "DEFAULT_PHYSICAL_SCHEMA_REGISTRY",
    "DataFrameApi",
    "DataFrameDimensions",
    "DataFrameInput",
    "DataFrameInputError",
    "DataFramePhysicalSchema",
    "DataQualityError",
    "DistributionMode",
    "DuplicateDetectorNameError",
    "DuplicateSchemaDiscovererError",
    "EvaluationMode",
    "FrameKind",
    "MissingDataFrameError",
    "NativeColumnSchema",
    "PhysicalSchemaDiscoverer",
    "PhysicalSchemaDiscoveryError",
    "PhysicalSchemaError",
    "PhysicalSchemaRegistry",
    "RowCountMode",
    "RowCountSource",
    "UnsupportedDataFrameError",
    "UnsupportedSchemaDiscoveryBackendError",
    "discover_physical_schema",
    "discover_physical_schema_from_input",
    "identify_backend",
    "resolve_dataframe_input",
]