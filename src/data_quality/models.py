"""Value objects used while accepting a user-supplied dataframe"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Generic, TypeVar



DataFrameType = TypeVar("DataFrameType", covariant = True)

class BackendName(str, 
                  Enum):
    """Native execution engine backing the supplied dataframe"""

    PANDAS = "pandas"
    POLARS = "polars"
    PYSPARK = "pyspark"

class DataFrameApi(str, 
                   Enum):
    """
    User-facing dataframe API exposed by the supplied object
    """

    PANDAS = "pandas"
    POLARS = "polars"
    SPARK_SQL = "spark_sql"

class FrameKind(str, 
                Enum):
    """
    Concrete frame abstraction received at the public API boundary
    """

    DATAFRAME = "dataframe"
    LAZY_FRAME = "lazy_frame"

class EvaluationMode(str, 
                     Enum):
    """
    Whether operations normally execute immediately or by building a query plan
    """

    EAGER = "eager"
    LAZY = "lazy"

class DistributionMode(str, 
                       Enum):
    """
    Whether execution is local or distributed across workers
    """

    LOCAL = "local"
    DISTRIBUTED = "distributed"


class ConnectionMode(str, 
                     Enum):
    """
    How the Python API communicates with its execution engine
    """

    IN_PROCESS = "in_process"
    SPARK_CLASSIC = "spark_classic"
    SPARK_CONNECT = "spark_connect"


@dataclass(frozen = True, slots = True)
class BackendIdentity:
    """
    Serialisable backend metadata used for downstream dispatch

    - `client_library_version` intentionally records only the Python package version
    - In Spark Connect, the remote Spark server can have a different version, 
      discovering it would require communication with the server and is
      outside the no-execution contract of backend identification
    """

    name: BackendName
    dataframe_api: DataFrameApi
    frame_kind: FrameKind
    evaluation_mode: EvaluationMode
    distribution_mode: DistributionMode
    connection_mode: ConnectionMode
    dataframe_type: str
    client_library_version: str | None = None

    def as_dict(self) -> dict[str, str | None]:
        """
        Return report-friendly primitive values
        """

        return {
            "name": self.name.value,
            "dataframe_api": self.dataframe_api.value,
            "frame_kind": self.frame_kind.value,
            "evaluation_mode": self.evaluation_mode.value,
            "distribution_mode": self.distribution_mode.value,
            "connection_mode": self.connection_mode.value,
            "dataframe_type": self.dataframe_type,
            "client_library_version": self.client_library_version,
        }

@dataclass(frozen = True, slots = True, eq = False)
class DataFrameInput(Generic[DataFrameType]):
    """
    The original native dataframe paired with its detected backend

    - The dataframe is deliberately excluded from `repr` so logging this
      context cannot accidentally print user data or trigger an expensive frame representation 
    - Equality is disabled because dataframe equality operators commonly return 
      another dataframe rather than a single boolean value
    """

    dataframe: DataFrameType = field(repr = False)
    backend: BackendIdentity