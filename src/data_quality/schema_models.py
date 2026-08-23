"""Value objects produced by native physical-schema discovery"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from data_quality.models import BackendIdentity



class RowCountMode(str, 
                   Enum):
    """
    Whether discovery may execute a dataframe query for row count or not
    """

    METADATA_ONLY = "metadata_only"
    EXACT = "exact"

class RowCountSource(str, 
                     Enum):
    """
    Describe how or why not an exact row count was obtained
    """

    FRAME_METADATA = "frame_metadata"
    EXECUTED_QUERY = "executed_query"
    NOT_COMPUTED = "not_computed"
    UNBOUNDED = "unbounded"

@dataclass(frozen = True, slots = True, eq = False)
class NativeColumnSchema:
    """
    One native column definition in its original ordinal position

    - `name` and `native_data_type` retain the exact backend objects for the later 
      schema-normalisation stage 
    - Their string forms are stored separately so reports and logs do not have to understand 
      pandas, Polars, or PySpark type objects
    - Equality is disabled because third-party objects may define non-scalar equality operations
    """

    position: int
    name: object = field(repr = False)
    name_string: str
    native_data_type: object = field(repr = False)
    native_data_type_string: str
    native_data_type_class: str
    nullable: bool | None

    def __post_init__(self) -> None:
        if self.position < 0:
            raise ValueError("Column position must be non-negative")

    def as_dict(self) -> dict[str, int | str | bool | None]:
        """
        Return a report-friendly representation without backend objects
        """

        return {
            "position": self.position,
            "name": self.name_string,
            "native_data_type": self.native_data_type_string,
            "native_data_type_class": self.native_data_type_class,
            "nullable": self.nullable,
        }

@dataclass(frozen = True, slots = True)
class DataFrameDimensions:
    """
    Known dataframe dimensions and provenance for the row count

    - `row_count` is `None` only when a row-count query was deliberately not executed 
      or when the dataframe represents an unbounded stream 
    - Column count is always known after native schema discovery
    """

    row_count: int | None
    column_count: int
    row_count_source: RowCountSource

    def __post_init__(self) -> None:
        if self.column_count < 0:
            raise ValueError("Column count must be non-negative")
        if self.row_count is not None and self.row_count < 0:
            raise ValueError("Row count must be non-negative when present")

        source_without_count = self.row_count_source in {
            RowCountSource.NOT_COMPUTED,
            RowCountSource.UNBOUNDED,
        }
        if (self.row_count is None) != source_without_count:
            raise ValueError(
                "Row count and row-count source describe inconsistent states"
            )

    @property
    def shape(self) -> tuple[int | None, int]:
        """
        Return `(rows, columns)`, rows may be unknown or unbounded
        """

        return (self.row_count, self.column_count)

    @property
    def has_exact_row_count(self) -> bool:
        """
        Return whether a finite exact row count is available
        """

        return self.row_count is not None

    def as_dict(self) -> dict[str, int | str | bool | None | list[int | None]]:
        """
        Return report-friendly primitive values
        """

        return {
            "row_count": self.row_count,
            "column_count": self.column_count,
            "shape": [self.row_count, self.column_count],
            "row_count_source": self.row_count_source.value,
            "has_exact_row_count": self.has_exact_row_count,
        }

@dataclass(frozen = True, slots = True, eq = False)
class DataFramePhysicalSchema:
    """
    Native column schema and dimensions for one dataframe

    - The complete backend-native schema object is retained for the next pipeline
    stage and excluded from `repr` 
    - `columns` provides a backend-neutral envelope without normalising the native type information itself
    """

    backend: BackendIdentity
    native_schema: object = field(repr = False)
    columns: tuple[NativeColumnSchema, ...]
    dimensions: DataFrameDimensions

    def __post_init__(self) -> None:
        if len(self.columns) != self.dimensions.column_count:
            raise ValueError("Discovered column definitions do not match the column count")

        positions = tuple(column.position for column in self.columns)
        expected_positions = tuple(range(len(self.columns)))
        if positions != expected_positions:
            raise ValueError("Discovered column positions must be contiguous and ordered")

    @property
    def shape(self) -> tuple[int | None, int]:
        """
        Return the discovered `(rows, columns)` dimensions
        """

        return self.dimensions.shape

    def as_dict(self) -> dict[str, object]:
        """
        Return a serialisable representation without native type objects
        """

        return {
            "backend": self.backend.as_dict(),
            "dimensions": self.dimensions.as_dict(),
            "columns": [column.as_dict() for column in self.columns],
        }