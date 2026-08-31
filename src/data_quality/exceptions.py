"""Exceptions raised at the dataframe intake boundary"""

from __future__ import annotations

from collections.abc import Sequence



class DataQualityError(Exception):
    """
    Base exception for the data-quality library
    """

class DataFrameInputError(DataQualityError, 
                          TypeError):
    """
    Base exception for invalid dataframe inputs
    """

class MissingDataFrameError(DataFrameInputError):
    """
    Raised when the caller supplies `None` instead of a dataframe
    """

    def __init__(self) -> None:
        super().__init__("Expected a dataframe but received `None`")

class UnsupportedDataFrameError(DataFrameInputError):
    """
    Raised when no registered detector supports the supplied type
    """

    def __init__(self, 
                 dataframe_type: str, 
                 supported_inputs: Sequence[str]) -> None:
        if supported_inputs:
            supported = ", ".join(supported_inputs)
            message = (
                f"Unsupported dataframe type '{dataframe_type}', supported inputs are: {supported}"
            )
        else:
            message = (
                f"Unsupported dataframe type '{dataframe_type}', no backend detectors are registered"
            )
        super().__init__(message)
        self.dataframe_type = dataframe_type
        self.supported_inputs = tuple(supported_inputs)

class AmbiguousBackendError(DataFrameInputError):
    """
    Raised when more than one detector claims the same object
    """

    def __init__(self, 
                 dataframe_type: str, 
                 detector_names: Sequence[str]) -> None:
        names = ", ".join(detector_names)
        super().__init__(
            f"Backend detection is ambiguous for '{dataframe_type}' with matching detectors of {names}"
        )
        self.dataframe_type = dataframe_type
        self.detector_names = tuple(detector_names)

class BackendDetectorError(DataQualityError, 
                           RuntimeError):
    """
    Raised when a detector itself fails unexpectedly
    """

    def __init__(self, 
                 detector_name: str, 
                 dataframe_type: str) -> None:
        super().__init__(
            f"Backend detector '{detector_name}' failed while inspecting '{dataframe_type}'"
        )
        self.detector_name = detector_name
        self.dataframe_type = dataframe_type

class DuplicateDetectorNameError(DataQualityError, 
                                 ValueError):
    """
    Raised when an immutable registry is built with duplicate names
    """

    def __init__(self, 
                 detector_name: str) -> None:
        super().__init__(f"A backend detector named '{detector_name}' is already registered")
        self.detector_name = detector_name

class PhysicalSchemaError(DataQualityError):
    """
    Base exception for native physical-schema discovery failures
    """

class DuplicateSchemaDiscovererError(PhysicalSchemaError, 
                                     ValueError):
    """
    Raised when more than one discoverer is registered for a backend
    """

    def __init__(self, 
                 backend_name: str) -> None:
        super().__init__(
            f"A physical-schema discoverer for backend '{backend_name}' is already registered"
        )
        self.backend_name = backend_name

class UnsupportedSchemaDiscoveryBackendError(PhysicalSchemaError, 
                                             NotImplementedError):
    """
    Raised when a detected backend has no schema-discovery adapter
    """

    def __init__(self, 
                 backend_name: str) -> None:
        super().__init__(
            f"No physical-schema discoverer is registered for backend '{backend_name}'"
        )
        self.backend_name = backend_name

class PhysicalSchemaDiscoveryError(PhysicalSchemaError, 
                                   RuntimeError):
    """
    Raised when a backend discoverer fails unexpectedly
    """

    def __init__(self, 
                 backend_name: str, 
                 dataframe_type: str) -> None:
        super().__init__(
            f"Physical-schema discovery failed for backend '{backend_name}' and dataframe type '{dataframe_type}'"
        )
        self.backend_name = backend_name
        self.dataframe_type = dataframe_type
class CardinalityError(DataQualityError):
    """Base exception for dataframe cardinality profiling failures."""


class DuplicateCardinalityAdapterError(CardinalityError, ValueError):
    """Raised when more than one cardinality adapter is registered for a backend."""

    def __init__(self, backend_name: str) -> None:
        super().__init__(
            f"A cardinality adapter for backend '{backend_name}' is already registered"
        )
        self.backend_name = backend_name


class UnsupportedCardinalityBackendError(CardinalityError, NotImplementedError):
    """Raised when a detected backend has no cardinality adapter."""

    def __init__(self, backend_name: str) -> None:
        super().__init__(f"No cardinality adapter is registered for backend '{backend_name}'")
        self.backend_name = backend_name


class CardinalityProfilingError(CardinalityError, RuntimeError):
    """Raised when a backend cardinality adapter fails unexpectedly."""

    def __init__(self, backend_name: str, dataframe_type: str) -> None:
        super().__init__(
            f"Cardinality profiling failed for backend '{backend_name}' "
            f"and dataframe type '{dataframe_type}'"
        )
        self.backend_name = backend_name
        self.dataframe_type = dataframe_type
