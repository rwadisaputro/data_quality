"""Immutable backend detector registry"""

from __future__ import annotations

from collections.abc import Iterable

from data_quality.backends.base import BackendDetector, qualified_type_name
from data_quality.exceptions import (
    AmbiguousBackendError,
    BackendDetectorError,
    DuplicateDetectorNameError,
    MissingDataFrameError,
    UnsupportedDataFrameError,
)
from data_quality.models import BackendIdentity



class BackendRegistry:
    """
    Resolve dataframe objects with an immutable detector snapshot

    - The supplied iterable is eagerly converted to a tuple so later changes to
      the caller's collection cannot mutate this registry
    - Extension is functional, `with_detector` returns a new registry and leaves the original unchanged
    """

    def __init__(self, 
                 detectors: Iterable[BackendDetector]) -> None:
        detector_snapshot = tuple(detectors)
        detector_names: set[str] = set()

        for detector in detector_snapshot:
            if detector.name in detector_names:
                raise DuplicateDetectorNameError(detector.name)
            detector_names.add(detector.name)

        self._detectors = detector_snapshot

    @property
    def detectors(self) -> tuple[BackendDetector, ...]:
        """
        Return the registry's ordered immutable detector snapshot
        """

        return self._detectors

    @property
    def supported_inputs(self) -> tuple[str, ...]:
        """
        Return the input type labels advertised by all detectors
        """

        return tuple(
            supported_input
            for detector in self._detectors
            for supported_input in detector.supported_inputs
        )

    def with_detector(self, 
                      detector: BackendDetector) -> BackendRegistry:
        """
        Return a new registry containing `detector` at the end
        """

        return BackendRegistry((*self._detectors, detector))

    def identify(self, 
                 dataframe: object) -> BackendIdentity:
        """
        Identify `dataframe` without inspecting or evaluating its data

        - Every registered detector is consulted so conflicting matches cannot
          be hidden by detector order
        - Unexpected detector exceptions are wrapped with detector and 
          dataframe-type context while preserving the original exception as `__cause__`
        """

        if dataframe is None:
            raise MissingDataFrameError()

        dataframe_type = qualified_type_name(dataframe)
        matches: list[tuple[str, BackendIdentity]] = []

        for detector in self._detectors:
            try:
                backend_identity = detector.detect(dataframe)
            except Exception as error:
                raise BackendDetectorError(
                    detector_name=detector.name,
                    dataframe_type=dataframe_type,
                ) from error

            if backend_identity is not None:
                matches.append((detector.name, backend_identity))

        if not matches:
            raise UnsupportedDataFrameError(
                dataframe_type=dataframe_type,
                supported_inputs=self.supported_inputs,
            )

        if len(matches) > 1:
            raise AmbiguousBackendError(
                dataframe_type=dataframe_type,
                detector_names=tuple(
                    detector_name for detector_name, _ in matches
                ),
            )

        return matches[0][1]