"""Backend detection public contracts"""

from data_quality.backends.base import BackendDetector
from data_quality.backends.builtin import builtin_backend_detectors
from data_quality.backends.registry import BackendRegistry

__all__ = [
    "BackendDetector",
    "BackendRegistry",
    "builtin_backend_detectors",
]