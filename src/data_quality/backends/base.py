"""Backend-detector contracts and shared introspection helpers"""

from __future__ import annotations

from typing import Protocol

from data_quality.models import BackendIdentity



def qualified_type_name(value: object) -> str:
    """
    Return a type name without evaluating `repr(value)`
    """

    value_type = type(value)

    return f"{value_type.__module__}.{value_type.__qualname__}"

def module_roots_in_method_resolution_order(value: object) -> frozenset[str]:
    """
    Return top-level module names for every class in the value's method-resolution order

    Including the full method-resolution order means subclasses declared in a
    user's own module are still recognized as native dataframe subclasses
    """

    return frozenset(
        base_type.__module__.partition(".")[0]
        for base_type in type(value).__mro__
    )

def method_resolution_order_uses_prefix(value: object, 
                                        module_prefix: str) -> bool:
    """
    Return whether any class in the method-resolution order originates below `module_prefix`
    """

    prefix_with_separator = f"{module_prefix}"

    return any(
        base_type.__module__ == module_prefix
        or base_type.__module__.startswith(prefix_with_separator)
        for base_type in type(value).__mro__
    )


class BackendDetector(Protocol):
    """
    Structural interface implemented by all backend detectors
    """

    name: str
    supported_inputs: tuple[str, ...]

    def detect(self, dataframe: object) -> BackendIdentity | None:
        """
        Return backend metadata when supported, otherwise return `None`
        """