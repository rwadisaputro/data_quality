"""Native physical-schema discovery contracts."""

from data_quality.schema_discovery.base import PhysicalSchemaDiscoverer
from data_quality.schema_discovery.registry import PhysicalSchemaRegistry

__all__ = [
    "PhysicalSchemaDiscoverer", 
    "PhysicalSchemaRegistry"
]