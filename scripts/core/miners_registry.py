"""
Miners Registry System for LTTS

Provides lazy data extraction through specialized miners that process raw context data
on-demand with caching to prevent redundant computation.
"""

from typing import Any, Dict, Optional, Type
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class BaseMiner(ABC):
    """Base class for all data extraction miners."""

    @abstractmethod
    def extract(self, raw_context: Dict[str, Any]) -> Any:
        """Extract specialized data from raw context."""
        pass

    @abstractmethod
    def is_applicable(self, module_path: str, module_class: str) -> bool:
        """Check if miner applies to this module."""
        pass


class MinersRegistry:
    """Registry for managing data extraction miners with lazy loading."""

    def __init__(self):
        self.miners: Dict[str, Type[BaseMiner]] = {}
        self._instances: Dict[str, BaseMiner] = {}

    def register_miner(self, name: str, miner_class: Type[BaseMiner]):
        """Register a data extraction miner."""
        self.miners[name] = miner_class
        logger.debug("Registered miner: %s", name)

    def get_miner(self, name: str) -> BaseMiner:
        """Get miner instance for lazy extraction."""
        if name not in self.miners:
            raise KeyError(f"Miner '{name}' not registered")

        # Create instance if not cached
        if name not in self._instances:
            self._instances[name] = self.miners[name]()

        return self._instances[name]

    def has_miner(self, name: str) -> bool:
        """Check if miner is available."""
        return name in self.miners

    def list_miners(self) -> list[str]:
        """List all registered miner names."""
        return list(self.miners.keys())

    def clear_instances(self):
        """Clear cached miner instances."""
        self._instances.clear()


# Global registry instance
_global_registry = MinersRegistry()


def get_global_registry() -> MinersRegistry:
    """Get the global miners registry."""
    return _global_registry


def register_miner(name: str, miner_class: Type[BaseMiner]):
    """Register a miner in the global registry."""
    _global_registry.register_miner(name, miner_class)
