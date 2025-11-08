"""Plugin discovery utilities for Synn Core."""
from __future__ import annotations

import importlib.util
import inspect
import logging
from pathlib import Path
from typing import Iterable, List, Protocol

LOGGER = logging.getLogger(__name__)


class SynnPlugin(Protocol):
    """Protocol that plugins must implement."""

    name: str

    def can_handle(self, text: str, context: Iterable[str]) -> bool:
        ...

    async def handle(self, text: str, context: Iterable[str]) -> str:
        ...


def load_plugins(directory: Path) -> List[SynnPlugin]:
    """Dynamically discover plugins in the given directory."""

    plugins: List[SynnPlugin] = []
    if not directory.exists():
        LOGGER.info("Plugin directory %s does not exist", directory)
        return plugins

    for path in directory.glob("*.py"):
        if path.name.startswith("__"):
            continue
        module_name = f"Synn_Core.plugins.{path.stem}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if not spec or not spec.loader:
            continue
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)  # type: ignore[arg-type]
        except Exception:  # pragma: no cover - plugin load failure
            LOGGER.exception("Failed to load plugin %s", path)
            continue

        for _, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, BasePlugin) and obj is not BasePlugin:
                try:
                    instance = obj()
                    plugins.append(instance)
                    LOGGER.info("Loaded plugin: %s", instance.name)
                except Exception:  # pragma: no cover - plugin init failure
                    LOGGER.exception("Failed to instantiate plugin %s", obj.__name__)
    return plugins


class BasePlugin:
    """Convenience base class for plugin authors."""

    name = "base"

    def can_handle(self, text: str, context: Iterable[str]) -> bool:
        return False

    async def handle(self, text: str, context: Iterable[str]) -> str:
        raise NotImplementedError


__all__ = ["SynnPlugin", "BasePlugin", "load_plugins"]
