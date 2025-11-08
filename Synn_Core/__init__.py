"""Top-level package for the Synn Core modular AI assistant."""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("synn-core")
except PackageNotFoundError:  # pragma: no cover - package metadata not available during development
    __version__ = "0.1.0"

__all__ = ["__version__"]
