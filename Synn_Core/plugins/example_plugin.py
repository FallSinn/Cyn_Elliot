"""Example plugin providing simple canned responses."""
from __future__ import annotations

from typing import Iterable

from . import BasePlugin


class MusicPlugin(BasePlugin):
    """Demonstrates how to extend Synn Core via plugins."""

    name = "music"

    def can_handle(self, text: str, context: Iterable[str]) -> bool:
        lowered = text.lower()
        return "play music" in lowered or "music" in lowered

    async def handle(self, text: str, context: Iterable[str]) -> str:
        return "Starting a virtual playlist for you. (Feature placeholder)"


__all__ = ["MusicPlugin"]
