"""Microphone monitoring utilities."""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

try:  # pragma: no cover - optional
    import sounddevice as sd  # type: ignore
    import numpy as np
except Exception:  # pragma: no cover
    sd = None  # type: ignore
    np = None  # type: ignore

LOGGER = logging.getLogger(__name__)


class MicrophoneMonitor:
    """Measures audio amplitude for emotional inference."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled and sd is not None and np is not None
        self.level: float = 0.0

    async def initialize(self) -> None:
        """Placeholder for compatibility."""

        if not self.enabled:
            LOGGER.warning("Microphone monitor disabled (missing dependencies or configuration)")

    async def get_level(self) -> Optional[float]:
        """Return normalized audio energy level."""

        if not self.enabled:
            return None

        duration = 0.5
        sample_rate = 16000
        recording = await asyncio.to_thread(sd.rec, int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="float32")
        await asyncio.to_thread(sd.wait)
        energy = float(np.linalg.norm(recording) / len(recording))  # type: ignore[operator]
        self.level = min(1.0, energy * 10)
        return self.level

    async def shutdown(self) -> None:
        """No cleanup required for passive monitor."""

        return None


__all__ = ["MicrophoneMonitor"]
