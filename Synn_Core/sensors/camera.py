"""Camera-based perception for Synn Core."""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional

try:  # pragma: no cover - optional
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore

LOGGER = logging.getLogger(__name__)


class CameraSensor:
    """Captures facial expressions to influence emotional state."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled and cv2 is not None
        self.capture_device: Optional["cv2.VideoCapture"] = None
        self.face_cascade: Optional[Any] = None
        self.last_emotion: Optional[Dict[str, float]] = None

    async def initialize(self) -> None:
        """Initialize camera device and load detection models."""

        if not self.enabled:
            LOGGER.warning("Camera sensor disabled or OpenCV missing")
            return
        self.capture_device = cv2.VideoCapture(0)
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        LOGGER.info("Camera sensor initialized using %s", cascade_path)

    async def capture(self) -> Optional[Dict[str, float]]:
        """Capture a frame and infer an approximate emotion."""

        if not self.enabled or not self.capture_device or not self.face_cascade:
            return None
        frame_ready, frame = await asyncio.to_thread(self.capture_device.read)
        if not frame_ready:
            return None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        emotion = {"neutral": 0.5}
        if len(faces) > 0:
            emotion = {"happy": 0.6}
        self.last_emotion = emotion
        return emotion

    def get_last_emotion(self) -> Optional[Dict[str, float]]:
        """Return the most recent emotion estimate."""

        return self.last_emotion

    async def shutdown(self) -> None:
        """Release camera resources."""

        if self.capture_device:
            await asyncio.to_thread(self.capture_device.release)
            self.capture_device = None


__all__ = ["CameraSensor"]
