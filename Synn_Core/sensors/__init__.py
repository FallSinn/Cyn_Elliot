"""Sensor abstractions for Synn Core."""

from .camera import CameraSensor
from .mic import MicrophoneMonitor
from .touch import TouchSensor

__all__ = ["CameraSensor", "MicrophoneMonitor", "TouchSensor"]
