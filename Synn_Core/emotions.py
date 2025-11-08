"""Emotion engine implementing a valence-arousal model."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class EmotionState:
    """Represents the assistant's emotional configuration."""

    name: str
    valence: float  # [-1, 1]
    arousal: float  # [-1, 1]


class EmotionEngine:
    """Computes emotional states and exposes modulation cues."""

    def __init__(self) -> None:
        self.states: Dict[str, EmotionState] = {
            "neutral": EmotionState("neutral", 0.0, 0.0),
            "calm": EmotionState("calm", 0.3, -0.4),
            "happy": EmotionState("happy", 0.8, 0.4),
            "sad": EmotionState("sad", -0.6, -0.5),
            "angry": EmotionState("angry", -0.8, 0.7),
        }
        self.current_state: EmotionState = self.states["neutral"]

    def update_from_perception(self, face_emotion: Optional[Dict[str, float]], audio_level: Optional[float]) -> None:
        """Update internal state based on perception cues."""

        if face_emotion:
            dominant = max(face_emotion, key=face_emotion.get)
            confidence = face_emotion[dominant]
            if confidence > 0.6 and dominant in self.states:
                self.current_state = self.states[dominant]
        if audio_level is not None:
            if audio_level > 0.7:
                self.current_state = self.states.get("angry", self.current_state)
            elif audio_level < 0.3:
                self.current_state = self.states.get("calm", self.current_state)

    def apply_shift(self, valence_delta: float = 0.0, arousal_delta: float = 0.0, target: Optional[str] = None) -> None:
        """Adjust emotional state by delta or explicit target."""

        if target and target in self.states:
            self.current_state = self.states[target]
            return

        new_valence = max(-1.0, min(1.0, self.current_state.valence + valence_delta))
        new_arousal = max(-1.0, min(1.0, self.current_state.arousal + arousal_delta))
        self.current_state = EmotionState("dynamic", new_valence, new_arousal)

    def get_speech_modifiers(self) -> Dict[str, float]:
        """Map emotional state to speech synthesis parameters."""

        return {
            "tone": self.current_state.valence,
            "pitch": self.current_state.arousal * 0.3,
            "speed": 1.0 + self.current_state.arousal * 0.25,
        }

    def describe(self) -> str:
        """Return human-readable description of the current emotional state."""

        return f"Emotion: {self.current_state.name} (valence={self.current_state.valence:.2f}, arousal={self.current_state.arousal:.2f})"


__all__ = ["EmotionEngine", "EmotionState"]
