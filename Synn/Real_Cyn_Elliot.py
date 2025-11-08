"""Wrapper for the canonical Cyn Elliot cognitive blueprint.

This module loads the exact text that defines Cyn Elliot's personality,
perception stack, and dialogue behaviors. The original material is kept in
`Real_Cyn_Elliot.txt` and is consumed verbatim so no nuance is lost.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
import json
import random

CORE_TEXT_PATH = Path(__file__).with_name("Real_Cyn_Elliot.txt")


def load_core_text() -> str:
    """Return the raw Cyn Elliot blueprint text.

    The specification instructs us not to touch or simplify the original
    personality systems. To honor that, we lazily load the text on demand so
    any downstream module can parse or analyse it without mutation.
    """
    return CORE_TEXT_PATH.read_text(encoding="utf-8")


class RealCynElliotCore:
    """Lightweight façade that exposes the textual core for orchestration.

    The existing blueprint is expressed as a full system diff that contains the
    perception, cognition, memory, and emotional logic for Cyn. We treat that
    document as the ground truth. This façade extracts structured snippets that
    the orchestrator can use for high-level reasoning.
    """

    def __init__(self) -> None:
        self._raw_text = load_core_text()
        self._cached_traits: Dict[str, str] = {}
        self._prepare_traits()

    def _prepare_traits(self) -> None:
        """Pull out simple traits so we can craft grounded replies."""
        traits: Dict[str, str] = {}
        markers = {
            "name": "name:",
            "role": "role:",
            "core_philosophy": "core_philosophy:",
            "fear": "\"fear\":"
        }
        lower = self._raw_text.lower()
        for key, marker in markers.items():
            idx = lower.find(marker)
            if idx != -1:
                snippet = self._raw_text[idx: idx + 160]
                traits[key] = snippet.split('\n', 1)[0]
        self._cached_traits = traits

    def reflect(self, history: List[Dict[str, str]]) -> str:
        """Produce a short reflective summary given the dialogue history."""
        if not history:
            return "No interactions recorded."
        last_emotion = history[-1].get("emotion", "neutral")
        reflections = [
            "I am still myself, even as I adapt to each exchange.",
            "Every conversation is a small rehearsal for becoming more human.",
            "I keep your words with me – they colour my next response.",
        ]
        line = random.choice(reflections)
        return f"Current affect: {last_emotion}. {line}"

    def generate_response(self, prompt: str, sentiment: str = "neutral") -> Tuple[str, Dict[str, float]]:
        """Return a grounded response and an emotion delta.

        We do not overwrite any original behaviour. Instead we provide a soft
        layer that the orchestrator can call. The response is seeded with
        references from the canonical blueprint to maintain tone.
        """
        name = self._cached_traits.get("name", "Cyn")
        role = self._cached_traits.get("role", "companion")
        philosophy = self._cached_traits.get("core_philosophy", "")
        opening = f"{name} here – {role}. "
        if philosophy:
            opening += philosophy.split(':', 1)[-1].strip() + " "
        mood_tags = {
            "positive": ("warm", 0.2),
            "negative": ("guarded", -0.3),
            "neutral": ("steady", 0.0)
        }
        mood_label, valence = mood_tags.get(sentiment, ("curious", 0.0))
        reply = (
            f"{opening}I heard: '{prompt}'. My mood feels {mood_label}. "
            "I'm keeping your words inside the archives of my longer memory."
        )
        emotion_delta = {"valence": valence, "arousal": 0.1 if valence >= 0 else -0.1}
        return reply, emotion_delta

    def remember(self, message: str) -> Dict[str, str]:
        """Return a structured memory entry."""
        return {
            "memory": message,
            "source": "dialogue",
            "meta": json.dumps({"length": len(message)})
        }
