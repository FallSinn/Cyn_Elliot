"""Reflection module for summarizing sessions."""
from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from .emotions import EmotionEngine
from .memory.long_term import LongTermMemory


class ReflectionManager:
    """Produces end-of-session summaries and logs emotional shifts."""

    def __init__(self, emotions: EmotionEngine, memory: LongTermMemory, log_path: str = "memory/reflection.log") -> None:
        self.emotions = emotions
        self.memory = memory
        self.log_file = Path(log_path)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.emotion_history: List[str] = []
        self.learned_preferences: Dict[str, str] = {}

    def record_emotion(self) -> None:
        """Capture current emotional state for later review."""

        self.emotion_history.append(self.emotions.describe())

    def record_preference(self, key: str, value: str) -> None:
        """Persist a discovered user preference."""

        self.learned_preferences[key] = value

    async def summarize_session(self) -> None:
        """Write a structured summary to disk when the assistant shuts down."""

        summary = await self.memory.summarize()
        payload = (
            f"=== Reflection {datetime.utcnow().isoformat()} ===\n"
            f"Recent emotions:\n" + "\n".join(self.emotion_history[-10:]) + "\n"
            f"Preferences:\n" + "\n".join(f"- {k}: {v}" for k, v in self.learned_preferences.items()) + "\n"
            f"Conversation snapshot:\n{summary}\n\n"
        )
        await asyncio.to_thread(self._append_log, payload)

    def _append_log(self, content: str) -> None:
        """Append content to the reflection log."""

        with self.log_file.open("a", encoding="utf-8") as fh:
            fh.write(content)


__all__ = ["ReflectionManager"]
