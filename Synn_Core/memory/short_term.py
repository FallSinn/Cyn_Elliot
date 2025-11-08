"""Short-term memory management for Synn Core."""
from __future__ import annotations

from collections import deque
from typing import Deque, List, Tuple


class ShortTermMemory:
    """Circular buffer storing recent conversation turns."""

    def __init__(self, max_items: int = 10) -> None:
        self.max_items = max_items
        self.buffer: Deque[Tuple[str, str]] = deque(maxlen=max_items)

    def add_entry(self, user: str, assistant: str) -> None:
        """Append a dialogue turn to the short-term memory."""

        self.buffer.append((user, assistant))

    def get_context(self) -> List[str]:
        """Return the context as alternating user/assistant lines."""

        context: List[str] = []
        for user, assistant in self.buffer:
            context.append(f"User: {user}")
            context.append(f"Synn: {assistant}")
        return context

    def clear(self) -> None:
        """Clear the stored context."""

        self.buffer.clear()


__all__ = ["ShortTermMemory"]
