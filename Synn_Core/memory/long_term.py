"""Long-term memory with semantic search backed by SQLite."""
from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:  # pragma: no cover - optional heavy dependency
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:  # pragma: no cover
    SentenceTransformer = None  # type: ignore

LOGGER = logging.getLogger(__name__)


@dataclass
class MemoryRecord:
    """Representation of a retrieved long-term memory."""

    user_text: str
    assistant_text: str
    metadata: Dict[str, Any]
    similarity: float


class LongTermMemory:
    """Stores conversations and retrieves them using vector similarity."""

    def __init__(self, db_path: str = "memory/long_term.db", model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.db_path = Path(db_path)
        self.model_name = model_name
        self.connection: Optional[sqlite3.Connection] = None
        self.embedding_model: Optional[Any] = None
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    async def initialize(self) -> None:
        """Open database connection and load embedding model."""

        self.connection = sqlite3.connect(self.db_path)
        await asyncio.to_thread(self._create_tables)
        if SentenceTransformer:
            self.embedding_model = await asyncio.to_thread(SentenceTransformer, self.model_name)
            LOGGER.info("Loaded embedding model %s", self.model_name)
        else:
            LOGGER.warning("sentence-transformers not available; semantic search disabled")

    async def shutdown(self) -> None:
        """Close the database connection."""

        if self.connection:
            await asyncio.to_thread(self.connection.close)
            self.connection = None

    async def store_interaction(self, user_text: str, assistant_text: str, metadata: Dict[str, Any]) -> None:
        """Persist a conversation turn with embedding."""

        if not self.connection:
            raise RuntimeError("LongTermMemory not initialized")

        embedding = await self._embed_text(f"User: {user_text}\nSynn: {assistant_text}")
        payload = json.dumps(metadata)
        timestamp = datetime.utcnow().isoformat()

        await asyncio.to_thread(
            self.connection.execute,
            "INSERT INTO interactions (user_text, assistant_text, metadata, embedding, timestamp) VALUES (?, ?, ?, ?, ?)",
            (user_text, assistant_text, payload, embedding, timestamp),
        )
        await asyncio.to_thread(self.connection.commit)

    async def search(self, query: str, limit: int = 5) -> List[MemoryRecord]:
        """Retrieve most relevant memories matching the query."""

        if not self.connection or not self.embedding_model:
            return []

        query_vec = await self._embed_text(query)
        rows = await asyncio.to_thread(
            self._fetch_all,
            "SELECT user_text, assistant_text, metadata, embedding FROM interactions",
        )

        memories: List[MemoryRecord] = []
        for user_text, assistant_text, metadata_json, blob in rows:
            if not blob:
                continue
            stored_vec = np.frombuffer(blob, dtype=np.float32)
            similarity = self._cosine_similarity(query_vec, stored_vec)
            metadata = json.loads(metadata_json)
            memories.append(MemoryRecord(user_text, assistant_text, metadata, similarity))

        memories.sort(key=lambda item: item.similarity, reverse=True)
        return memories[:limit]

    async def summarize(self) -> str:
        """Produce a lightweight summary of stored interactions."""

        if not self.connection:
            return ""
        rows = await asyncio.to_thread(
            self._fetch_all,
            "SELECT user_text, assistant_text FROM interactions ORDER BY id DESC LIMIT 20",
        )
        summary_lines = [f"User: {u}\nSynn: {a}" for u, a in rows]
        return "\n".join(summary_lines)

    async def add_knowledge(self, topic: str, content: str) -> None:
        """Store long-term knowledge snippets with embeddings."""

        if not self.connection:
            raise RuntimeError("LongTermMemory not initialized")

        embedding = await self._embed_text(f"{topic}\n{content}")
        await asyncio.to_thread(
            self.connection.execute,
            "INSERT INTO knowledge (topic, content, embedding, timestamp) VALUES (?, ?, ?, ?)",
            (topic, content, embedding, datetime.utcnow().isoformat()),
        )
        await asyncio.to_thread(self.connection.commit)

    async def search_knowledge(self, query: str, limit: int = 3) -> List[Tuple[str, str, float]]:
        """Search the knowledge base for relevant content."""

        if not self.connection or not self.embedding_model:
            return []

        query_vec = await self._embed_text(query)
        rows = await asyncio.to_thread(
            self._fetch_all,
            "SELECT topic, content, embedding FROM knowledge",
        )

        results: List[Tuple[str, str, float]] = []
        for topic, content, blob in rows:
            if not blob:
                continue
            stored_vec = np.frombuffer(blob, dtype=np.float32)
            similarity = self._cosine_similarity(query_vec, stored_vec)
            results.append((topic, content, similarity))

        results.sort(key=lambda item: item[2], reverse=True)
        return results[:limit]

    async def _embed_text(self, text: str) -> bytes:
        """Convert text into an embedding vector serialized for SQLite."""

        if not self.embedding_model:
            return text.encode("utf-8")
        vector = await asyncio.to_thread(self.embedding_model.encode, text)
        array = np.asarray(vector, dtype=np.float32)
        return array.tobytes()

    def _cosine_similarity(self, a_bytes: bytes, b: np.ndarray) -> float:
        """Compute cosine similarity between serialized and in-memory vectors."""

        a = np.frombuffer(a_bytes, dtype=np.float32)
        if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
            return 0.0
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def _create_tables(self) -> None:
        """Create required SQLite tables if they do not exist."""

        if not self.connection:
            raise RuntimeError("Database connection missing")
        cursor = self.connection.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_text TEXT,
                assistant_text TEXT,
                metadata TEXT,
                embedding BLOB,
                timestamp TEXT
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS knowledge (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic TEXT,
                content TEXT,
                embedding BLOB,
                timestamp TEXT
            )
            """
        )
        self.connection.commit()

    def _fetch_all(self, query: str) -> List[Tuple[Any, ...]]:
        """Execute a SELECT query and return all rows."""

        if not self.connection:
            return []
        cursor = self.connection.execute(query)
        return cursor.fetchall()


__all__ = ["LongTermMemory", "MemoryRecord"]
