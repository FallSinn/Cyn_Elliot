"""Dialogue manager combining rule-based logic with LLM integration."""
from __future__ import annotations

import asyncio
import logging
import random
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:  # pragma: no cover - optional
    import openai  # type: ignore
except Exception:  # pragma: no cover
    openai = None  # type: ignore

try:  # pragma: no cover
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # type: ignore
except Exception:  # pragma: no cover
    SentimentIntensityAnalyzer = None  # type: ignore

from .emotions import EmotionEngine
from .memory.long_term import LongTermMemory, MemoryRecord
from .plugins import SynnPlugin

LOGGER = logging.getLogger(__name__)


class DialogueManager:
    """Processes user input and generates responses."""

    def __init__(self, llm_config: Dict[str, Any], emotion_engine: EmotionEngine, memory: LongTermMemory, plugins: Sequence[SynnPlugin]) -> None:
        self.llm_config = llm_config
        self.emotion_engine = emotion_engine
        self.memory = memory
        self.sentiment_analyzer: Optional[SentimentIntensityAnalyzer] = None
        self.temperature = llm_config.get("temperature", 0.6)
        self.openai_client = None
        self.plugins = list(plugins)

    async def initialize(self) -> None:
        """Lazy-load heavy dependencies like sentiment models or LLM clients."""

        if SentimentIntensityAnalyzer:
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
        else:
            LOGGER.warning("Sentiment analyzer unavailable; emotional cues will be limited")

        if openai and self.llm_config.get("provider") == "openai":
            api_key = self.llm_config.get("api_key")
            if api_key:
                openai.api_key = api_key
                self.openai_client = openai
                LOGGER.info("OpenAI client configured")
            else:
                LOGGER.warning("OpenAI API key missing; defaulting to rule-based responses")

    async def generate_reply(self, user_text: str, context: Iterable[str], memories: Sequence[MemoryRecord], emotional_cues: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Generate a response and metadata for downstream systems."""

        sentiment = self._analyze_sentiment(user_text)
        topic = self._detect_topic(user_text)
        memory_snippets = self._format_memories(memories)
        plugin_reply = await self._process_plugins(user_text, context)

        metadata: Dict[str, Any] = {
            "sentiment": sentiment,
            "topic": topic,
            "memory_references": [m.metadata for m in memories],
            "emotion_shift": self._derive_emotion_shift(sentiment),
            "speech_modifiers": {"speed": 1.0},
        }

        if plugin_reply:
            metadata["source"] = "plugin"
            return plugin_reply, metadata

        knowledge_hits = await self.memory.search_knowledge(user_text)
        reply = None

        if self.openai_client:
            reply = await self._call_openai(user_text, context, memory_snippets, knowledge_hits, emotional_cues)
            metadata["source"] = "openai"
        if not reply:
            reply = self._rule_based_response(user_text, sentiment, topic, memory_snippets, knowledge_hits)
            metadata["source"] = metadata.get("source", "rule")

        metadata["speech_modifiers"] = self._speech_modifiers_from_sentiment(sentiment)
        return reply, metadata

    async def _process_plugins(self, user_text: str, context: Iterable[str]) -> Optional[str]:
        """Pass control to plugins when they can handle the request."""

        for plugin in self.plugins:
            try:
                if plugin.can_handle(user_text, context):
                    return await plugin.handle(user_text, context)
            except Exception:  # pragma: no cover - plugin safety
                LOGGER.exception("Plugin %s failed to handle input", plugin)
        return None

    async def _call_openai(self, user_text: str, context: Iterable[str], memory_snippets: List[str], knowledge_hits: List[Tuple[str, str, float]], emotional_cues: Dict[str, Any]) -> Optional[str]:
        """Invoke OpenAI chat completion asynchronously."""

        if not self.openai_client:
            return None

        prompt = self._build_prompt(user_text, context, memory_snippets, knowledge_hits, emotional_cues)

        try:
            response = await asyncio.to_thread(
                self.openai_client.ChatCompletion.create,
                model=self.llm_config.get("model", "gpt-3.5-turbo"),
                temperature=self.temperature,
                messages=[{"role": "system", "content": prompt["system"]}, {"role": "user", "content": prompt["user"]}],
            )
            content = response["choices"][0]["message"]["content"]
            return content
        except Exception:
            LOGGER.exception("Failed to obtain completion from OpenAI")
            return None

    def _rule_based_response(self, user_text: str, sentiment: Dict[str, float], topic: str, memory_snippets: List[str], knowledge_hits: List[Tuple[str, str, float]]) -> str:
        """Fallback deterministic logic for offline deployments."""

        if "hello" in user_text.lower() or "hi" in user_text.lower():
            return random.choice([
                "Hello! How can I support you today?",
                "Hey there, I'm Synn. What would you like to explore?",
            ])
        if "bye" in user_text.lower():
            return "Goodbye for now. I'll keep our conversation safe."
        if topic == "feelings":
            return "I'm sensing some emotion in your words. Want to talk more about how you're feeling?"

        if knowledge_hits:
            top_topic, content, _ = knowledge_hits[0]
            return f"You once told me about {top_topic}. Here's what I remember: {content}"

        if memory_snippets:
            return f"Earlier we discussed: {' | '.join(memory_snippets)}. Let's build on that."

        if sentiment.get("compound", 0.0) < -0.2:
            return "I'm here with you. Let's take a deep breath together. What can I do to help?"

        return "Tell me more. I'm listening and ready to assist."

    def _analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Return sentiment scores."""

        if self.sentiment_analyzer:
            scores = self.sentiment_analyzer.polarity_scores(text)
            return scores
        return {"compound": 0.0, "pos": 0.0, "neg": 0.0, "neu": 1.0}

    def _detect_topic(self, text: str) -> str:
        """Naive keyword-based topic detection."""

        lowered = text.lower()
        if any(word in lowered for word in ["feel", "emotion", "sad", "happy"]):
            return "feelings"
        if any(word in lowered for word in ["remember", "memory", "recall"]):
            return "memory"
        if any(word in lowered for word in ["face", "camera", "vision"]):
            return "vision"
        if any(word in lowered for word in ["music", "song", "play"]):
            return "music"
        return "general"

    def _format_memories(self, memories: Sequence[MemoryRecord]) -> List[str]:
        """Prepare memory snippets for prompt injection."""

        snippets = []
        for memory in memories:
            snippets.append(f"Previously you said: {memory.user_text}")
        return snippets

    def _speech_modifiers_from_sentiment(self, sentiment: Dict[str, float]) -> Dict[str, float]:
        """Translate sentiment score into voice adjustments."""

        compound = sentiment.get("compound", 0.0)
        return {
            "speed": 1.0 + compound * 0.1,
            "tone": compound,
        }

    def _derive_emotion_shift(self, sentiment: Dict[str, float]) -> Dict[str, float]:
        """Generate emotion deltas from sentiment."""

        compound = sentiment.get("compound", 0.0)
        return {"valence_delta": compound * 0.2, "arousal_delta": compound * 0.1}

    def _build_prompt(self, user_text: str, context: Iterable[str], memory_snippets: List[str], knowledge_hits: List[Tuple[str, str, float]], emotional_cues: Dict[str, Any]) -> Dict[str, str]:
        """Compose prompt payload for LLM calls."""

        system_prompt = self.llm_config.get(
            "system_prompt",
            "You are Synn, an empathetic AI that leverages stored memories, emotions, and perception cues to respond helpfully.",
        )
        combined_context = "\n".join(list(context))
        memory_section = "\n".join(memory_snippets)
        knowledge_section = "\n".join([f"Topic: {topic}\nContent: {content}" for topic, content, _ in knowledge_hits])
        emotion_desc = ", ".join(f"{k}: {v}" for k, v in emotional_cues.items())

        user_prompt = (
            f"Context:\n{combined_context}\n\n"
            f"Memories:\n{memory_section}\n\n"
            f"Knowledge:\n{knowledge_section}\n\n"
            f"Perception cues:\n{emotion_desc}\n\n"
            f"User says: {user_text}"
        )

        return {"system": system_prompt, "user": user_prompt}


__all__ = ["DialogueManager"]
