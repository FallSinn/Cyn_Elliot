"""Speech recognition and text-to-speech module for Synn Core."""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, Optional

try:  # pragma: no cover - optional heavy dependencies
    import vosk  # type: ignore
except Exception:  # pragma: no cover
    vosk = None  # type: ignore

try:  # pragma: no cover
    import sounddevice as sd  # type: ignore
except Exception:  # pragma: no cover
    sd = None  # type: ignore

try:  # pragma: no cover
    import pyttsx3  # type: ignore
except Exception:  # pragma: no cover
    pyttsx3 = None  # type: ignore

try:  # pragma: no cover
    from gtts import gTTS  # type: ignore
except Exception:  # pragma: no cover
    gTTS = None  # type: ignore

LOGGER = logging.getLogger(__name__)


class SpeechInterface:
    """Handles wake-word detection, recognition, and voice synthesis."""

    def __init__(self, config: Dict[str, Any], wake_word: str, loop: Optional[asyncio.AbstractEventLoop] = None) -> None:
        self.config = config
        self.wake_word = wake_word
        self.loop = loop
        self.recognizer_model: Optional[Any] = None
        self.tts_engine: Any = None
        self.voice_backend = config.get("tts_backend", "pyttsx3")
        self.microphone_samplerate = config.get("sample_rate", 16000)
        self.voice_persona = config.get("voice_persona", "neutral")
        self.default_rate = config.get("rate", 175)
        self.default_pitch = config.get("pitch", 50)
        self._stop_listening = False

    def attach_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Attach the running event loop after construction."""

        self.loop = loop

    async def initialize(self) -> None:
        """Load models and initialize synthesis backend."""

        if vosk and self.config.get("recognizer", "vosk").lower() == "vosk":
            model_path = self.config.get("vosk_model_path", "")
            if model_path and Path(model_path).exists():
                self.recognizer_model = vosk.Model(model_path)
                LOGGER.info("Loaded Vosk model from %s", model_path)
            else:
                LOGGER.warning("Vosk model path not provided or missing; defaulting to console input")
        else:
            LOGGER.warning("Speech recognition backend unavailable; falling back to console input")

        if self.voice_backend == "pyttsx3" and pyttsx3:
            self.tts_engine = pyttsx3.init()
            self.tts_engine.setProperty("rate", self.default_rate)
            self.tts_engine.setProperty("pitch", self.default_pitch)  # type: ignore[attr-defined]
            voice = self.config.get("voice_id")
            if voice:
                try:
                    self.tts_engine.setProperty("voice", voice)
                except Exception:  # pragma: no cover - depends on local voices
                    LOGGER.exception("Unable to set requested voice ID")
        elif self.voice_backend == "gtts" and gTTS:
            LOGGER.info("Configured gTTS backend for speech synthesis")
        else:
            LOGGER.warning("No TTS backend available; speech output will be textual only")

    async def listen(self) -> AsyncGenerator[str, None]:
        """Yield transcripts either from microphone or console input."""

        while not self._stop_listening:
            if self.recognizer_model and sd:
                transcript = await self._listen_from_microphone()
            else:
                transcript = await self._listen_from_console()
            yield transcript

    async def speak(self, text: str, tone: float = 0.0, pitch: float = 0.0, speed: float = 1.0, **_: Any) -> None:
        """Synthesize speech with emotional modulation."""

        LOGGER.info("Synn: %s", text)
        if self.voice_backend == "pyttsx3" and self.tts_engine:
            await asyncio.to_thread(self._speak_pyttsx3, text, tone, pitch, speed)
        elif self.voice_backend == "gtts" and gTTS:
            await asyncio.to_thread(self._speak_gtts, text)
        else:
            print(f"Synn says: {text}")

    async def shutdown(self) -> None:
        """Release audio resources."""

        self._stop_listening = True
        if self.tts_engine and self.voice_backend == "pyttsx3":
            await asyncio.to_thread(self.tts_engine.stop)

    async def _listen_from_console(self) -> str:
        """Fallback to console input for development environments."""

        prompt = f"[{self.wake_word}]> "
        text = await asyncio.to_thread(input, prompt)
        return text.strip()

    async def _listen_from_microphone(self) -> str:
        """Capture audio from the microphone and run recognition."""

        assert self.recognizer_model is not None
        assert sd is not None
        recognizer = vosk.KaldiRecognizer(self.recognizer_model, self.microphone_samplerate)
        duration = self.config.get("listen_duration", 4)
        audio = await asyncio.to_thread(sd.rec, int(duration * self.microphone_samplerate), samplerate=self.microphone_samplerate, channels=1, dtype="int16")
        await asyncio.to_thread(sd.wait)
        await asyncio.to_thread(recognizer.AcceptWaveform, audio.tobytes())
        result = recognizer.Result()
        try:
            import json

            parsed = json.loads(result)
            return parsed.get("text", "")
        except Exception:
            LOGGER.exception("Failed to parse recognizer output")
            return result

    def _speak_pyttsx3(self, text: str, tone: float, pitch: float, speed: float) -> None:
        """Speak text via pyttsx3 with modulated parameters."""

        if not self.tts_engine:
            return
        base_rate = self.default_rate
        adjusted_rate = max(80, min(300, int(base_rate * speed)))
        self.tts_engine.setProperty("rate", adjusted_rate)

        try:  # pragma: no cover - depends on pyttsx3 support
            self.tts_engine.setProperty("pitch", self.default_pitch + pitch * 50)
        except Exception:
            LOGGER.debug("Pitch control not supported by backend")

        volume = min(1.0, max(0.2, 0.5 + tone / 2))
        self.tts_engine.setProperty("volume", volume)
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()

    def _speak_gtts(self, text: str) -> None:
        """Speak text using Google Text-to-Speech by generating temporary audio."""

        if not gTTS:
            return
        tts = gTTS(text=text, lang=self.config.get("language", "en"))
        temp_path = Path("tts_output.mp3")
        tts.save(temp_path)
        LOGGER.info("Generated speech audio at %s (playback not automated)", temp_path)


__all__ = ["SpeechInterface"]
