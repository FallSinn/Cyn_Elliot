"""Main orchestrator for the Synn Core assistant."""
from __future__ import annotations

# Standard library imports
import asyncio
import logging
import queue
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

# Third-party imports guarded to keep module importable even if optional deps are missing
try:  # pragma: no cover - optional dependency
    import uvicorn
except Exception:  # pragma: no cover
    uvicorn = None  # type: ignore

# Local imports
from .speech import SpeechInterface
from .memory.short_term import ShortTermMemory
from .memory.long_term import LongTermMemory
from .emotions import EmotionEngine
from .dialogue_manager import DialogueManager
from .sensors.camera import CameraSensor
from .sensors.mic import MicrophoneMonitor
from .sensors.touch import TouchSensor
from .reflection import ReflectionManager
from .gui import SynnConsoleGUI
from .api import create_app
from .plugins import load_plugins


@dataclass
class SynnConfig:
    """Dataclass representing configuration parameters for Synn Core."""

    wake_word: str
    speech: Dict[str, Any]
    llm: Dict[str, Any]
    memory: Dict[str, Any]
    logging_dir: str
    api: Dict[str, Any]
    gui: Dict[str, Any]


class SynnCore:
    """Coordinates perception, dialogue, memory, emotions, and interfaces."""

    def __init__(self, config: SynnConfig) -> None:
        self.config = config
        self._setup_logging()
        self.logger = logging.getLogger("SynnCore")
        self.event_loop: Optional[asyncio.AbstractEventLoop] = None
        self.input_queue: Optional[asyncio.Queue[str]] = None
        self.output_queue: Optional[asyncio.Queue[str]] = None
        self.shutdown_event: Optional[asyncio.Event] = None

        # Initialize subsystems
        self.speech = SpeechInterface(config.speech, wake_word=config.wake_word)
        self.short_term_memory = ShortTermMemory(max_items=config.memory.get("short_term_limit", 10))
        self.long_term_memory = LongTermMemory(db_path=config.memory.get("long_term_path", "memory/long_term.db"))
        self.emotion_engine = EmotionEngine()
        self.camera_sensor = CameraSensor(enabled=config.api.get("enable_vision", True))
        self.microphone_monitor = MicrophoneMonitor(enabled=config.api.get("enable_audio_monitor", True))
        self.touch_sensor = TouchSensor()
        self.reflection_manager = ReflectionManager(self.emotion_engine, self.long_term_memory)
        self.gui: Optional[SynnConsoleGUI] = None

        # Plugin registry
        self.plugins = load_plugins(Path(config.memory.get("plugin_dir", "Synn_Core/plugins")))

        self.dialogue_manager = DialogueManager(config.llm, self.emotion_engine, self.long_term_memory, self.plugins)

        self.logger.info("Synn Core initialized with wake word '%s'", config.wake_word)

    def _setup_logging(self) -> None:
        """Configure application-wide logging handlers."""

        Path(self.config.logging_dir).mkdir(parents=True, exist_ok=True)
        log_path = Path(self.config.logging_dir) / f"session_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.txt"
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            handlers=[logging.FileHandler(log_path), logging.StreamHandler()]
        )

    async def start(self) -> None:
        """Start the assistant by booting all asynchronous tasks."""

        self.logger.info("Starting Synn Core event loop")
        self.event_loop = asyncio.get_running_loop()
        self.input_queue = asyncio.Queue()
        self.output_queue = asyncio.Queue()
        self.shutdown_event = asyncio.Event()

        self.speech.attach_loop(self.event_loop)
        await self.speech.initialize()
        await self.long_term_memory.initialize()
        await self.dialogue_manager.initialize()
        await self.camera_sensor.initialize()
        await self.microphone_monitor.initialize()

        # Start GUI if requested
        if self.config.gui.get("enabled", True):
            self._start_gui()

        # Launch FastAPI server in background thread if available
        if uvicorn and self.config.api.get("enabled", True):
            self._start_api_server()

        # Launch concurrent tasks
        tasks = [
            asyncio.create_task(self._speech_listener()),
            asyncio.create_task(self._dialogue_worker()),
            asyncio.create_task(self._sensor_monitor()),
        ]

        self.logger.info("Synn Core is now running")
        await self.shutdown_event.wait()
        self.logger.info("Shutdown signal received")

        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

        await self._shutdown()

    async def _shutdown(self) -> None:
        """Flush memory, store reflections, and close resources."""

        self.logger.info("Performing graceful shutdown")
        await self.reflection_manager.summarize_session()
        await self.camera_sensor.shutdown()
        await self.microphone_monitor.shutdown()
        await self.speech.shutdown()
        await self.long_term_memory.shutdown()
        if self.gui:
            self.gui.stop()

    def _start_gui(self) -> None:
        """Create and run the Tkinter UI in a dedicated thread."""

        self.logger.debug("Booting GUI interface")
        gui_queue: "queue.Queue[str]" = queue.Queue()

        def on_user_input(text: str) -> None:
            if not self.event_loop or not self.input_queue:
                return
            self.event_loop.call_soon_threadsafe(lambda: self.event_loop.create_task(self.input_queue.put(text)))

        self.gui = SynnConsoleGUI(on_user_input=on_user_input, output_queue=gui_queue, title=self.config.gui.get("title", "Synn Console"))

        def gui_updater() -> None:
            if not self.shutdown_event:
                return
            while not self.shutdown_event.is_set():
                try:
                    message = gui_queue.get(timeout=0.1)
                    if self.gui:
                        self.gui.display_response(message)
                except queue.Empty:
                    continue

        threading.Thread(target=self.gui.run, daemon=True).start()
        threading.Thread(target=gui_updater, daemon=True).start()

        async def forward_output() -> None:
            if not self.shutdown_event or not self.output_queue:
                return
            while not self.shutdown_event.is_set():
                response = await self.output_queue.get()
                gui_queue.put(response)

        if self.event_loop:
            self.event_loop.create_task(forward_output())

    def _start_api_server(self) -> None:
        """Launch FastAPI service for remote control."""

        app = create_app(self)

        def run_server() -> None:
            if not uvicorn:
                return
            config = uvicorn.Config(app, host=self.config.api.get("host", "0.0.0.0"), port=self.config.api.get("port", 8000), log_level="info")
            server = uvicorn.Server(config)
            self.logger.info("Starting API server on %s:%s", config.host, config.port)
            if self.event_loop:
                asyncio.run_coroutine_threadsafe(self._announce_start("API server running"), self.event_loop)
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(server.serve())
            loop.close()

        threading.Thread(target=run_server, daemon=True).start()

    async def _speech_listener(self) -> None:
        """Listen for speech input, filter by wake word, and enqueue utterances."""

        async for transcript in self.speech.listen():
            if not transcript:
                continue
            self.logger.debug("Transcript received: %s", transcript)
            if not transcript.lower().startswith(self.config.wake_word.lower()):
                self.logger.debug("Ignoring transcript without wake word")
                continue
            stripped = transcript[len(self.config.wake_word):].strip()
            if self.input_queue:
                await self.input_queue.put(stripped)

    async def _dialogue_worker(self) -> None:
        """Process user utterances, update memory/emotion, and output responses."""

        if not self.shutdown_event or not self.input_queue:
            return
        while not self.shutdown_event.is_set():
            user_text = await self.input_queue.get()
            self.logger.info("User: %s", user_text)

            context = self.short_term_memory.get_context()
            memory_hits = await self.long_term_memory.search(user_text)
            emotional_cues = self.camera_sensor.get_last_emotion() or {}

            reply, metadata = await self.dialogue_manager.generate_reply(user_text, context, memory_hits, emotional_cues)

            self.short_term_memory.add_entry(user_text, reply)
            await self.long_term_memory.store_interaction(user_text, reply, metadata)

            await self._handle_emotion(metadata)
            await self._speak(reply, metadata)
            if self.output_queue:
                await self.output_queue.put(reply)

    async def _sensor_monitor(self) -> None:
        """Poll sensors and update emotional state accordingly."""

        if not self.shutdown_event:
            return
        while not self.shutdown_event.is_set():
            await asyncio.sleep(1.5)
            face_emotion = await self.camera_sensor.capture()
            audio_level = await self.microphone_monitor.get_level()
            self.emotion_engine.update_from_perception(face_emotion, audio_level)
            self.reflection_manager.record_emotion()

    async def _speak(self, text: str, metadata: Dict[str, Any]) -> None:
        """Send output to TTS while respecting emotional modulation."""

        speech_params = self.emotion_engine.get_speech_modifiers()
        speech_params.update(metadata.get("speech_modifiers", {}))
        await self.speech.speak(text, **speech_params)

    async def _handle_emotion(self, metadata: Dict[str, Any]) -> None:
        """Adjust emotion engine based on dialogue metadata."""

        emotion_shift = metadata.get("emotion_shift")
        if emotion_shift:
            self.emotion_engine.apply_shift(**emotion_shift)
            self.reflection_manager.record_emotion()

    async def _announce_start(self, message: str) -> None:
        """Utility to inform user about background service startup."""

        if self.output_queue:
            await self.output_queue.put(message)

    def request_shutdown(self) -> None:
        """Signal the assistant to stop running."""
        if self.event_loop and self.shutdown_event:
            self.event_loop.call_soon_threadsafe(self.shutdown_event.set)


__all__ = ["SynnCore", "SynnConfig"]
